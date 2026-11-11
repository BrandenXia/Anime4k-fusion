"""Generate Metal kernels for the pinned medium 2x Anime4K shader."""
import re
import sys
from pathlib import Path
from decimal import Decimal

root = Path(__file__).resolve().parent
source = (root / "Anime4K_Upscale_CNN_x2_M.glsl").read_text()
passes = source.split("//!DESC")[1:]
assert len(passes) == 9
parts = [source.split("//!DESC")[0], """
#include <metal_stdlib>
using namespace metal;
using Src = texture2d<half, access::sample>;
using Dst = texture2d<half, access::write>;
constexpr sampler point(coord::pixel, address::clamp_to_edge, filter::nearest);
constexpr sampler linearFilter(coord::pixel, address::clamp_to_edge, filter::linear);
inline bool inside(uint2 p, Src s) {
    return p.x < s.get_width() && p.y < s.get_height();
}
inline float4 fetch(Src s, int2 p) {
    return float4(s.sample(point, float2(p) + 0.5f));
}
// Each low-resolution invocation writes the same 2x2 group in every schedule.
inline void upscale(Src rgb, Dst dst, uint2 p, half4 residual) {
    for (uint y = 0; y < 2; ++y) {
        for (uint x = 0; x < 2; ++x) {
            uint2 out = 2 * p + uint2(x,y);
            float4 base = float4(rgb.sample(linearFilter, (float2(out) + 0.5f) * 0.5f));
            dst.write(half4(base + float(residual[2*y+x])), out);
        }
    }
}
"""]
def emit(block):
    parts.append(block)

def kernel(name, textures, body, tiled=False):
    params = [f"{typ} {var} [[texture({i})]]" for i, (typ, var) in enumerate(textures)]
    params += ["uint2 p [[thread_position_in_grid]]"]
    if tiled:
        params += ["uint2 lid [[thread_position_in_threadgroup]]",
                   "uint lane [[thread_index_in_threadgroup]]",
                   "uint2 group [[threadgroup_position_in_grid]]"]
    emit("kernel void " + name + "(\n    " + ",\n    ".join(params) + ") {\n" + body + "\n}")

offsets = [(x, y) for x in (-1,0,1) for y in (-1,0,1)]
def args(index, values):
    return f"layer{index}(" + ", ".join(values) + ")"

for i, block in enumerate(passes[:8]):
    params = [f"float4 v{j}" for j in range(7 if i == 7 else 9)]
    body = []
    terms = re.findall(r"mat4\(([^)]+)\) \* (go_[01]\([^)]+\)|g_\d+);", block)
    assert len(terms) == (9 if i == 0 else 14 if i == 7 else 18)
    for n, (coeffs, operand) in enumerate(terms):
        weights = coeffs.split(", ")
        assert len(weights) == 16
        columns = ["float4(" + ", ".join(weights[k:k+4]) + ")" for k in range(0,16,4)]
        if i == 7:
            channel = int(operand[2:])
            j, neg = divmod(channel, 2)
        else:
            sign, x, y = re.fullmatch(r"go_(\d)\(([^,]+), (.+)\)", operand).groups()
            j, neg = offsets.index((int(float(x)), int(float(y)))), int(sign)
        value = f"v{j}" if i == 0 else f"max({'-' if neg else ''}v{j}, 0.0f)"
        prefix = "float4 sum = " if n == 0 else "sum += "
        body.append(prefix + "float4x4(" + ", ".join(columns) + ") * " + value + ";")
    bias = re.search(r"result \+= vec4\(([^)]+)\);", block)[1]
    body.append(f"return half4(sum + float4({bias}));")
    emit(f"inline half4 layer{i}(" + ", ".join(params) + ") {\n    " + "\n    ".join(body) + "\n}")
    if i > 0:
        # P*max(x,0) + N*max(-x,0) = P*x + (P+N)*max(-x,0).
        # Fold P+N offline. Preserve the upstream term order.
        algebra = []
        for n, (coeffs, operand) in enumerate(terms):
            positive = n - 1 if i == 7 else n - 9
            j, neg = divmod(n, 2) if i == 7 else (n % 9, n >= 9)
            weights = [Decimal(w) for w in coeffs.split(", ")]
            if neg:
                weights = [w + Decimal(v) for w,v in zip(weights,terms[positive][0].split(", "))]
            columns = ["float4(" + ", ".join(str(w) for w in weights[k:k+4]) + ")" for k in range(0,16,4)]
            value = f"max(-v{j}, 0.0f)" if neg else f"v{j}"
            prefix = "float4 sum = " if n == 0 else "sum += "
            algebra.append(prefix + "float4x4(" + ", ".join(columns) + ") * " + value + ";")
        algebra.append(body[-1])
        emit(f"inline half4 algebra{i}(" + ", ".join(params) + ") {\n    " + "\n    ".join(algebra) + "\n}")
    if i < 7:
        values = [f"fetch(src, p + int2({x},{y}))" for x,y in offsets]
        emit(f"inline half4 conv{i}(Src src, int2 p) {{ return {args(i, values)}; }}")
        if i:
            emit(f"inline half4 rewritten{i}(Src src, int2 p) {{ return algebra{i}(" + ", ".join(values) + "); }")
            kernel(f"mathSingle{i}", [("Src","src"),("Dst","dst")],
                   f"    if (inside(p,src)) dst.write(rewritten{i}(src, int2(p)),p);")
        kernel(f"single{i}", [("Src","src"),("Dst","dst")],
               f"    if (inside(p,src)) dst.write(conv{i}(src, int2(p)),p);")

features = [("Src", f"f{i}") for i in range(7)]
feature_reads = [f"float4(f{i}.read(p))" for i in range(7)]
head = args(7, feature_reads)
kernel("head", features+[("Dst","dst")], f"    if (inside(p,f0)) dst.write({head},p);")
kernel("shuffle", [("Src","rgb"),("Src","residual"),("Dst","dst")],
       "    if (inside(p,rgb)) upscale(rgb,dst,p,residual.read(p));")
kernel("headShuffle", features+[("Src","rgb"),("Dst","dst")],
       f"    if (inside(p,rgb)) upscale(rgb,dst,p,{head});")
tail = args(7, feature_reads[:6]+["float4(conv6(f5,int2(p)))"])
kernel("tail", features[:6]+[("Src","rgb"),("Dst","dst")],
       f"    if (inside(p,rgb)) upscale(rgb,dst,p,{tail});")

rewritten_tail = tail.replace("layer7","algebra7").replace("conv6","rewritten6")
kernel("mathTail", features[:6]+[("Src","rgb"),("Dst","dst")],
       f"    if (inside(p,rgb)) upscale(rgb,dst,p,{rewritten_tail});")

# Each tile computes its own one-pixel halo. The head needs both layers'
# features, so interior first-layer values must also be saved to a texture.
for i in (0,2,4):
    tile_reads = [f"float4(tile[(lid.y+{y+1})*18+lid.x+{x+1}])" for x,y in offsets]
    second = args(i+1,tile_reads)
    kernel(f"pair{i}", [("Src","src"),("Dst","first"),("Dst","second")], f"""
    threadgroup half4 tile[18*10];
    int2 limit = int2(src.get_width(), src.get_height()) - 1;
    for (uint k = lane; k < 18*10; k += 16*8) {{
        int2 q = int2(group * uint2(16,8)) + int2(k%18,k/18) - 1;
        tile[k] = conv{i}(src,clamp(q,int2(0),limit));
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (inside(p,src)) {{
        first.write(tile[(lid.y+1)*18+lid.x+1],p);
        second.write({second},p);
    }}""", tiled=True)
Path(sys.argv[1]).write_text("\n".join(parts))
