"""Generate OpenGL 4.1 fragment fusion from Anime4K's original GLSL shaders."""
import argparse
import re
from pathlib import Path

ROUNDING = """
// Emulate the intermediate RGBA16F texture rounding using GLSL 4.1 operations.
float round_half(float value) {
    uint bits = floatBitsToUint(value);
    uint sign = bits & 0x80000000u;
    uint magnitude = bits & 0x7fffffffu;
    if (magnitude < 0x38800000u) {
        return sign == 0u ? roundEven(abs(value) * 16777216.0) / 16777216.0
                          : -roundEven(abs(value) * 16777216.0) / 16777216.0;
    }
    if (magnitude >= 0x7f800000u) return value;
    magnitude = (magnitude + 0xfffu + ((magnitude >> 13u) & 1u)) & 0xffffe000u;
    return uintBitsToFloat(sign | (magnitude >= 0x47800000u ? 0x7f800000u : magnitude));
}
vec4 round_feature(vec4 value) {
    return vec4(round_half(value.x), round_half(value.y),
                round_half(value.z), round_half(value.w));
}
"""

def fields(block, name):
    return re.findall(r'^//!' + name + r' (.+)$', block, re.M)

def code(block):
    start = block.index('#define')
    macros = block[start:block.index('vec4 hook()')].strip()
    body = block.split('vec4 hook() {')[1].split('return result')[0].strip()
    return macros, body

def same_resolution(name, original, restore):
    header, *passes = original.split('//!DESC')
    # Medium final 3x3 + 1x1 projection. VL restoration has two 3x3 outputs.
    count = 3 if name == 'Restore_CNN_VL' else 2
    end = len(passes) if restore else len(passes)-1
    tail = passes[end-count:end]
    inputs, produced = [], {}
    for block in tail:
        for texture in fields(block, 'BIND'):
            if texture not in produced and texture not in inputs:
                inputs.append(texture)
        produced[fields(block, 'SAVE')[0]] = True
    final = tail[-1]
    output = fields(final,'SAVE')[0]
    text = [f'//!DESC Anime4K {name} OpenGL fused convolution and projection','//!HOOK MAIN']
    text += ['//!BIND '+v for v in inputs]
    text += [line for line in final.splitlines()
             if line.startswith(('//!SAVE','//!WIDTH','//!HEIGHT','//!COMPONENTS','//!WHEN'))]
    text += [ROUNDING, 'vec4 hook() {']
    produced = {}
    for block in tail:
        target = fields(block,'SAVE')[0]
        macros, body = code(block)
        for texture,value in produced.items():
            macros = macros.replace(f'{texture}_tex({texture}_pos)',value)
        text += [macros]
        if target != output:
            text += [f'    vec4 value_{target};']
        text += ['    {']+['        '+line.strip() for line in body.splitlines()]
        if target == output:
            text += ['        return result + MAIN_tex(MAIN_pos);' if restore else '        return result;']
        else:
            text += [f'        value_{target} = round_feature(result);']
        text += ['    }']+['#undef '+v for v in re.findall(r'^#define (\w+)',macros,re.M)]
        produced[target] = 'value_'+target
    text += ['}']
    suffix = [] if restore else ['//!DESC'+passes[-1]]
    return '//!DESC'.join([header,*passes[:end-count]])+'\n'+'\n'.join(text+suffix)+'\n'

def scalar_projection(body):
    # GLSL constructors are column-major. Each fragment calculates only the
    # row belonging to its subpixel, reducing duplicated arithmetic at 2x.
    def matrix_term(match):
        weights, operand = match.groups()
        return f'dot(transpose(mat4({weights}))[component], {operand})'
    body = re.sub(r'mat4\(([^)]+)\) \* (go_\d+\([^)]+\)|g_\d+)', matrix_term, body)
    body = body.replace('vec4 result =', 'float result =')
    return re.sub(r'result \+= vec4\(([^)]+)\);',
                  r'result += vec4(\1)[component];', body)

def subpixel(name, original):
    header,*passes = original.split('//!DESC')
    count = 4 if name.endswith('_VL') else 2
    projections = passes[-count:-1]
    inputs = ['MAIN']
    for block in projections:
        inputs.extend(v for v in fields(block,'BIND') if v not in inputs)
    text = [f'//!DESC Anime4K {name} OpenGL fused scalar output','//!HOOK MAIN']
    text += ['//!BIND '+v for v in inputs]
    text += ['//!SAVE MAIN','//!WIDTH MAIN.w 2 *','//!HEIGHT MAIN.h 2 *']
    text += ['//!WHEN '+fields(passes[-1],'WHEN')[0]]
    text += [ROUNDING,'vec4 hook() {',
             '    vec2 center = floor(MAIN_pos * MAIN_size) + vec2(0.5);',
             '    ivec2 subpixel = ivec2(fract(MAIN_pos * MAIN_size) * 2.0);',
             '    int component = 2 * subpixel.y + subpixel.x;']
    for channel,block in enumerate(projections):
        macros,body = code(block)
        for texture in fields(block,'BIND'):
            macros=macros.replace(f'{texture}_tex({texture}_pos)',f'{texture}_tex(center * {texture}_pt)')
            macros=macros.replace(f'{texture}_texOff(vec2(x_off, y_off))',
                                  f'{texture}_tex((center + vec2(x_off,y_off)) * {texture}_pt)')
        text += [macros,f'    float residual{channel};','    {']
        text += ['        '+line.strip() for line in scalar_projection(body).splitlines()]
        text += [f'        residual{channel} = round_half(result);','    }']
        text += ['#undef '+v for v in re.findall(r'^#define (\w+)',macros,re.M)]
    residual = 'vec4(residual0)' if len(projections)==1 else 'vec4(residual0, residual1, residual2, residual2)'
    text += [f'    return MAIN_tex(MAIN_pos) + {residual};','}']
    return '//!DESC'.join([header,*passes[:-count]])+'\n'+'\n'.join(text)+'\n'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path,help='Directory containing original Anime4K shaders')
    parser.add_argument('destination',type=Path)
    args=parser.parse_args()
    args.destination.mkdir(parents=True,exist_ok=True)
    names=['Restore_CNN_M','Restore_CNN_Soft_M','Restore_CNN_VL',
           'Upscale_CNN_x2_M','Upscale_Denoise_CNN_x2_M','Upscale_CNN_x2_S','Upscale_CNN_x2_VL']
    for name in names:
        original=(args.source/f'Anime4K_{name}.glsl').read_text()
        if name in ['Upscale_CNN_x2_S','Upscale_CNN_x2_VL']:
            result=subpixel(name,original)
        else:
            result=same_resolution(name,original,name.startswith('Restore'))
        (args.destination/f'Anime4K_{name}_Fused_OpenGL.glsl').write_text(result.rstrip() + '\n')
if __name__=='__main__':
    main()
