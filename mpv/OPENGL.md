# OpenGL fused Anime4K shaders

These fragment shaders run on the OpenGL 4.1 renderer bundled with IINA 1.5.1 (build 183). They were implemented and tested on a 7-core Apple M1. Copy the seven `*_Fused_OpenGL.glsl` files to your shader folder and select them using your shader-management plugin. Each file replaces its corresponding original Anime4K file in a preset chain. Use the original clamp, downscale, and small restoration stages alongside them.

| Shader | Original passes | OpenGL fused passes |
|---|---:|---:|
| Restore CNN M | 8 | 7 |
| Restore CNN Soft M | 8 | 7 |
| Restore CNN VL | 17 | 15 |
| Upscale CNN x2 M | 9 | 8 |
| Upscale Denoise CNN x2 M | 9 | 8 |
| Upscale CNN x2 S | 5 | 4 |
| Upscale CNN x2 VL | 18 | 15 |

Medium models fuse the final 3×3 convolution with the 1×1 projection at input resolution. Their pixel shuffle remains a separate pass. VL restoration fuses both final feature groups with the projection. S and VL upscalers fuse the output convolution or projections with pixel shuffle. Each output fragment calculates the matrix row for its subpixel. Biases, activation functions, sampling conditions and residual additions follow the source shaders. Software rounding replaces the intermediate FP16 texture conversion using GLSL 4.1 operations.

All seven individual files and all seven preset chains compiled and rendered with IINA's bundled libmpv and an offscreen OpenGL 4.1 context. There were 35 output comparisons against original shaders using a 171×97 RGB crop. Full chains were tested at 1×, approximately 1.54×, 2× and 4× output size. The maximum absolute RGBA8 component difference across all comparisons was one level out of 255. Floating-point reassociation and texture sampling can produce small differences. [Validation data](opengl-validation.csv)

The following exploratory timings use a 1920×1080 YUV420 test clip rendered to 3840×2160. Each cell is the median of 20 render calls after five warm-up calls. The measurement wraps the render call and waits for GPU completion, with target presentation waits disabled. It includes CPU submission and GPU work. Each variant ran in its own process, with stock followed by fused. Thermal drift and scheduling can affect these short measurements. The test uses default renderer settings and excludes display presentation, IINA UI, ICC processing, and the user's debanding options. IINA playback performance may differ.

| Preset | Original ms/frame | Fused ms/frame |
|---|---:|---:|
| A | 45.22 | 43.66 |
| B | 45.69 | 43.62 |
| C | 31.42 | 33.57 |
| A+A | 66.07 | 63.72 |
| B+B | 68.45 | 63.73 |
| C+A | 54.59 | 52.05 |
| A+A HQ | 330.13 | 344.57 |

The Mode C and HQ chains were slower in this run despite their reduced pass counts. These measurements apply to the OpenGL implementations. The repository's root timing table measures the separate native Metal implementation. [Raw OpenGL timings](opengl-timings.csv)

To regenerate the files from the original Anime4K shader directory, run from the repository root.

```sh
python3 mpv/generate_opengl.py /path/to/original/Anime4K/shaders mpv/shaders
```

The source weights and original MIT notices are retained in each generated file.
