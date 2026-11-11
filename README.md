# Anime4K layer fusion on a 7-core Apple M1

At 1920×1080 input, fusing the final 3×3 convolution, the 1×1 projection, and the 2× pixel shuffle reduced median GPU time from **14.90 to 12.16 ms (18.4% less GPU time)** for the full medium upscaler. All output values matched the separate-pass Metal implementation. Applying an algebraic rewrite to the same seven-dispatch schedule took **11.31 ms (24.1% less than baseline)** with a maximum absolute output difference of 0.000977.

## GPU timings

Times are median milliseconds per frame. Brackets give the 10th and 90th percentiles of the 30 batch averages.

| Execution schedule | Dispatches | 1280×720 → 2560×1440 | 1920×1080 → 3840×2160 |
|---|---:|---:|---:|
| Separate passes | 9 | 6.84 [6.27, 8.41] | 14.90 [14.42, 15.63] |
| Fuse projection and pixel shuffle | 8 | 6.55 [6.05, 7.59] | 13.99 [13.87, 14.63] |
| Fuse final convolution, projection and pixel shuffle | 7 | 5.75 [5.19, 6.64] | 12.16 [12.08, 12.80] |
| Fuse three pairs of adjacent 3×3 layers | 6 | 8.09 [7.74, 9.64] | 17.29 [16.83, 18.18] |
| Fuse the spatial pairs and the final three passes | 4 | 6.84 [6.28, 8.17] | 15.01 [14.50, 15.64] |
| Final three fused, plus algebraic rewrite across the network | 7 | 5.49 [4.80, 6.27] | 11.31 [11.24, 12.40] |

## Usage

Shader files are provided in [`mpv/shaders`](./mpv/shaders)

| Mode | Combination |
|:----:| ----------- |
| A | `~~/shaders/Anime4K_Clamp_Highlights.glsl:~~/shaders/Anime4K_Restore_CNN_M_Fused.glsl:~~/shaders/Anime4K_Upscale_CNN_x2_M_Fused.glsl:~~/shaders/Anime4K_AutoDownscalePre_x2.glsl:~~/shaders/Anime4K_AutoDownscalePre_x4.glsl:~~/shaders/Anime4K_Upscale_CNN_x2_S_Fused.glsl` |
| B | `~~/shaders/Anime4K_Clamp_Highlights.glsl:~~/shaders/Anime4K_Restore_CNN_Soft_M_Fused.glsl:~~/shaders/Anime4K_Upscale_CNN_x2_M_Fused.glsl:~~/shaders/Anime4K_AutoDownscalePre_x2.glsl:~~/shaders/Anime4K_AutoDownscalePre_x4.glsl:~~/shaders/Anime4K_Upscale_CNN_x2_S_Fused.glsl` |
| C | `~~/shaders/Anime4K_Clamp_Highlights.glsl:~~/shaders/Anime4K_Upscale_Denoise_CNN_x2_M_Fused.glsl:~~/shaders/Anime4K_AutoDownscalePre_x2.glsl:~~/shaders/Anime4K_AutoDownscalePre_x4.glsl:~~/shaders/Anime4K_Upscale_CNN_x2_S_Fused.glsl` |
| A + A | `~~/shaders/Anime4K_Clamp_Highlights.glsl:~~/shaders/Anime4K_Restore_CNN_M_Fused.glsl:~~/shaders/Anime4K_Upscale_CNN_x2_M_Fused.glsl:~~/shaders/Anime4K_Restore_CNN_S.glsl:~~/shaders/Anime4K_AutoDownscalePre_x2.glsl:~~/shaders/Anime4K_AutoDownscalePre_x4.glsl:~~/shaders/Anime4K_Upscale_CNN_x2_S_Fused.glsl` |
| B + B | `~~/shaders/Anime4K_Clamp_Highlights.glsl:~~/shaders/Anime4K_Restore_CNN_Soft_M_Fused.glsl:~~/shaders/Anime4K_Upscale_CNN_x2_M_Fused.glsl:~~/shaders/Anime4K_AutoDownscalePre_x2.glsl:~~/shaders/Anime4K_AutoDownscalePre_x4.glsl:~~/shaders/Anime4K_Restore_CNN_Soft_S.glsl:~~/shaders/Anime4K_Upscale_CNN_x2_S_Fused.glsl` |
| C + A | `~~/shaders/Anime4K_Clamp_Highlights.glsl:~~/shaders/Anime4K_Upscale_Denoise_CNN_x2_M_Fused.glsl:~~/shaders/Anime4K_AutoDownscalePre_x2.glsl:~~/shaders/Anime4K_AutoDownscalePre_x4.glsl:~~/shaders/Anime4K_Restore_CNN_S.glsl:~~/shaders/Anime4K_Upscale_CNN_x2_S_Fused.glsl` |
| A + A (HQ) | `~~/shaders/Anime4K_Clamp_Highlights.glsl:~~/shaders/Anime4K_Restore_CNN_VL.glsl:~~/shaders/Anime4K_Upscale_CNN_x2_VL_Fused.glsl:~~/shaders/Anime4K_Restore_CNN_M_Fused.glsl:~~/shaders/Anime4K_AutoDownscalePre_x2.glsl:~~/shaders/Anime4K_AutoDownscalePre_x4.glsl:~~/shaders/Anime4K_Upscale_CNN_x2_M_Fused.glsl` |

## What changes inside a kernel

The 1×1 projection reads all seven convolution outputs at one input pixel and produces four subpixel residuals. Fusing the last convolution, projection and shuffle lets the thread calculate the seventh feature from the sixth feature texture, combine it with the six earlier features, and write its four output pixels. This saves two intermediate texture writes and subsequent reads per input pixel, while preserving each layer's FP16 rounding.

The spatial-pair kernel has a different tradeoff. A 16×8 output tile computes an 18×10 tile of the first convolution so the next 3×3 convolution has a one-pixel halo. That costs about 41% extra first-layer evaluations per full tile and a threadgroup barrier. The projection later needs every feature map, so each pair still writes its interior first-layer activations. These are properties of the tested implementation. GPU hardware counters were not collected to attribute the time difference to individual costs.

The algebraic variant uses the following identity at each signed activation, where P and N are the two learned weight matrices for the same tap. The generator precomputes P+N. This removes one max per split while retaining both matrix products. The remaining sign dependence prevents folding consecutive learned convolutions into one fixed linear convolution in general.

    P max(x,0) + N max(-x,0) = P x + (P+N) max(-x,0)

## Method and validation

The shader source is pinned to upstream commit 7684e9586f8dcc738af08a1cdceb024cc184f426. The generator copies the matrix columns, offsets, accumulation order and biases into Metal. The baseline runs seven 3×3 convolutions, a 1×1 projection and a residual pixel shuffle. The benchmark uses clamp-to-edge sampling, bilinear base-image upsampling, RGBA16Float textures, float accumulators, Metal safe math mode, and a common low-resolution dispatch grid for all schedules. [Upstream shader](https://github.com/bloc97/Anime4K/blob/7684e9586f8dcc738af08a1cdceb024cc184f426/glsl/Upscale/Anime4K_Upscale_CNN_x2_M.glsl)

Measured on macOS 26.6.2 (25G83). Each variant reused a deterministic RGB noise frame with alpha one. After validation and five warm-up batches per variant, 30 rounds ran all six schedules in shuffled order. Each sample used three frame executions in one command buffer. GPUStartTime and GPUEndTime were read after completion and the elapsed time divided by three. Compilation, texture allocation, CPU submission, video decoding, player rendering and display presentation are outside these timings.

Each fused output was compared at every pixel and RGBA component to the separate-pass Metal port at 37×23, 1280×720 and 1920×1080 input. All four fusion-only schedules matched numerically at every value. With the algebraic rewrite, at 1080p input 349,979 of 33,177,600 output components differed. Mean absolute error was 0.00000317 and maximum absolute error was 0.0009765625. The port was not cross-validated against an mpv-rendered frame. Only this model, tile shape, precision and device were profiled.

## Reproduce

With Python 3 and recent macOS Command Line Tools installed, run these commands from the repository root. The first command generates the Metal source from the included upstream shader. The executable needs access to the Metal device.

```sh
python3 make_kernels.py kernels.metal
xcrun clang++ -std=c++17 -O2 -fobjc-arc -framework Metal -framework Foundation benchmark.mm -o bench
./bench kernels.metal > rerun.csv 2> rerun.log
```

The recorded run contains 360 batch measurements and component-wise checks at three image sizes. [Raw GPU timings](timings.csv) · [Validation log](validation.log)
