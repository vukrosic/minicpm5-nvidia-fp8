# MiniCPM5-1B FP8 for NVIDIA GPUs

The optimized profile is no longer artificially restricted to one GPU model.

## Compatibility

- architecture compatibility with NVIDIA compute capability 8.0 or newer
  (Ampere, Ada, Hopper and newer), subject to available VRAM;
- automatic compatibility preflight using the selected GPU;
- clear `compatible-unbenchmarked` status outside the measured RTX 3060;
- RTX 3060 remains the source of every published performance number.

The current BF16 profile does not support Turing, even though Marlin's
weight-only FP8 path can run there. AMD, Intel, Apple GPUs and CPUs are outside
this package.

## RTX 3060 benchmark

- **248.52 decode tok/s** for one active request;
- **1.503×** the matched BF16 vLLM reference;
- HumanEval+: **96/164 FP8** vs **95/164 BF16**;
- frozen cross-domain slices: **87/200 FP8** vs **89/200 BF16**.

The GitHub repository is now `minicpm5-nvidia-fp8`. Run `./serve`; the FP8
profile remains the default.
