// HALO activation functions for the layer kernels (included with -I shaders/common).
// Every expression is `precise` (SPIR-V NoContraction): no multiply-add may be fused, so
// the arithmetic around the transcendental calls rounds exactly as halo::cpu's
// (-ffp-contract=off). The formulas are the CPU ones (backends/cpu/kernel_common.h):
//   sigmoid(x) = 1 / (1 + exp(-x)),  silu(x) = x / (1 + exp(-x)),
//   softplus(x) = x > 20 ? x : log1p(exp(x))   (torch threshold 20).
// Only exp / log differ from the CPU's libm. Vulkan's precision rules: exp is within
// 3 + 2|x| ULP, division within 2.5 ULP, and log within 3 ULP outside [0.5, 2] but only
// 2^-21 ABSOLUTE inside it — so log(1 + e) would lose all relative precision for small e.
// halo_log1p therefore uses the alternating series for e <= 0.5 (24 terms: truncation
// < 0.5^23 / 24 < 2^-26 relative) and log(1 + e) only when the result is >= log(1.5).
#ifndef HALO_MATH_GLSL
#define HALO_MATH_GLSL

float halo_sigmoid(float x) {
    precise float e = exp(-x);
    precise float d = 1.0 + e;
    precise float r = 1.0 / d;
    return r;
}

float halo_silu(float x) {
    precise float e = exp(-x);
    precise float d = 1.0 + e;
    precise float r = x / d;
    return r;
}

// log(1 + e) for e >= 0.
float halo_log1p(float e) {
    if (e <= 0.5) {
        // e - e^2/2 + e^3/3 - ... (Horner, from the 24th term down)
        precise float s = 0.0;
        for (int n = 24; n >= 1; --n) {
            precise float c = ((n & 1) == 1 ? 1.0 : -1.0) / float(n);
            s = c + e * s;
        }
        precise float r = e * s;
        return r;
    }
    precise float u = 1.0 + e;
    return log(u);
}

float halo_softplus(float x) {
    if (x > 20.0) return x;
    return halo_log1p(exp(x));
}

#endif
