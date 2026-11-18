// Preprocessor-only prologue of the matvec_<type>.comp shaders that are built as several -D
// variants (HALO_VK_SHADERS entries `<name>=<path>:<DEFINE>` in backends/vulkan/CMakeLists.txt).
// Include it directly after `#version`, before any other token: #extension must come first.
#ifdef HALO_SUBGROUP_REDUCE
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable
#endif
#if defined(HALO_SUBGROUP_REDUCE) && defined(HALO_FUSE_NORM)
#error HALO_SUBGROUP_REDUCE and HALO_FUSE_NORM are separate variants of this shader, never combined
#endif

// HALO_MV_ENTRY(q4k, fast): the per-type body main() calls, q4k_sg_fast in the HALO_SUBGROUP_REDUCE
// build (matvec_<type>_sg_main.glsl), q4k_fast otherwise (matvec_<type>_main.glsl).
#ifdef HALO_SUBGROUP_REDUCE
#define HALO_MV_ENTRY(p, fn) p##_sg_##fn
#else
#define HALO_MV_ENTRY(p, fn) p##_##fn
#endif
