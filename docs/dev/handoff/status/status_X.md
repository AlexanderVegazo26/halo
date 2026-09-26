# WS-X status (EVO-X2 field kit)
Updated: 2026-09-25
## State
- M1 (package + build options): IN PROGRESS
- M2 (collect.sh): not started
- M3 (docs + final package): not started
## Decisions so far
- HALO_MARCH=x86-64-v3 for the package: dev host (Ryzen 7 4800H, Zen 2) supports v3, NOT v4 (glibc-hwcaps probe).
- Tests bake absolute paths (HALO_REF_DIR, HALO_SOURCE_DIR, HALO_API_FIXTURES, HALO_CHILD_HELPER): kit uses a fixed anchor prefix /tmp/halo-kit-<sha12>.
- Reference data (/root/halo-ref, 1.9 GB) ships as a separate tarball.
- Package builds from git archive HEAD + overlay of WS-X paths (not the live shared tree).
## Scratch tooling: x_*
