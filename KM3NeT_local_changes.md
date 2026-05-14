# KM3NeT local patches on feature/nubar-and-charm-threshold

These changes are local-only (not pushed to upstream). They fix bugs
discovered during KM3NeT/ORCA dimuon production on CC-IN2P3, May 2026.

## 1. inject_convert_neutrinos_km3net.py -- vestigial quark_type removed (2026-05-14)

File: resources/Examples/Example1/inject_convert_neutrinos_km3net.py
Lines: ~270, ~278 (configure_charm_cross_sections)

Commit 3341768c updated the C++ QuarkDISFromSpline pybind constructor (removed quark_type arg)
but did NOT update configure_charm_cross_sections in the injector. It was calling with 9 args
where arg4 was the vestigial quark_type=1. Fixed to 8-arg call:
  siren.interactions.QuarkDISFromSpline(dsdy, sigma, interaction, isoscalar_mass, 1, [primary], [target], "m")

## 2. inject_convert_neutrinos_km3net.py -- CreateTempSirenDetector stale-dir fix (2026-05-13)

Function: CreateTempSirenDetector
- Uncommented rm -rf lines for stale Detectors/materials/nu{seed} and densities/nu{seed}.
- Added block to create detectors/nu{seed}/ (lowercase, physics-model dir) by copying from
  detectors/KM3NeTORCA. SIREN_Controller searches this tree; old code only wrote to Detectors/.

## 3. cmake/Packages/CFITSIO.cmake -- make CFITSIO optional (2026-05-13)

Changed find_package(CFITSIO REQUIRED) to find_package(CFITSIO) so the build does not abort
when CFITSIO headers are absent in the container. Library bind-mounted at runtime.

## Build details (IN2P3, 2026-05-13)

- Branch: feature/nubar-and-charm-threshold
- Build dir: ~/SIREN/build_nubar/
- Staging dir: ~/SIREN/build_nubar/python_staging/ (siren.smk sets PYTHONPATH here first)
- Runtime Python: ~/python_envs/siren_env/ (interpreter + non-SIREN packages only)
- SIREN .so override via PYTHONPATH: build_nubar/python_staging placed FIRST to shadow old siren_env
