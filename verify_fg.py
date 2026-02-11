#!/usr/bin/env python3
"""Verify f,g piecewise averaging against known reference."""
import sys
import numpy as np

sys.path.insert(0, '/home/user/psfinal')
from persson_model.utils.data_loader import (
    compute_fg_from_strain_sweep,
    average_fg_curves,
    create_fg_interpolator,
    persson_strain_grid,
)

# ── 1. Raw input data ──────────────────────────────────────
# Format per row: (Temperature, Frequency, Strain%, ReE, ImE)
raw_data = {
    0.02: [
        (1, 0.00998199, 12.3965, 1.76989),
        (1, 0.0133129, 12.3579, 1.77681),
        (1, 0.0177461, 12.3418, 1.78331),
        (1, 0.0236596, 12.3492, 1.80447),
        (1, 0.031558, 12.2931, 1.8164),
        (1, 0.0420875, 12.2247, 1.83397),
        (1, 0.0561241, 12.1139, 1.85492),
        (1, 0.0748532, 11.96, 1.88646),
        (1, 0.099816, 11.7526, 1.92113),
        (1, 0.133117, 11.4914, 1.95493),
        (1, 0.177512, 11.1729, 1.98251),
        (1, 0.236721, 10.8024, 2.00125),
        (1, 0.315669, 10.3899, 2.00493),
        (1, 0.42097, 9.94642, 1.99124),
        (1, 0.561402, 9.47247, 1.95595),
        (1, 0.748652, 8.97981, 1.90113),
        (1, 0.998458, 8.46497, 1.82624),
        (1, 1.33148, 7.93229, 1.73507),
        (1, 1.77568, 7.38546, 1.63276),
        (1, 2.36817, 6.81151, 1.52133),
        (1, 3.15846, 6.20654, 1.40454),
        (1, 4.21252, 5.57118, 1.28468),
        (1, 5.6189, 4.91792, 1.16317),
        (1, 7.49479, 4.27492, 1.04087),
        (1, 9.99712, 3.67123, 0.915441),
        (1, 13.3315, 3.12654, 0.783644),
        (1, 17.7678, 2.65958, 0.647271),
        (1, 24.0898, 2.39459, 0.510474),
        (1, 25.1467, 2.34769, 0.47739),
    ],
    9.94: [
        (1, 0.0099875, 10.9768, 1.35416),
        (1, 0.0133212, 10.9577, 1.34117),
        (1, 0.017767, 10.943, 1.34462),
        (1, 0.0236861, 10.9141, 1.35255),
        (1, 0.0315812, 10.8752, 1.3618),
        (1, 0.042114, 10.7959, 1.37742),
        (1, 0.0561403, 10.6951, 1.39078),
        (1, 0.0748584, 10.5596, 1.41098),
        (1, 0.0998339, 10.3817, 1.43529),
        (1, 0.133138, 10.1641, 1.46189),
        (1, 0.177559, 9.89803, 1.48585),
        (1, 0.236777, 9.5962, 1.50286),
        (1, 0.315749, 9.26689, 1.50928),
        (1, 0.421055, 8.9119, 1.50267),
        (1, 0.561495, 8.53631, 1.48018),
        (1, 0.748819, 8.13766, 1.44313),
        (1, 0.998588, 7.72817, 1.3897),
        (1, 1.33171, 7.29787, 1.3244),
        (1, 1.77597, 6.84293, 1.24907),
        (1, 2.3686, 6.35803, 1.16787),
        (1, 3.15909, 5.83214, 1.08212),
        (1, 4.21328, 5.2652, 0.993642),
        (1, 5.61951, 4.66899, 0.9036),
        (1, 7.49538, 4.07185, 0.812517),
        (1, 9.9975, 3.50604, 0.719437),
        (1, 13.3325, 3.00446, 0.623393),
        (1, 17.7732, 2.56688, 0.523096),
        (1, 23.7157, 2.29471, 0.420665),
        (1, 26.3193, 2.23593, 0.381457),
    ],
    29.9: [
        (1, 0.00993227, 7.92113, 0.812626),
        (1, 0.0133018, 7.85741, 0.793673),
        (1, 0.0177287, 7.82279, 0.797681),
        (1, 0.0236355, 7.81039, 0.8005),
        (1, 0.0315658, 7.75717, 0.806546),
        (1, 0.0420936, 7.71052, 0.81615),
        (1, 0.0560989, 7.64443, 0.823229),
        (1, 0.0748367, 7.56082, 0.823972),
        (1, 0.0997999, 7.4614, 0.825199),
        (1, 0.133073, 7.33974, 0.82181),
        (1, 0.177427, 7.20522, 0.824052),
        (1, 0.236635, 7.05659, 0.828837),
        (1, 0.315539, 6.89171, 0.832713),
        (1, 0.420773, 6.71473, 0.834018),
        (1, 0.561073, 6.52162, 0.829128),
        (1, 0.748208, 6.31242, 0.817982),
        (1, 0.99778, 6.08633, 0.799891),
        (1, 1.33063, 5.83939, 0.775213),
        (1, 1.77458, 5.56679, 0.744597),
        (1, 2.3666, 5.2637, 0.709913),
        (1, 3.1564, 4.92131, 0.672274),
        (1, 4.20982, 4.52905, 0.631911),
        (1, 5.61481, 4.08678, 0.588998),
        (1, 7.4889, 3.61581, 0.543269),
        (1, 9.98856, 3.15133, 0.494747),
        (1, 13.3223, 2.72296, 0.441901),
        (1, 17.7651, 2.35499, 0.383864),
        (1, 23.6793, 2.05225, 0.322323),
        (1, 31.7316, 1.90996, 0.263056),
    ],
    49.99: [
        (1, 0.00991974, 6.36074, 0.564004),
        (1, 0.0133148, 6.32266, 0.531796),
        (1, 0.0176711, 6.33331, 0.545917),
        (1, 0.0236869, 6.29266, 0.530814),
        (1, 0.0315095, 6.29257, 0.542256),
        (1, 0.0420438, 6.25757, 0.544083),
        (1, 0.0560571, 6.232, 0.543356),
        (1, 0.0747942, 6.19966, 0.550299),
        (1, 0.0997915, 6.15963, 0.553352),
        (1, 0.133103, 6.10052, 0.559198),
        (1, 0.177507, 6.03305, 0.56475),
        (1, 0.236681, 5.95228, 0.570907),
        (1, 0.315597, 5.85864, 0.57597),
        (1, 0.420808, 5.75195, 0.578493),
        (1, 0.561103, 5.63059, 0.577807),
        (1, 0.748215, 5.49496, 0.573712),
        (1, 0.997827, 5.34245, 0.565579),
        (1, 1.33076, 5.16955, 0.553316),
        (1, 1.77476, 4.97189, 0.537297),
        (1, 2.36687, 4.74285, 0.517747),
        (1, 3.15656, 4.47503, 0.495432),
        (1, 4.2097, 4.16225, 0.470468),
        (1, 5.61436, 3.8047, 0.443194),
        (1, 7.48803, 3.4133, 0.413561),
        (1, 9.98679, 3.01052, 0.381353),
        (1, 13.3194, 2.62562, 0.345818),
        (1, 17.761, 2.27965, 0.305989),
        (1, 23.6769, 1.99052, 0.26321),
    ],
}

# Convert to format expected by compute_fg_from_strain_sweep
# Each row: (freq, strain%, ReE, ImE)
data_by_T = {}
for T, rows in raw_data.items():
    data_by_T[T] = [(freq, strain, ReE, ImE) for (freq, strain, ReE, ImE) in rows]

# ── 2. Expected answer ─────────────────────────────────────
expected = np.array([
    [1.49E-04, 9.97E-01, 1.00E+00],
    [2.22E-04, 9.92E-01, 1.01E+00],
    [3.33E-04, 9.82E-01, 1.02E+00],
    [4.98E-04, 9.62E-01, 1.04E+00],
    [7.46E-04, 9.24E-01, 1.06E+00],
    [1.12E-03, 8.66E-01, 1.09E+00],
    [1.67E-03, 7.93E-01, 1.12E+00],
    [2.50E-03, 7.02E-01, 1.13E+00],
    [3.74E-03, 6.04E-01, 1.10E+00],
    [5.60E-03, 5.10E-01, 1.03E+00],
    [8.39E-03, 4.25E-01, 9.40E-01],
    [1.26E-02, 3.49E-01, 8.28E-01],
    [1.88E-02, 2.82E-01, 7.09E-01],
    [2.82E-02, 2.23E-01, 5.87E-01],
    [4.22E-02, 1.74E-01, 4.69E-01],
    [6.32E-02, 1.47E-01, 3.55E-01],
    [9.46E-02, 1.27E-01, 2.88E-01],
    [1.42E-01, 1.27E-01, 2.88E-01],
    [2.12E-01, 1.27E-01, 2.88E-01],
    [3.17E-01, 1.27E-01, 2.88E-01],
])
exp_strain = expected[:, 0]
exp_f = expected[:, 1]
exp_g = expected[:, 2]

# ── 3. Compute f,g per temperature ─────────────────────────
print("=" * 80)
print("STEP 1: Compute f,g per temperature (clip_leq_1=False)")
print("=" * 80)

fg_by_T = compute_fg_from_strain_sweep(
    data_by_T,
    target_freq=1.0,
    freq_mode='nearest',
    strain_is_percent=True,
    e0_n_points=5,
    clip_leq_1=False   # Don't clip g
)

for T in sorted(fg_by_T.keys()):
    d = fg_by_T[T]
    s, f, g = d['strain'], d['f'], d['g']
    print(f"\nT={T}°C: E0_re={d['E0_re']:.4f}, E0_im={d['E0_im']:.6f}")
    print(f"  strain range: [{s[0]:.6e}, {s[-1]:.6e}], {len(s)} points")
    print(f"  f range: [{f.min():.4f}, {f.max():.4f}]")
    print(f"  g range: [{g.min():.4f}, {g.max():.4f}]")

# ── 4. Create Persson grid ─────────────────────────────────
print("\n" + "=" * 80)
print("STEP 2: Create Persson strain grid")
print("=" * 80)

# Persson grid with 20 points, max_strain ~0.33
grid_strain = persson_strain_grid(0.33, start_strain=1.49e-4, ratio=1.4975)
# Ensure we have ~20 points
print(f"Grid: {len(grid_strain)} points, [{grid_strain[0]:.4e}, {grid_strain[-1]:.4e}]")
print(f"Grid points: {grid_strain}")
print(f"Expected:     {exp_strain}")

# Use the expected grid instead for precise comparison
grid_strain = exp_strain.copy()
print(f"\nUsing expected strain grid for comparison ({len(grid_strain)} points)")

# ── 5. Try different averaging approaches ───────────────────
temps_all = sorted(fg_by_T.keys())
temps_A = [0.02, 9.94, 29.9]   # AAA
temps_B = [49.99]               # BBB

print("\n" + "=" * 80)
print("STEP 3: Average f,g curves - Various approaches")
print("=" * 80)

# Approach A: Simple mean of ALL temperatures (no clipping)
result_all = average_fg_curves(
    fg_by_T, temps_all, grid_strain,
    interp_kind='loglog_linear', avg_mode='mean', clip_leq_1=False
)

# Approach B: Only Group A (AAA temperatures)
result_A = average_fg_curves(
    fg_by_T, temps_A, grid_strain,
    interp_kind='loglog_linear', avg_mode='mean', clip_leq_1=False
)

# Approach C: Only Group B (BBB temperature)
result_B = average_fg_curves(
    fg_by_T, temps_B, grid_strain,
    interp_kind='loglog_linear', avg_mode='mean', clip_leq_1=False
)

# Approach D: Simple mean with clipping (old behavior)
fg_by_T_clipped = compute_fg_from_strain_sweep(
    data_by_T,
    target_freq=1.0,
    freq_mode='nearest',
    strain_is_percent=True,
    e0_n_points=5,
    clip_leq_1=True
)
result_all_clipped = average_fg_curves(
    fg_by_T_clipped, temps_all, grid_strain,
    interp_kind='loglog_linear', avg_mode='mean', clip_leq_1=True
)

# ── Print comparison table ──────────────────────────────────
print("\n" + "=" * 80)
print("COMPARISON TABLE")
print("=" * 80)

def print_comparison(label, result, exp_f, exp_g, grid_strain):
    if result is None:
        print(f"\n{label}: FAILED (returned None)")
        return
    f_avg = result['f_avg']
    g_avg = result['g_avg']
    print(f"\n{label}:")
    print(f"{'Strain':>12s}  {'f_calc':>8s}  {'f_exp':>8s}  {'f_err%':>8s}  "
          f"{'g_calc':>8s}  {'g_exp':>8s}  {'g_err%':>8s}")
    print("-" * 80)
    f_errs, g_errs = [], []
    for i in range(len(grid_strain)):
        f_err = (f_avg[i] - exp_f[i]) / exp_f[i] * 100 if exp_f[i] != 0 else 0
        g_err = (g_avg[i] - exp_g[i]) / exp_g[i] * 100 if exp_g[i] != 0 else 0
        f_errs.append(abs(f_err))
        g_errs.append(abs(g_err))
        print(f"{grid_strain[i]:12.4e}  {f_avg[i]:8.4f}  {exp_f[i]:8.4f}  {f_err:+8.2f}  "
              f"{g_avg[i]:8.4f}  {exp_g[i]:8.4f}  {g_err:+8.2f}")
    print(f"{'':>12s}  {'':>8s}  {'':>8s}  {'MAE':>8s}  "
          f"{'':>8s}  {'':>8s}  {'MAE':>8s}")
    print(f"{'':>12s}  {'':>8s}  {'':>8s}  {np.mean(f_errs):8.2f}%  "
          f"{'':>8s}  {'':>8s}  {np.mean(g_errs):8.2f}%")

print_comparison("A) All temps, no clip (NEW)", result_all, exp_f, exp_g, grid_strain)
print_comparison("B) Group A only (AAA)", result_A, exp_f, exp_g, grid_strain)
print_comparison("C) Group B only (BBB)", result_B, exp_f, exp_g, grid_strain)
print_comparison("D) All temps, clipped (OLD)", result_all_clipped, exp_f, exp_g, grid_strain)

# ── 6. Individual temperature curves ───────────────────────
print("\n" + "=" * 80)
print("STEP 4: Individual temperature f,g at grid points")
print("=" * 80)
for T in sorted(fg_by_T.keys()):
    d = fg_by_T[T]
    f_interp, g_interp = create_fg_interpolator(
        d['strain'], d['f'], d['g'],
        interp_kind='loglog_linear', extrap_mode='hold'
    )
    f_vals = np.array([f_interp(s) for s in grid_strain])
    g_vals = np.array([g_interp(s) for s in grid_strain])
    print(f"\nT={T}°C:")
    print(f"{'Strain':>12s}  {'f':>8s}  {'g':>8s}")
    for i in range(len(grid_strain)):
        print(f"{grid_strain[i]:12.4e}  {f_vals[i]:8.4f}  {g_vals[i]:8.4f}")
