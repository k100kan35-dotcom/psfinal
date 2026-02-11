#!/usr/bin/env python3
"""Strain 구간별 다른 온도 조합 적용 탐색.

전략: 특정 strain threshold를 기준으로 저/고 strain 구간에
서로 다른 온도 조합 + 가중치를 독립 적용.
f와 g도 독립적으로 최적화.
"""
import sys
import numpy as np
from itertools import combinations

sys.path.insert(0, '/home/user/psfinal')
from persson_model.utils.data_loader import compute_fg_from_strain_sweep, average_fg_curves

# ── Raw data ────────────────────────────────────
raw_data = {
    0.02: [
        (1,0.00998199,12.3965,1.76989),(1,0.0133129,12.3579,1.77681),(1,0.0177461,12.3418,1.78331),
        (1,0.0236596,12.3492,1.80447),(1,0.031558,12.2931,1.8164),(1,0.0420875,12.2247,1.83397),
        (1,0.0561241,12.1139,1.85492),(1,0.0748532,11.96,1.88646),(1,0.099816,11.7526,1.92113),
        (1,0.133117,11.4914,1.95493),(1,0.177512,11.1729,1.98251),(1,0.236721,10.8024,2.00125),
        (1,0.315669,10.3899,2.00493),(1,0.42097,9.94642,1.99124),(1,0.561402,9.47247,1.95595),
        (1,0.748652,8.97981,1.90113),(1,0.998458,8.46497,1.82624),(1,1.33148,7.93229,1.73507),
        (1,1.77568,7.38546,1.63276),(1,2.36817,6.81151,1.52133),(1,3.15846,6.20654,1.40454),
        (1,4.21252,5.57118,1.28468),(1,5.6189,4.91792,1.16317),(1,7.49479,4.27492,1.04087),
        (1,9.99712,3.67123,0.915441),(1,13.3315,3.12654,0.783644),(1,17.7678,2.65958,0.647271),
        (1,24.0898,2.39459,0.510474),(1,25.1467,2.34769,0.47739),
    ],
    9.94: [
        (1,0.0099875,10.9768,1.35416),(1,0.0133212,10.9577,1.34117),(1,0.017767,10.943,1.34462),
        (1,0.0236861,10.9141,1.35255),(1,0.0315812,10.8752,1.3618),(1,0.042114,10.7959,1.37742),
        (1,0.0561403,10.6951,1.39078),(1,0.0748584,10.5596,1.41098),(1,0.0998339,10.3817,1.43529),
        (1,0.133138,10.1641,1.46189),(1,0.177559,9.89803,1.48585),(1,0.236777,9.5962,1.50286),
        (1,0.315749,9.26689,1.50928),(1,0.421055,8.9119,1.50267),(1,0.561495,8.53631,1.48018),
        (1,0.748819,8.13766,1.44313),(1,0.998588,7.72817,1.3897),(1,1.33171,7.29787,1.3244),
        (1,1.77597,6.84293,1.24907),(1,2.3686,6.35803,1.16787),(1,3.15909,5.83214,1.08212),
        (1,4.21328,5.2652,0.993642),(1,5.61951,4.66899,0.9036),(1,7.49538,4.07185,0.812517),
        (1,9.9975,3.50604,0.719437),(1,13.3325,3.00446,0.623393),(1,17.7732,2.56688,0.523096),
        (1,23.7157,2.29471,0.420665),(1,26.3193,2.23593,0.381457),
    ],
    29.9: [
        (1,0.00993227,7.92113,0.812626),(1,0.0133018,7.85741,0.793673),(1,0.0177287,7.82279,0.797681),
        (1,0.0236355,7.81039,0.8005),(1,0.0315658,7.75717,0.806546),(1,0.0420936,7.71052,0.81615),
        (1,0.0560989,7.64443,0.823229),(1,0.0748367,7.56082,0.823972),(1,0.0997999,7.4614,0.825199),
        (1,0.133073,7.33974,0.82181),(1,0.177427,7.20522,0.824052),(1,0.236635,7.05659,0.828837),
        (1,0.315539,6.89171,0.832713),(1,0.420773,6.71473,0.834018),(1,0.561073,6.52162,0.829128),
        (1,0.748208,6.31242,0.817982),(1,0.99778,6.08633,0.799891),(1,1.33063,5.83939,0.775213),
        (1,1.77458,5.56679,0.744597),(1,2.3666,5.2637,0.709913),(1,3.1564,4.92131,0.672274),
        (1,4.20982,4.52905,0.631911),(1,5.61481,4.08678,0.588998),(1,7.4889,3.61581,0.543269),
        (1,9.98856,3.15133,0.494747),(1,13.3223,2.72296,0.441901),(1,17.7651,2.35499,0.383864),
        (1,23.6793,2.05225,0.322323),(1,31.7316,1.90996,0.263056),
    ],
    49.99: [
        (1,0.00991974,6.36074,0.564004),(1,0.0133148,6.32266,0.531796),(1,0.0176711,6.33331,0.545917),
        (1,0.0236869,6.29266,0.530814),(1,0.0315095,6.29257,0.542256),(1,0.0420438,6.25757,0.544083),
        (1,0.0560571,6.232,0.543356),(1,0.0747942,6.19966,0.550299),(1,0.0997915,6.15963,0.553352),
        (1,0.133103,6.10052,0.559198),(1,0.177507,6.03305,0.56475),(1,0.236681,5.95228,0.570907),
        (1,0.315597,5.85864,0.57597),(1,0.420808,5.75195,0.578493),(1,0.561103,5.63059,0.577807),
        (1,0.748215,5.49496,0.573712),(1,0.997827,5.34245,0.565579),(1,1.33076,5.16955,0.553316),
        (1,1.77476,4.97189,0.537297),(1,2.36687,4.74285,0.517747),(1,3.15656,4.47503,0.495432),
        (1,4.2097,4.16225,0.470468),(1,5.61436,3.8047,0.443194),(1,7.48803,3.4133,0.413561),
        (1,9.98679,3.01052,0.381353),(1,13.3194,2.62562,0.345818),(1,17.761,2.27965,0.305989),
        (1,23.6769,1.99052,0.26321),
    ],
}

data_by_T = {}
for T, rows in raw_data.items():
    data_by_T[T] = [(freq, strain, ReE, ImE) for (freq, strain, ReE, ImE) in rows]

# ── CORRECT expected values ─────────────────────
exp_strain = np.array([
    1.49E-04, 2.22E-04, 3.33E-04, 4.98E-04, 7.46E-04,
    1.12E-03, 1.67E-03, 2.50E-03, 3.75E-03, 5.61E-03,
    8.40E-03, 1.26E-02, 1.88E-02, 2.82E-02, 4.22E-02,
    6.32E-02, 9.46E-02, 1.42E-01, 2.12E-01, 3.17E-01,
])
exp_f = np.array([
    0.995, 0.992, 0.988, 0.978, 0.965,
    0.944, 0.917, 0.896, 0.859, 0.818,
    0.773, 0.725, 0.691, 0.633, 0.567,
    0.493, 0.418, 0.374, 0.318, 0.304,
])
exp_g = np.array([
    0.983, 0.986, 0.997, 1.010, 1.025,
    1.042, 1.060, 1.071, 1.077, 1.068,
    1.037, 0.989, 0.949, 0.881, 0.808,
    0.733, 0.656, 0.601, 0.490, 0.413,
])

fg_by_T = compute_fg_from_strain_sweep(
    data_by_T, target_freq=1.0, freq_mode='nearest',
    strain_is_percent=True, e0_n_points=5, clip_leq_1=False
)

temps = sorted(fg_by_T.keys())

# ── 각 온도별 단독 f, g 미리 계산 ──────────────
single_T_results = {}
for T in temps:
    res = average_fg_curves(
        fg_by_T, [T], exp_strain,
        interp_kind='loglog_linear', avg_mode='mean', clip_leq_1=False
    )
    single_T_results[T] = res

# ── 가중 평균 헬퍼 ──────────────────────────────
def weighted_avg_at_indices(indices, temps_pair, weight, target='f'):
    """특정 strain 인덱스에서 가중 평균 f 또는 g 계산."""
    t1, t2 = temps_pair
    w1, w2 = weight, 1.0 - weight
    key = 'f_avg' if target == 'f' else 'g_avg'
    v1 = single_T_results[t1][key][indices]
    v2 = single_T_results[t2][key][indices]
    return w1 * v1 + w2 * v2

def mae_at_indices(calc, expected, indices):
    e = expected[indices]
    return np.nanmean(np.abs(calc - e) / np.where(e > 0, e, 1) * 100)


# ══════════════════════════════════════════════════
# STRATEGY: strain threshold 기준으로 low/high 구간별
# 독립적으로 (온도 조합, 가중치) 최적화
# ══════════════════════════════════════════════════

weight_range = np.arange(0.0, 1.05, 0.05)
temp_pairs = list(combinations(temps, 2))
# 단독 온도도 포함 (w=1.0으로)
temp_singles = [(T, T) for T in temps]
all_temp_options = temp_pairs + temp_singles

print("=" * 120)
print("Strain 구간별 다른 온도 적용 - 전수 탐색")
print("=" * 120)

# 가능한 split 지점: 각 strain 값을 threshold로
best_overall = {'mae': 999, 'detail': None}

split_results = []

for split_idx in range(2, len(exp_strain) - 1):  # 최소 2개씩은 있어야 함
    threshold = exp_strain[split_idx]
    lo_idx = np.arange(0, split_idx)
    hi_idx = np.arange(split_idx, len(exp_strain))

    # f: low구간 최적, high구간 최적 독립 탐색
    best_f_lo = {'mae': 999}
    best_f_hi = {'mae': 999}
    best_g_lo = {'mae': 999}
    best_g_hi = {'mae': 999}

    for t_pair in all_temp_options:
        for w in weight_range:
            # f low
            f_lo = weighted_avg_at_indices(lo_idx, t_pair, w, 'f')
            f_lo_mae = mae_at_indices(f_lo, exp_f, lo_idx)
            if f_lo_mae < best_f_lo['mae']:
                best_f_lo = {'mae': f_lo_mae, 'temps': t_pair, 'w': w, 'vals': f_lo}

            # f high
            f_hi = weighted_avg_at_indices(hi_idx, t_pair, w, 'f')
            f_hi_mae = mae_at_indices(f_hi, exp_f, hi_idx)
            if f_hi_mae < best_f_hi['mae']:
                best_f_hi = {'mae': f_hi_mae, 'temps': t_pair, 'w': w, 'vals': f_hi}

            # g low
            g_lo = weighted_avg_at_indices(lo_idx, t_pair, w, 'g')
            g_lo_mae = mae_at_indices(g_lo, exp_g, lo_idx)
            if g_lo_mae < best_g_lo['mae']:
                best_g_lo = {'mae': g_lo_mae, 'temps': t_pair, 'w': w, 'vals': g_lo}

            # g high
            g_hi = weighted_avg_at_indices(hi_idx, t_pair, w, 'g')
            g_hi_mae = mae_at_indices(g_hi, exp_g, hi_idx)
            if g_hi_mae < best_g_hi['mae']:
                best_g_hi = {'mae': g_hi_mae, 'temps': t_pair, 'w': w, 'vals': g_hi}

    # 전체 f MAE (가중 합산)
    n_lo, n_hi = len(lo_idx), len(hi_idx)
    n_total = n_lo + n_hi
    f_total_mae = (best_f_lo['mae'] * n_lo + best_f_hi['mae'] * n_hi) / n_total
    g_total_mae = (best_g_lo['mae'] * n_lo + best_g_hi['mae'] * n_hi) / n_total
    combined = (f_total_mae + g_total_mae) / 2

    split_results.append({
        'split_idx': split_idx,
        'threshold': threshold,
        'f_lo': best_f_lo, 'f_hi': best_f_hi,
        'g_lo': best_g_lo, 'g_hi': best_g_hi,
        'f_total': f_total_mae, 'g_total': g_total_mae,
        'combined': combined,
    })

split_results.sort(key=lambda x: x['combined'])

print(f"\n{'Rank':>4s}  {'Threshold':>12s}  {'Split':>5s}  "
      f"{'f_lo MAE%':>10s}  {'f_hi MAE%':>10s}  {'f_tot%':>8s}  "
      f"{'g_lo MAE%':>10s}  {'g_hi MAE%':>10s}  {'g_tot%':>8s}  {'TOTAL%':>8s}")
print("-" * 120)
for i, r in enumerate(split_results):
    print(f"{i+1:4d}  {r['threshold']:12.4e}  {r['split_idx']:5d}  "
          f"{r['f_lo']['mae']:10.2f}  {r['f_hi']['mae']:10.2f}  {r['f_total']:8.2f}  "
          f"{r['g_lo']['mae']:10.2f}  {r['g_hi']['mae']:10.2f}  {r['g_total']:8.2f}  "
          f"{r['combined']:8.2f}")

# ── TOP 3 상세 ──────────────────────────────────
for rank in range(min(3, len(split_results))):
    r = split_results[rank]
    print(f"\n{'=' * 120}")
    print(f"RANK {rank+1}: Split at strain = {r['threshold']:.4e} (index {r['split_idx']})")
    print(f"  f LOW  : T=({r['f_lo']['temps'][0]:.2f}, {r['f_lo']['temps'][1]:.2f}), "
          f"w={r['f_lo']['w']:.2f}, MAE={r['f_lo']['mae']:.2f}%")
    print(f"  f HIGH : T=({r['f_hi']['temps'][0]:.2f}, {r['f_hi']['temps'][1]:.2f}), "
          f"w={r['f_hi']['w']:.2f}, MAE={r['f_hi']['mae']:.2f}%")
    print(f"  g LOW  : T=({r['g_lo']['temps'][0]:.2f}, {r['g_lo']['temps'][1]:.2f}), "
          f"w={r['g_lo']['w']:.2f}, MAE={r['g_lo']['mae']:.2f}%")
    print(f"  g HIGH : T=({r['g_hi']['temps'][0]:.2f}, {r['g_hi']['temps'][1]:.2f}), "
          f"w={r['g_hi']['w']:.2f}, MAE={r['g_hi']['mae']:.2f}%")
    print(f"  f total={r['f_total']:.2f}%, g total={r['g_total']:.2f}%, COMBINED={r['combined']:.2f}%")
    print(f"{'=' * 120}")

    # 상세 출력
    print(f"{'Strain':>12s}  {'Zone':>4s}  "
          f"{'f_calc':>8s}  {'f_exp':>8s}  {'f_err%':>8s}  "
          f"{'g_calc':>8s}  {'g_exp':>8s}  {'g_err%':>8s}")
    print("-" * 80)
    for j in range(len(exp_strain)):
        zone = "LOW" if j < r['split_idx'] else "HIGH"
        if j < r['split_idx']:
            f_c = r['f_lo']['vals'][j]
            g_c = r['g_lo']['vals'][j]
        else:
            hi_j = j - r['split_idx']
            f_c = r['f_hi']['vals'][hi_j]
            g_c = r['g_hi']['vals'][hi_j]
        f_err = (f_c - exp_f[j]) / exp_f[j] * 100 if exp_f[j] != 0 else 0
        g_err = (g_c - exp_g[j]) / exp_g[j] * 100 if exp_g[j] != 0 else 0
        sep = " <<<" if j == r['split_idx'] else ""
        print(f"{exp_strain[j]:12.4e}  {zone:>4s}  "
              f"{f_c:8.4f}  {exp_f[j]:8.3f}  {f_err:+8.2f}  "
              f"{g_c:8.4f}  {exp_g[j]:8.3f}  {g_err:+8.2f}{sep}")

# ── 비교: 기존 방식 vs 구간 분리 ────────────────
print(f"\n{'=' * 120}")
print("비교 요약")
print("=" * 120)
# 기존: 전체 단일 온도 조합
best_single = split_results[-1]  # worst split = essentially no split benefit

# 이전 결과: f=T(9.94,49.99) 50:50 → 2.80%, g=T(29.9,49.99) 55:45 → 1.57%
print(f"  이전 최적 (단일 구간, 독립 f/g):  f=2.80%, g=1.57%, 합산=2.19%")
best = split_results[0]
print(f"  구간 분리 RANK 1:                  f={best['f_total']:.2f}%, g={best['g_total']:.2f}%, 합산={best['combined']:.2f}%")
print(f"  개선율: {(2.19 - best['combined'])/2.19*100:.1f}%")
