"""
상세 출력 예제 (Detailed Output Example)
=========================================

작업 지시서에 따라 모든 중간 계산값을 CSV로 출력하는 예제입니다.

출력 파일:
- persson_detailed_results.csv: 전체 계산 과정 (Index, log_q, q, C(q),
  Avg_Modulus_Term, G_Integrand, Delta_G, G(q), Contact_Area_Ratio)
- persson_summary.txt: 계산 결과 요약
- persson_*_plot_data.csv: 개별 그래프용 데이터
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from persson_model import GCalculator, FractalPSD, ViscoelasticMaterial
from persson_model.utils import (
    save_calculation_details_csv,
    save_summary_txt,
    export_for_plotting,
    format_parameters_dict
)


def main():
    print("=" * 80)
    print("Persson Contact Mechanics - 상세 출력 예제")
    print("=" * 80)

    # ========================================================================
    # 1. 입력 파라미터 설정 (Input Parameters)
    # ========================================================================
    print("\n[1] 입력 파라미터 설정")
    print("-" * 80)

    # 제어 변수 (Global Variables)
    sigma_0 = 1.0e6      # Nominal pressure (Pa) = 1.0 MPa
    velocity = 0.01      # Sliding velocity (m/s)
    temperature = 20.0   # Temperature (°C)
    poisson = 0.5        # Poisson's ratio

    # 파수 범위 (Wavenumber Range)
    q_min = 2 * np.pi / 0.01  # ~ 628 1/m (tread block size ~ 1 cm)
    q_max = 1e7               # 10^7 1/m (cut-off)

    print(f"  명목 압력:          {sigma_0/1e6:.2f} MPa")
    print(f"  미끄럼 속도:        {velocity:.4f} m/s")
    print(f"  온도:              {temperature:.1f} °C")
    print(f"  Poisson 비:        {poisson:.2f}")
    print(f"  파수 범위:          {q_min:.2e} ~ {q_max:.2e} 1/m")
    print(f"  배율 범위:          1 ~ {q_max/q_min:.2e}")

    # ========================================================================
    # 2. 재료 물성 설정 (Material Properties)
    # ========================================================================
    print("\n[2] 재료 물성 로드")
    print("-" * 80)

    material = ViscoelasticMaterial.create_example_sbr()
    print(f"  재료명:            {material.name}")
    print(f"  기준 온도:          {material.reference_temp}°C")
    print(f"  주파수 범위:        {material._frequencies[0]:.2e} ~ {material._frequencies[-1]:.2e} rad/s")

    # ========================================================================
    # 3. 표면 거칠기 설정 (Surface Roughness PSD)
    # ========================================================================
    print("\n[3] 표면 거칠기 PSD 모델 생성")
    print("-" * 80)

    hurst_exponent = 0.8
    rms_roughness = 10e-6  # 10 μm

    psd = FractalPSD(
        hurst_exponent=hurst_exponent,
        rms_roughness=rms_roughness,
        q_min=q_min,
        q_max=q_max
    )

    print(f"  PSD 모델:          Fractal (Self-affine)")
    print(f"  Hurst 지수 H:      {hurst_exponent:.2f}")
    print(f"  RMS 거칠기:        {rms_roughness*1e6:.2f} μm")
    print(f"  PSD 지수:          -2(1+H) = {-2*(1+hurst_exponent):.2f}")

    # Verify RMS
    h_rms_calc = psd.get_rms_roughness(q_min, q_max)
    print(f"  검증 RMS:          {h_rms_calc*1e6:.2f} μm")

    # ========================================================================
    # 4. G(q) 계산 (Core Calculation)
    # ========================================================================
    print("\n[4] G(q) 계산 실행")
    print("-" * 80)

    # Create G calculator
    g_calc = GCalculator(
        psd_func=psd,
        modulus_func=lambda w: material.get_modulus(w, temperature=temperature),
        sigma_0=sigma_0,
        velocity=velocity,
        poisson_ratio=poisson,
        n_angle_points=36,
        integration_method='trapz'
    )

    # Create wavenumber array (logarithmic spacing)
    n_points = 100
    q_values = np.logspace(np.log10(q_min), np.log10(q_max), n_points)

    print(f"  계산 포인트 수:     {n_points}")
    print(f"  각도 적분 포인트:   {g_calc.n_angle_points}")
    print(f"  적분 방법:         {g_calc.integration_method}")
    print()
    print("  계산 중...")

    # Calculate with detailed intermediate values
    results = g_calc.calculate_G_with_details(q_values, q_min=q_min)

    print(f"  ✓ 계산 완료!")
    print()
    print(f"  최종 결과:")
    print(f"    G(q_max)         = {results['G'][-1]:.6e}")
    print(f"    P(q_max)         = {results['contact_area_ratio'][-1]:.6f}")
    print(f"    접촉 면적 비율    = {results['contact_area_ratio'][-1]*100:.3f}%")

    # ========================================================================
    # 5. 결과 저장 (Save Results)
    # ========================================================================
    print("\n[5] 결과 파일 저장")
    print("-" * 80)

    # Format parameters for output
    params = format_parameters_dict(
        sigma_0=sigma_0,
        velocity=velocity,
        temperature=temperature,
        poisson_ratio=poisson,
        q_min=q_min,
        q_max=q_max,
        material_name=material.name,
        hurst_exponent=hurst_exponent,
        rms_roughness_um=rms_roughness*1e6,
        n_points=n_points,
        n_angle_points=g_calc.n_angle_points
    )

    # Save detailed CSV
    csv_file = 'persson_detailed_results.csv'
    save_calculation_details_csv(results, csv_file, params)
    print(f"  ✓ {csv_file}")
    print(f"    (Index, log_q, q, C(q), Avg_Modulus_Term, G_Integrand, Delta_G, G, P)")

    # Save summary
    summary_file = 'persson_summary.txt'
    save_summary_txt(results, summary_file, params)
    print(f"  ✓ {summary_file}")

    # Export plotting data
    export_for_plotting(results, output_dir='output', prefix='persson')
    print(f"  ✓ output/persson_*.csv (plotting data)")

    # ========================================================================
    # 6. 시각화 (Visualization)
    # ========================================================================
    print("\n[6] 결과 시각화")
    print("-" * 80)

    fig = plt.figure(figsize=(16, 12))

    # Subplot 1: G(q) vs q
    ax1 = plt.subplot(3, 3, 1)
    zeta = q_values / q_min
    ax1.loglog(zeta, results['G'], 'b-', linewidth=2)
    ax1.set_xlabel('Magnification ζ = q/q₀', fontsize=10)
    ax1.set_ylabel('G(ζ)', fontsize=10)
    ax1.set_title('(a) Persson G Function', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Contact Area vs Magnification
    ax2 = plt.subplot(3, 3, 2)
    ax2.semilogx(zeta, results['contact_area_ratio'], 'r-', linewidth=2)
    ax2.set_xlabel('Magnification ζ', fontsize=10)
    ax2.set_ylabel('Contact Area Ratio P(q)', fontsize=10)
    ax2.set_title('(b) Contact Area vs Magnification', fontsize=11, fontweight='bold')
    ax2.set_ylim([0, 1.1])
    ax2.grid(True, alpha=0.3)
    ax2.axhline(results['contact_area_ratio'][-1], color='k', linestyle='--',
                label=f'Final: {results["contact_area_ratio"][-1]:.4f}')
    ax2.legend()

    # Subplot 3: PSD C(q)
    ax3 = plt.subplot(3, 3, 3)
    ax3.loglog(q_values, results['C_q'], 'g-', linewidth=2)
    ax3.set_xlabel('Wavenumber q (1/m)', fontsize=10)
    ax3.set_ylabel('PSD C(q) (m⁴)', fontsize=10)
    ax3.set_title('(c) Surface Roughness PSD', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    # Add slope reference line
    q_ref = np.array([q_min, q_max])
    C_ref = results['C_q'][0] * (q_ref / q_min)**(-2*(1+hurst_exponent))
    ax3.loglog(q_ref, C_ref, 'k--', alpha=0.5,
               label=f'Slope: -2(1+H) = {-2*(1+hurst_exponent):.1f}')
    ax3.legend()

    # Subplot 4: G Integrand
    ax4 = plt.subplot(3, 3, 4)
    ax4.loglog(q_values, results['G_integrand'], 'purple', linewidth=2)
    ax4.set_xlabel('Wavenumber q (1/m)', fontsize=10)
    ax4.set_ylabel('G Integrand: q³C(q)∫dφ|E|²', fontsize=10)
    ax4.set_title('(d) G Integrand', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # Subplot 5: Delta_G
    ax5 = plt.subplot(3, 3, 5)
    ax5.loglog(q_values[1:], results['delta_G'][1:], 'orange', linewidth=2)
    ax5.set_xlabel('Wavenumber q (1/m)', fontsize=10)
    ax5.set_ylabel('ΔG (incremental contribution)', fontsize=10)
    ax5.set_title('(e) Incremental G Contribution', fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3)

    # Subplot 6: Avg Modulus Term
    ax6 = plt.subplot(3, 3, 6)
    ax6.loglog(q_values, results['avg_modulus_term'], 'brown', linewidth=2)
    ax6.set_xlabel('Wavenumber q (1/m)', fontsize=10)
    ax6.set_ylabel('∫dφ |E/(1-ν²)σ₀|²', fontsize=10)
    ax6.set_title('(f) Avg Modulus Term (Angle Integral)', fontsize=11, fontweight='bold')
    ax6.grid(True, alpha=0.3)

    # Subplot 7: Cumulative G
    ax7 = plt.subplot(3, 3, 7)
    ax7.semilogx(q_values, results['G'], 'b-', linewidth=2)
    ax7.set_xlabel('Wavenumber q (1/m)', fontsize=10)
    ax7.set_ylabel('Cumulative G(q)', fontsize=10)
    ax7.set_title('(g) Cumulative G(q)', fontsize=11, fontweight='bold')
    ax7.grid(True, alpha=0.3)

    # Subplot 8: Material Master Curve
    ax8 = plt.subplot(3, 3, 8)
    freq = np.logspace(-2, 6, 100)
    E_storage = material.get_storage_modulus(freq, temperature)
    E_loss = material.get_loss_modulus(freq, temperature)
    ax8.loglog(freq, E_storage, 'g-', linewidth=2, label="E' (Storage)")
    ax8.loglog(freq, E_loss, 'orange', linewidth=2, label="E'' (Loss)")
    ax8.set_xlabel('Frequency ω (rad/s)', fontsize=10)
    ax8.set_ylabel('Modulus (Pa)', fontsize=10)
    ax8.set_title('(h) Material Master Curve', fontsize=11, fontweight='bold')
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    # Subplot 9: Summary Text
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    summary_text = f"""
계산 결과 요약
{'='*40}

입력 조건:
  σ₀ = {sigma_0/1e6:.2f} MPa
  v  = {velocity:.4f} m/s
  T  = {temperature:.1f}°C
  ν  = {poisson:.2f}

표면 거칠기:
  H     = {hurst_exponent:.2f}
  h_rms = {rms_roughness*1e6:.2f} μm
  q_min = {q_min:.2e} 1/m
  q_max = {q_max:.2e} 1/m

계산 결과:
  G(q_max) = {results['G'][-1]:.6e}
  P(q_max) = {results['contact_area_ratio'][-1]:.6f}

실 접촉 면적:
  {results['contact_area_ratio'][-1]*100:.3f}%

물리적 의미:
  거칠기로 인해 실제 접촉하는
  면적은 명목 면적의 약
  {results['contact_area_ratio'][-1]*100:.1f}%에 불과함
    """
    ax9.text(0.05, 0.5, summary_text, fontsize=9, family='monospace',
             verticalalignment='center', transform=ax9.transAxes)

    plt.suptitle('Persson Contact Mechanics - Detailed Calculation Results',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('persson_detailed_visualization.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ persson_detailed_visualization.png")

    plt.show()

    # ========================================================================
    # 7. 중간값 검증 (Intermediate Values Verification)
    # ========================================================================
    print("\n[7] 중간값 샘플 출력 (처음 5개, 마지막 5개)")
    print("-" * 80)
    print(f"{'Index':>6} {'log_q':>10} {'q (1/m)':>12} {'C(q)':>12} {'Avg_Mod':>12} {'G_Int':>12} {'ΔG':>12} {'G(q)':>12} {'P(q)':>8}")
    print("-" * 80)

    # First 5
    for i in range(min(5, len(q_values))):
        print(f"{i:6d} {results['log_q'][i]:10.4f} {results['q'][i]:12.4e} "
              f"{results['C_q'][i]:12.4e} {results['avg_modulus_term'][i]:12.4e} "
              f"{results['G_integrand'][i]:12.4e} {results['delta_G'][i]:12.4e} "
              f"{results['G'][i]:12.4e} {results['contact_area_ratio'][i]:8.5f}")

    if len(q_values) > 10:
        print("   ...")

    # Last 5
    for i in range(max(0, len(q_values)-5), len(q_values)):
        print(f"{i:6d} {results['log_q'][i]:10.4f} {results['q'][i]:12.4e} "
              f"{results['C_q'][i]:12.4e} {results['avg_modulus_term'][i]:12.4e} "
              f"{results['G_integrand'][i]:12.4e} {results['delta_G'][i]:12.4e} "
              f"{results['G'][i]:12.4e} {results['contact_area_ratio'][i]:8.5f}")

    print("\n" + "=" * 80)
    print("계산 완료! 출력 파일을 확인하세요.")
    print("=" * 80)


if __name__ == "__main__":
    main()
