"""
Output utilities for Persson model results
===========================================

Provides functions to export calculation results to various formats.
"""

import numpy as np
import csv
from typing import Dict, Optional
import os


def save_calculation_details_csv(
    results: Dict,
    filename: str,
    parameters: Optional[Dict] = None
):
    """
    Save detailed G(q) calculation results to CSV file.

    Creates a CSV file with columns as specified in the work instruction:
    Index, log_q, q, C(q), Avg_Modulus_Term, G_Integrand, Delta_G, G(q), Contact_Area_Ratio

    Parameters
    ----------
    results : dict
        Results dictionary from GCalculator.calculate_G_with_details()
    filename : str
        Output CSV filename
    parameters : dict, optional
        Input parameters to include in header comments
    """
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Write header comments with parameters
        if parameters:
            f.write("# Persson Contact Mechanics Calculation Results\n")
            f.write("# =============================================\n")
            f.write("#\n")
            f.write("# Input Parameters:\n")
            for key, value in parameters.items():
                f.write(f"# {key}: {value}\n")
            f.write("#\n")
            f.write("# Calculation Details:\n")
            f.write(f"# Number of points: {len(results['q'])}\n")
            f.write(f"# q_min: {results['q'][0]:.3e} 1/m\n")
            f.write(f"# q_max: {results['q'][-1]:.3e} 1/m\n")
            f.write(f"# Final G(q_max): {results['G'][-1]:.6e}\n")
            f.write(f"# Final Contact Area: {results['contact_area_ratio'][-1]:.6f}\n")
            f.write("#\n")

        # Write column headers
        headers = [
            'Index',
            'log_q',
            'q (1/m)',
            'C(q) (m^4)',
            'Avg_Modulus_Term',
            'G_Integrand',
            'Delta_G',
            'G(q)',
            'Contact_Area_Ratio (P(q))'
        ]
        writer.writerow(headers)

        # Write data rows
        n = len(results['q'])
        for i in range(n):
            row = [
                i,
                f"{results['log_q'][i]:.6f}",
                f"{results['q'][i]:.6e}",
                f"{results['C_q'][i]:.6e}",
                f"{results['avg_modulus_term'][i]:.6e}",
                f"{results['G_integrand'][i]:.6e}",
                f"{results['delta_G'][i]:.6e}",
                f"{results['G'][i]:.6e}",
                f"{results['contact_area_ratio'][i]:.6f}"
            ]
            writer.writerow(row)


def save_summary_txt(
    results: Dict,
    filename: str,
    parameters: Dict
):
    """
    Save calculation summary to text file.

    Parameters
    ----------
    results : dict
        Results dictionary
    filename : str
        Output text filename
    parameters : dict
        Input parameters
    """
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("Persson Contact Mechanics - Calculation Summary\n")
        f.write("=" * 70 + "\n\n")

        f.write("Input Parameters:\n")
        f.write("-" * 70 + "\n")
        for key, value in parameters.items():
            f.write(f"  {key:30s}: {value}\n")
        f.write("\n")

        f.write("Calculation Results:\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Number of wavenumber points  : {len(results['q'])}\n")
        f.write(f"  q_min (1/m)                  : {results['q'][0]:.3e}\n")
        f.write(f"  q_max (1/m)                  : {results['q'][-1]:.3e}\n")
        f.write(f"  Magnification ζ_max          : {results['q'][-1]/results['q'][0]:.2f}\n")
        f.write(f"\n")
        f.write(f"  Final G(q_max)               : {results['G'][-1]:.6e}\n")
        f.write(f"  Contact Area Ratio (P)       : {results['contact_area_ratio'][-1]:.6f}\n")
        f.write(f"  Contact Area Percentage      : {results['contact_area_ratio'][-1]*100:.3f}%\n")
        f.write("\n")

        # Find point where contact area reaches certain thresholds
        P = results['contact_area_ratio']
        q = results['q']

        thresholds = [0.1, 0.5, 0.9]
        f.write("Contact Area Thresholds:\n")
        f.write("-" * 70 + "\n")
        for threshold in thresholds:
            idx = np.argmin(np.abs(P - threshold))
            if idx > 0:
                f.write(f"  P = {threshold:.1f} reached at q = {q[idx]:.3e} 1/m (ζ = {q[idx]/q[0]:.2f})\n")
        f.write("\n")

        f.write("=" * 70 + "\n")


def export_for_plotting(
    results: Dict,
    output_dir: str = '.',
    prefix: str = 'persson'
):
    """
    Export data files suitable for plotting with external tools.

    Creates separate CSV files for different plots:
    - G vs q
    - Contact area vs magnification
    - PSD vs q
    - etc.

    Parameters
    ----------
    results : dict
        Results dictionary
    output_dir : str, optional
        Output directory
    prefix : str, optional
        Prefix for output filenames
    """
    os.makedirs(output_dir, exist_ok=True)

    # G(q) data
    g_file = os.path.join(output_dir, f'{prefix}_G_vs_q.csv')
    with open(g_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['q (1/m)', 'log_q', 'G(q)', 'log_G'])
        for i in range(len(results['q'])):
            log_G = np.log10(results['G'][i]) if results['G'][i] > 0 else -np.inf
            writer.writerow([
                results['q'][i],
                results['log_q'][i],
                results['G'][i],
                log_G
            ])

    # Contact area data
    p_file = os.path.join(output_dir, f'{prefix}_contact_area.csv')
    with open(p_file, 'w', newline='') as f:
        writer = csv.writer(f)
        zeta = results['q'] / results['q'][0]
        writer.writerow(['Magnification_zeta', 'log_zeta', 'Contact_Area_Ratio', 'Percentage'])
        for i in range(len(results['q'])):
            writer.writerow([
                zeta[i],
                np.log10(zeta[i]),
                results['contact_area_ratio'][i],
                results['contact_area_ratio'][i] * 100
            ])

    # PSD data
    psd_file = os.path.join(output_dir, f'{prefix}_PSD.csv')
    with open(psd_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['q (1/m)', 'log_q', 'C(q) (m^4)', 'log_C'])
        for i in range(len(results['q'])):
            log_C = np.log10(results['C_q'][i]) if results['C_q'][i] > 0 else -np.inf
            writer.writerow([
                results['q'][i],
                results['log_q'][i],
                results['C_q'][i],
                log_C
            ])

    print(f"Exported plotting data to {output_dir}/")
    print(f"  - {os.path.basename(g_file)}")
    print(f"  - {os.path.basename(p_file)}")
    print(f"  - {os.path.basename(psd_file)}")


def format_parameters_dict(
    sigma_0: float,
    velocity: float,
    temperature: float,
    poisson_ratio: float,
    q_min: float,
    q_max: float,
    material_name: str = "Unknown",
    **kwargs
) -> Dict:
    """
    Format parameters into a dictionary for output.

    Parameters
    ----------
    sigma_0 : float
        Nominal pressure (Pa)
    velocity : float
        Sliding velocity (m/s)
    temperature : float
        Temperature (°C)
    poisson_ratio : float
        Poisson's ratio
    q_min : float
        Minimum wavenumber (1/m)
    q_max : float
        Maximum wavenumber (1/m)
    material_name : str, optional
        Material name
    **kwargs
        Additional parameters

    Returns
    -------
    dict
        Formatted parameters dictionary
    """
    params = {
        'Material': material_name,
        'Nominal Pressure (Pa)': f"{sigma_0:.3e}",
        'Nominal Pressure (MPa)': f"{sigma_0/1e6:.3f}",
        'Sliding Velocity (m/s)': f"{velocity:.4f}",
        'Temperature (°C)': f"{temperature:.1f}",
        'Poisson Ratio': f"{poisson_ratio:.3f}",
        'q_min (1/m)': f"{q_min:.3e}",
        'q_max (1/m)': f"{q_max:.3e}",
        'Magnification Range': f"{q_max/q_min:.2f}",
    }

    # Add any additional parameters
    for key, value in kwargs.items():
        params[key] = str(value)

    return params
