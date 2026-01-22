"""
Enhanced Main GUI for Persson Friction Model (v2.1)
===================================================

Work Instruction v2.1 Implementation:
- Velocity range: 0.0001~10 m/s (log scale)
- G(q,v) 2D matrix calculation
- Input data verification tab
- Multi-velocity G(q) plotting
- Default measured data loading
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from typing import Optional
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from persson_model.core.g_calculator import GCalculator
from persson_model.core.psd_models import FractalPSD, MeasuredPSD
from persson_model.core.viscoelastic import ViscoelasticMaterial
from persson_model.core.contact import ContactMechanics
from persson_model.utils.output import (
    save_calculation_details_csv,
    save_summary_txt,
    export_for_plotting,
    format_parameters_dict
)
from persson_model.utils.data_loader import (
    load_psd_from_file,
    load_dma_from_file,
    create_material_from_dma,
    create_psd_from_data
)

# Configure matplotlib for better Korean font support
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 12


class PerssonModelGUI_V2:
    """Enhanced GUI for Persson friction model (Work Instruction v2.1)."""

    def __init__(self, root):
        """Initialize enhanced GUI."""
        self.root = root
        self.root.title("Persson Friction Calculator v2.1")
        self.root.geometry("1600x1000")

        # Initialize variables
        self.material = None
        self.psd_model = None
        self.g_calculator = None
        self.results = {}

        # Create UI
        self._create_menu()
        self._create_main_layout()
        self._create_status_bar()

        # Load default measured data
        self._load_default_data()

    def _load_default_data(self):
        """Load default measured data on startup."""
        try:
            # Get data directory
            base_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(base_dir, 'examples', 'data')

            psd_file = os.path.join(data_dir, 'measured_psd.txt')
            dma_file = os.path.join(data_dir, 'measured_dma.txt')

            if os.path.exists(psd_file) and os.path.exists(dma_file):
                # Load DMA data
                omega, E_storage, E_loss = load_dma_from_file(
                    dma_file,
                    skip_header=1,
                    freq_unit='Hz',
                    modulus_unit='MPa'
                )

                self.material = create_material_from_dma(
                    omega=omega,
                    E_storage=E_storage,
                    E_loss=E_loss,
                    material_name="Measured Rubber",
                    reference_temp=20.0
                )

                # Load PSD data
                q, C_q = load_psd_from_file(psd_file, skip_header=1)
                self.psd_model = create_psd_from_data(q, C_q, interpolation_kind='log-log')

                # Update UI
                self.q_min_var.set(f"{q[0]:.2e}")
                self.q_max_var.set(f"{q[-1]:.2e}")
                self.psd_type_var.set("measured")

                self._update_material_display()
                self._update_verification_plots()

                self.status_var.set(f"초기 데이터 로드 완료: PSD ({len(q)}개), DMA ({len(omega)}개)")
            else:
                # Use example material
                self.material = ViscoelasticMaterial.create_example_sbr()
                self._update_material_display()
                self.status_var.set("예제 재료 (SBR) 로드됨")

        except Exception as e:
            print(f"Default data loading error: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to example material
            self.material = ViscoelasticMaterial.create_example_sbr()
            self.status_var.set("예제 재료 (SBR) 로드됨")

    def _create_menu(self):
        """Create menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load DMA Data", command=self._load_material)
        file_menu.add_command(label="Load PSD Data", command=self._load_psd_data)
        file_menu.add_separator()
        file_menu.add_command(label="Save Results (CSV)", command=self._save_detailed_csv)
        file_menu.add_command(label="Export All", command=self._export_all_results)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="User Guide", command=self._show_help)
        help_menu.add_command(label="About", command=self._show_about)

    def _create_main_layout(self):
        """Create main application layout with tabs."""
        # Create notebook (tabbed interface)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Tab 1: Input Data Verification
        self.tab_verification = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_verification, text="1. Input Data Verification")
        self._create_verification_tab(self.tab_verification)

        # Tab 2: Calculation Parameters
        self.tab_parameters = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_parameters, text="2. Calculation Setup")
        self._create_parameters_tab(self.tab_parameters)

        # Tab 3: G(q,v) Results
        self.tab_results = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_results, text="3. G(q,v) Results")
        self._create_results_tab(self.tab_results)

        # Tab 4: Friction Coefficient
        self.tab_friction = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_friction, text="4. Friction Analysis")
        self._create_friction_tab(self.tab_friction)

    def _create_verification_tab(self, parent):
        """Create input data verification tab."""
        # Instruction label
        instruction = ttk.LabelFrame(parent, text="Tab Purpose", padding=10)
        instruction.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(instruction, text=
            "This tab verifies that material properties and surface roughness data\n"
            "are correctly loaded before calculation. Check E', E'', tan(δ) and C(q).",
            font=('Arial', 10)
        ).pack()

        # Plot area
        plot_frame = ttk.Frame(parent)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Create figure with 2 subplots
        self.fig_verification = Figure(figsize=(14, 6), dpi=100)

        self.ax_master_curve = self.fig_verification.add_subplot(121)
        self.ax_psd = self.fig_verification.add_subplot(122)

        self.canvas_verification = FigureCanvasTkAgg(self.fig_verification, plot_frame)
        self.canvas_verification.draw()
        self.canvas_verification.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas_verification, plot_frame)
        toolbar.update()

        # Refresh button
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(
            btn_frame,
            text="Refresh Plots",
            command=self._update_verification_plots
        ).pack(side=tk.LEFT, padx=5)

    def _create_parameters_tab(self, parent):
        """Create calculation parameters tab."""
        # Instruction
        instruction = ttk.LabelFrame(parent, text="Tab Purpose", padding=10)
        instruction.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(instruction, text=
            "Set calculation parameters: pressure, velocity range (log scale), temperature.\n"
            "Velocity range: 0.0001~10 m/s (log spaced) for frequency sweep.",
            font=('Arial', 10)
        ).pack()

        # Input panel
        input_frame = ttk.LabelFrame(parent, text="Calculation Parameters", padding=10)
        input_frame.pack(fill=tk.X, padx=10, pady=5)

        # Create input fields
        row = 0

        # Nominal pressure
        ttk.Label(input_frame, text="Nominal Pressure (MPa):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.sigma_0_var = tk.StringVar(value="1.0")
        ttk.Entry(input_frame, textvariable=self.sigma_0_var, width=15).grid(row=row, column=1, pady=5)

        # Velocity range
        row += 1
        ttk.Label(input_frame, text="Velocity Range:").grid(row=row, column=0, sticky=tk.W, pady=5)
        ttk.Label(input_frame, text="Log scale: 0.0001~10 m/s").grid(row=row, column=1, sticky=tk.W, pady=5)

        row += 1
        ttk.Label(input_frame, text="  v_min (m/s):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.v_min_var = tk.StringVar(value="0.0001")
        ttk.Entry(input_frame, textvariable=self.v_min_var, width=15).grid(row=row, column=1, pady=5)

        row += 1
        ttk.Label(input_frame, text="  v_max (m/s):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.v_max_var = tk.StringVar(value="10.0")
        ttk.Entry(input_frame, textvariable=self.v_max_var, width=15).grid(row=row, column=1, pady=5)

        row += 1
        ttk.Label(input_frame, text="  Number of velocities:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.n_velocity_var = tk.StringVar(value="30")
        ttk.Entry(input_frame, textvariable=self.n_velocity_var, width=15).grid(row=row, column=1, pady=5)

        # Temperature
        row += 1
        ttk.Label(input_frame, text="Temperature (°C):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.temperature_var = tk.StringVar(value="20")
        ttk.Entry(input_frame, textvariable=self.temperature_var, width=15).grid(row=row, column=1, pady=5)

        # Poisson ratio
        row += 1
        ttk.Label(input_frame, text="Poisson Ratio:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.poisson_var = tk.StringVar(value="0.5")
        ttk.Entry(input_frame, textvariable=self.poisson_var, width=15).grid(row=row, column=1, pady=5)

        # Wavenumber range
        row += 1
        ttk.Label(input_frame, text="q_min (1/m):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.q_min_var = tk.StringVar(value="2e1")
        ttk.Entry(input_frame, textvariable=self.q_min_var, width=15).grid(row=row, column=1, pady=5)

        row += 1
        ttk.Label(input_frame, text="q_max (1/m):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.q_max_var = tk.StringVar(value="1e9")
        ttk.Entry(input_frame, textvariable=self.q_max_var, width=15).grid(row=row, column=1, pady=5)

        row += 1
        ttk.Label(input_frame, text="Number of q points:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.n_q_var = tk.StringVar(value="100")
        ttk.Entry(input_frame, textvariable=self.n_q_var, width=15).grid(row=row, column=1, pady=5)

        # PSD type
        row += 1
        ttk.Label(input_frame, text="PSD Type:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.psd_type_var = tk.StringVar(value="measured")
        ttk.Combobox(
            input_frame,
            textvariable=self.psd_type_var,
            values=["measured", "fractal"],
            state="readonly",
            width=12
        ).grid(row=row, column=1, pady=5)

        # Calculate button
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)

        self.calc_button = ttk.Button(
            btn_frame,
            text="Run G(q,v) Calculation",
            command=self._run_calculation
        )
        self.calc_button.pack(fill=tk.X, pady=5)

        # Progress bar
        self.progress_var = tk.IntVar()
        self.progress_bar = ttk.Progressbar(
            btn_frame,
            variable=self.progress_var,
            maximum=100
        )
        self.progress_bar.pack(fill=tk.X, pady=5)

    def _create_results_tab(self, parent):
        """Create G(q,v) results tab."""
        # Instruction
        instruction = ttk.LabelFrame(parent, text="Tab Purpose", padding=10)
        instruction.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(instruction, text=
            "G(q,v) 2D matrix results: Multi-velocity G(q) curves, heatmap, and contact area.\n"
            "All velocities are plotted on the same graph with color coding.",
            font=('Arial', 10)
        ).pack()

        # Plot area
        plot_frame = ttk.Frame(parent)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.fig_results = Figure(figsize=(16, 10), dpi=100)
        self.canvas_results = FigureCanvasTkAgg(self.fig_results, plot_frame)
        self.canvas_results.draw()
        self.canvas_results.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas_results, plot_frame)
        toolbar.update()

    def _create_friction_tab(self, parent):
        """Create friction coefficient analysis tab."""
        # Instruction
        instruction = ttk.LabelFrame(parent, text="Tab Purpose", padding=10)
        instruction.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(instruction, text=
            "Friction coefficient μ(v) and contact area analysis.\n"
            "Results show velocity-dependent friction and real contact area ratio.",
            font=('Arial', 10)
        ).pack()

        # Results text
        text_frame = ttk.Frame(parent)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.results_text = tk.Text(text_frame, font=("Courier", 10))
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(text_frame, command=self.results_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.config(yscrollcommand=scrollbar.set)

    def _create_status_bar(self):
        """Create status bar."""
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(
            self.root,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _update_verification_plots(self):
        """Update input data verification plots."""
        if self.material is None:
            return

        # Clear previous plots
        self.ax_master_curve.clear()
        self.ax_psd.clear()

        # Plot 1: Master Curve (E', E'', tan δ)
        omega = np.logspace(-2, 12, 200)
        E_storage = self.material.get_storage_modulus(omega)
        E_loss = self.material.get_loss_modulus(omega)
        tan_delta = E_loss / E_storage

        ax1 = self.ax_master_curve
        ax1_twin = ax1.twinx()

        ax1.loglog(omega, E_storage/1e6, 'g-', linewidth=2, label="E' (Storage)")
        ax1.loglog(omega, E_loss/1e6, 'orange', linewidth=2, label="E'' (Loss)")
        ax1_twin.semilogx(omega, tan_delta, 'r--', linewidth=2, label="tan(δ)")

        ax1.set_xlabel('Angular Frequency ω (rad/s)', fontweight='bold')
        ax1.set_ylabel('Modulus (MPa)', fontweight='bold', color='g')
        ax1_twin.set_ylabel('tan(δ)', fontweight='bold', color='r')
        ax1.set_title('Viscoelastic Master Curve', fontweight='bold', fontsize=12)
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # Plot 2: PSD C(q)
        if self.psd_model is not None:
            q_min = float(self.q_min_var.get())
            q_max = float(self.q_max_var.get())
            q_plot = np.logspace(np.log10(q_min), np.log10(q_max), 200)
            C_q = self.psd_model(q_plot)

            self.ax_psd.loglog(q_plot, C_q, 'b-', linewidth=2, label='Loaded PSD')
            self.ax_psd.set_xlabel('Wavenumber q (1/m)', fontweight='bold')
            self.ax_psd.set_ylabel('PSD C(q) (m⁴)', fontweight='bold')
            self.ax_psd.set_title('Surface Roughness PSD', fontweight='bold', fontsize=12)
            self.ax_psd.legend()
            self.ax_psd.grid(True, alpha=0.3)

        self.fig_verification.tight_layout()
        self.canvas_verification.draw()

    def _update_material_display(self):
        """Update material information (if needed)."""
        pass  # Simplified for now

    def _load_material(self):
        """Load DMA data from file."""
        filename = filedialog.askopenfilename(
            title="Select DMA Data File",
            filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if filename:
            try:
                omega, E_storage, E_loss = load_dma_from_file(
                    filename, skip_header=1, freq_unit='Hz', modulus_unit='MPa'
                )

                self.material = create_material_from_dma(
                    omega, E_storage, E_loss,
                    material_name=os.path.splitext(os.path.basename(filename))[0],
                    reference_temp=float(self.temperature_var.get())
                )

                self._update_verification_plots()
                messagebox.showinfo("Success", f"DMA data loaded: {len(omega)} points")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load DMA data:\n{str(e)}")

    def _load_psd_data(self):
        """Load PSD data from file."""
        filename = filedialog.askopenfilename(
            title="Select PSD Data File",
            filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if filename:
            try:
                q, C_q = load_psd_from_file(filename, skip_header=1)
                self.psd_model = create_psd_from_data(q, C_q, interpolation_kind='log-log')

                self.q_min_var.set(f"{q[0]:.2e}")
                self.q_max_var.set(f"{q[-1]:.2e}")
                self.psd_type_var.set("measured")

                self._update_verification_plots()
                messagebox.showinfo("Success", f"PSD data loaded: {len(q)} points")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load PSD data:\n{str(e)}")

    def _run_calculation(self):
        """Run G(q,v) 2D calculation."""
        if self.material is None or self.psd_model is None:
            messagebox.showwarning("Warning", "Please load material and PSD data first!")
            return

        try:
            self.status_var.set("Calculating G(q,v)...")
            self.calc_button.config(state='disabled')
            self.root.update()

            # Get parameters
            sigma_0 = float(self.sigma_0_var.get()) * 1e6  # MPa to Pa
            v_min = float(self.v_min_var.get())
            v_max = float(self.v_max_var.get())
            n_v = int(self.n_velocity_var.get())
            q_min = float(self.q_min_var.get())
            q_max = float(self.q_max_var.get())
            n_q = int(self.n_q_var.get())
            poisson = float(self.poisson_var.get())
            temperature = float(self.temperature_var.get())

            # Create arrays
            v_array = np.logspace(np.log10(v_min), np.log10(v_max), n_v)
            q_array = np.logspace(np.log10(q_min), np.log10(q_max), n_q)

            # Create G calculator
            self.g_calculator = GCalculator(
                psd_func=self.psd_model,
                modulus_func=lambda w: self.material.get_modulus(w, temperature=temperature),
                sigma_0=sigma_0,
                velocity=v_array[0],  # Initial velocity
                poisson_ratio=poisson,
                n_angle_points=36,
                integration_method='trapz'
            )

            # Calculate G(q,v) 2D matrix
            def progress_callback(percent):
                self.progress_var.set(percent)
                self.root.update()

            results_2d = self.g_calculator.calculate_G_multi_velocity(
                q_array, v_array, q_min=q_min, progress_callback=progress_callback
            )

            self.results = {
                '2d_results': results_2d,
                'sigma_0': sigma_0,
                'temperature': temperature,
                'poisson': poisson
            }

            # Plot results
            self._plot_g_results()

            self.status_var.set("Calculation complete!")
            self.calc_button.config(state='normal')
            messagebox.showinfo("Success", f"G(q,v) calculated for {n_v} velocities and {n_q} wavenumbers")

        except Exception as e:
            self.calc_button.config(state='normal')
            messagebox.showerror("Error", f"Calculation failed:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def _plot_g_results(self):
        """Plot G(q,v) 2D results."""
        self.fig_results.clear()

        results_2d = self.results['2d_results']
        q = results_2d['q']
        v = results_2d['v']
        G_matrix = results_2d['G_matrix']
        P_matrix = results_2d['P_matrix']

        # Create 2x2 subplot layout
        ax1 = self.fig_results.add_subplot(2, 2, 1)
        ax2 = self.fig_results.add_subplot(2, 2, 2)
        ax3 = self.fig_results.add_subplot(2, 2, 3)
        ax4 = self.fig_results.add_subplot(2, 2, 4)

        # Plot 1: Multi-velocity G(q) curves
        cmap = plt.get_cmap('viridis')
        colors = [cmap(i / len(v)) for i in range(len(v))]

        for j, (v_val, color) in enumerate(zip(v, colors)):
            if j % max(1, len(v) // 10) == 0:  # Plot every 10th curve
                ax1.loglog(q, G_matrix[:, j], color=color, linewidth=1.5,
                          label=f'v={v_val:.4f} m/s')

        ax1.set_xlabel('Wavenumber q (1/m)', fontweight='bold')
        ax1.set_ylabel('G(q)', fontweight='bold')
        ax1.set_title('G(q) for Multiple Velocities', fontweight='bold')
        ax1.legend(fontsize=8, ncol=2)
        ax1.grid(True, alpha=0.3)

        # Plot 2: G(q,v) Heatmap
        zeta = q / q[0]
        im = ax2.pcolormesh(np.log10(v), np.log10(zeta), np.log10(G_matrix + 1e-20),
                            cmap='hot', shading='auto')
        ax2.set_xlabel('log₁₀(v) [m/s]', fontweight='bold')
        ax2.set_ylabel('log₁₀(ζ)', fontweight='bold')
        ax2.set_title('G(q,v) Heatmap', fontweight='bold')
        cbar = self.fig_results.colorbar(im, ax=ax2)
        cbar.set_label('log₁₀(G)', fontweight='bold')

        # Plot 3: Contact Area P(q,v)
        for j, (v_val, color) in enumerate(zip(v, colors)):
            if j % max(1, len(v) // 10) == 0:
                ax3.semilogx(q, P_matrix[:, j], color=color, linewidth=1.5,
                            label=f'v={v_val:.4f} m/s')

        ax3.set_xlabel('Wavenumber q (1/m)', fontweight='bold')
        ax3.set_ylabel('Contact Area Ratio P(q)', fontweight='bold')
        ax3.set_title('Contact Area for Multiple Velocities', fontweight='bold')
        ax3.legend(fontsize=8, ncol=2)
        ax3.grid(True, alpha=0.3)

        # Plot 4: Final contact area vs velocity
        P_final = P_matrix[-1, :]
        ax4.semilogx(v, P_final, 'ro-', linewidth=2, markersize=4)
        ax4.set_xlabel('Velocity v (m/s)', fontweight='bold')
        ax4.set_ylabel('Final Contact Area P(q_max)', fontweight='bold')
        ax4.set_title('Contact Area vs Velocity', fontweight='bold')
        ax4.grid(True, alpha=0.3)

        self.fig_results.suptitle('G(q,v) 2D Matrix Results', fontweight='bold', fontsize=14)
        self.fig_results.tight_layout()
        self.canvas_results.draw()

    def _save_detailed_csv(self):
        """Save detailed CSV results."""
        if not self.results:
            messagebox.showwarning("Warning", "No results to save. Run calculation first!")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if filename:
            # Implementation here
            messagebox.showinfo("Info", "CSV save functionality to be implemented")

    def _export_all_results(self):
        """Export all results."""
        if not self.results:
            messagebox.showwarning("Warning", "No results to export. Run calculation first!")
            return

        output_dir = filedialog.askdirectory(title="Select output directory")
        if output_dir:
            messagebox.showinfo("Info", "Export functionality to be implemented")

    def _show_help(self):
        """Show help dialog."""
        help_text = """
Persson Friction Calculator v2.1 - User Guide

1. Input Data Verification Tab:
   - Check that E', E'', tan(δ) and C(q) are correctly loaded
   - Verify master curve and PSD before calculation

2. Calculation Setup Tab:
   - Set pressure, velocity range (log scale: 0.0001~10 m/s)
   - Configure q range and number of points
   - Click "Run G(q,v) Calculation"

3. G(q,v) Results Tab:
   - View multi-velocity G(q) curves
   - Analyze G(q,v) heatmap
   - Check contact area P(q,v)

4. Friction Analysis Tab:
   - Velocity-dependent friction coefficient
   - Contact area ratio analysis
        """
        messagebox.showinfo("User Guide", help_text)

    def _show_about(self):
        """Show about dialog."""
        about_text = """
Persson Friction Calculator v2.1

Work Instruction v2.1 Implementation:
- Velocity range: 0.0001~10 m/s (log scale)
- G(q,v) 2D matrix calculation
- Multi-velocity plotting
- Input data verification

Based on:
Persson, B.N.J. (2001, 2006)
Rubber friction theory
        """
        messagebox.showinfo("About", about_text)


def main():
    """Run the enhanced application."""
    root = tk.Tk()
    app = PerssonModelGUI_V2(root)
    root.mainloop()


if __name__ == "__main__":
    main()
