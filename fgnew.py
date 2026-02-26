import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import re

START_STRAIN_DEFAULT = 0.0148
NUM_FINAL_POINTS_DEFAULT = 20
MAX_DATASETS = 4


def parse_3col_text(text: str):
    """
    Parse pasted text containing 3 columns: strain, f, g
    Separators: tab, comma, spaces.
    Ignores empty lines and header-like lines containing letters.
    Returns (strain, f, g) as numpy arrays sorted by strain.
    Duplicate strains are averaged.
    """
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return None

    rows = []
    for ln in lines:
        # Skip header-ish lines
        if re.search(r"[A-Za-z가-힣]", ln):
            continue

        parts = re.split(r"[,\s]+", ln.strip())
        if len(parts) < 3:
            continue

        try:
            x = float(parts[0])
            f = float(parts[1])
            g = float(parts[2])
        except ValueError:
            continue

        if x <= 0:
            continue
        rows.append((x, f, g))

    if len(rows) < 3:
        return None

    arr = np.array(rows, dtype=float)
    x = arr[:, 0]
    f = arr[:, 1]
    g = arr[:, 2]

    # sort
    order = np.argsort(x)
    x, f, g = x[order], f[order], g[order]

    # average duplicates
    ux = np.unique(x)
    if len(ux) != len(x):
        f_out, g_out = [], []
        for val in ux:
            m = (x == val)
            f_out.append(np.mean(f[m]))
            g_out.append(np.mean(g[m]))
        x, f, g = ux, np.array(f_out), np.array(g_out)

    if np.any(np.diff(x) <= 0):
        return None

    return x, f, g


def natural_cubic_spline_prepare(x: np.ndarray, y: np.ndarray):
    """
    Natural cubic spline: second derivatives at ends are zero.
    Returns m (second derivatives at knots).
    """
    n = len(x)
    if n < 3:
        raise ValueError("Need >= 3 points for spline.")
    h = np.diff(x)
    if np.any(h <= 0):
        raise ValueError("x must be strictly increasing.")

    # Build tri-diagonal system for m[1:n-1]
    A = np.zeros((n - 2, n - 2), dtype=float)
    rhs = np.zeros(n - 2, dtype=float)

    for i in range(1, n - 1):
        row = i - 1
        A[row, row] = 2.0 * (h[i - 1] + h[i])
        if row - 1 >= 0:
            A[row, row - 1] = h[i - 1]
        if row + 1 < n - 2:
            A[row, row + 1] = h[i]
        rhs[row] = 6.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

    m = np.zeros(n, dtype=float)
    if n - 2 > 0:
        m_inner = np.linalg.solve(A, rhs)
        m[1:n - 1] = m_inner
    return m


def natural_cubic_spline_eval(x: np.ndarray, y: np.ndarray, m: np.ndarray, xq: np.ndarray):
    """
    Evaluate spline at xq.
    Outside [x0, xn] -> NaN (no extrapolation).
    """
    xq = np.asarray(xq, dtype=float)
    yq = np.full_like(xq, np.nan, dtype=float)

    for k, xx in enumerate(xq):
        if xx < x[0] or xx > x[-1]:
            continue
        i = np.searchsorted(x, xx) - 1
        if i < 0:
            i = 0
        if i >= len(x) - 1:
            i = len(x) - 2

        h = x[i + 1] - x[i]
        if h <= 0:
            continue

        t = (x[i + 1] - xx) / h
        u = (xx - x[i]) / h

        yq[k] = (
            t * y[i] + u * y[i + 1]
            + (((t ** 3) - t) * m[i] + ((u ** 3) - u) * m[i + 1]) * (h ** 2) / 6.0
        )
    return yq


def spline_interp(x: np.ndarray, y: np.ndarray, xq: np.ndarray):
    m = natural_cubic_spline_prepare(x, y)
    return natural_cubic_spline_eval(x, y, m, xq)


def logspace_points(x_start: float, x_end: float, n: int):
    return 10 ** np.linspace(np.log10(x_start), np.log10(x_end), n)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Spline Interpolation + Mean (Strain, f, g) | Multi-dataset")
        self.geometry("1250x850")

        self.start_var = tk.StringVar(value=str(START_STRAIN_DEFAULT))
        self.npts_var = tk.StringVar(value=str(NUM_FINAL_POINTS_DEFAULT))

        self.include_vars = [tk.BooleanVar(value=True) for _ in range(MAX_DATASETS)]
        self.text_boxes = []

        self._build_ui()

    def _build_ui(self):
        top = ttk.Frame(self, padding=10)
        top.pack(fill="x")

        ttk.Label(top, text="Start strain:").pack(side="left")
        ttk.Entry(top, textvariable=self.start_var, width=12).pack(side="left", padx=6)

        ttk.Label(top, text="Final log-spaced points:").pack(side="left", padx=(20, 0))
        ttk.Entry(top, textvariable=self.npts_var, width=6).pack(side="left", padx=6)

        ttk.Button(top, text="Process", command=self.process).pack(side="left", padx=10)
        ttk.Button(top, text="Clear All", command=self.clear_all).pack(side="left")
        ttk.Button(top, text="Load Example", command=self.load_example).pack(side="left", padx=10)

        mid = ttk.Frame(self, padding=10)
        mid.pack(fill="both", expand=True)

        inputs = ttk.LabelFrame(mid, text="Inputs (paste 3 columns: strain, f, g). Choose which datasets to include.", padding=10)
        inputs.pack(side="top", fill="both", expand=True)

        grid = ttk.Frame(inputs)
        grid.pack(fill="both", expand=True)

        for i in range(MAX_DATASETS):
            lf = ttk.LabelFrame(grid, text=f"Dataset {i+1}", padding=6)
            lf.grid(row=i//2, column=i % 2, padx=6, pady=6, sticky="nsew")
            grid.grid_columnconfigure(i % 2, weight=1)
            grid.grid_rowconfigure(i//2, weight=1)

            chk = ttk.Checkbutton(lf, text="Include in mean", variable=self.include_vars[i])
            chk.pack(anchor="w", pady=(0, 4))

            txt = tk.Text(lf, height=12, wrap="none")
            txt.pack(fill="both", expand=True)
            self.text_boxes.append(txt)

        out = ttk.LabelFrame(mid, text="Output (copy/paste).", padding=10)
        out.pack(side="bottom", fill="both", expand=True)

        self.out_text = tk.Text(out, height=18, wrap="none")
        self.out_text.pack(fill="both", expand=True)

    def clear_all(self):
        for t in self.text_boxes:
            t.delete("1.0", "end")
        self.out_text.delete("1.0", "end")
        for v in self.include_vars:
            v.set(True)

    def load_example(self):
        # Example with dummy g column
        example = """strain\tf\tg
0.010017900\t7.19695\t1.000
0.013\t7.19194\t0.990
0.018\t7.2112\t0.980
0.024\t7.21234\t0.970
0.032\t7.19743\t0.960
0.042\t7.17105\t0.950
0.056\t7.12882\t0.940
0.075\t7.08183\t0.930
0.100\t7.02169\t0.920
0.133\t6.93847\t0.910
0.178\t6.84153\t0.900
0.237\t6.72807\t0.890
0.316\t6.60048\t0.880
0.422\t6.45959\t0.870
0.562\t6.30351\t0.860
0.750\t6.12953\t0.850
1.000\t5.93674\t0.840
1.333\t5.72142\t0.830
1.778\t5.47839\t0.820
2.371\t5.203\t0.810
3.162\t4.88701\t0.800
4.217\t4.52454\t0.790
5.624\t4.11889\t0.780
7.501\t3.68436\t0.770
10.004\t3.24201\t0.760
13.341\t2.81759\t0.750
17.788\t2.44624\t0.740
23.727\t2.16471\t0.730
27.407\t2.07242\t0.720
"""
        self.text_boxes[0].delete("1.0", "end")
        self.text_boxes[0].insert("1.0", example)
        for i in range(1, MAX_DATASETS):
            self.text_boxes[i].delete("1.0", "end")
        for v in self.include_vars:
            v.set(True)

    def process(self):
        # Params
        try:
            start_strain = float(self.start_var.get().strip())
            if start_strain <= 0:
                raise ValueError
        except Exception:
            messagebox.showerror("Input error", "Start strain must be a positive number.")
            return

        try:
            n_final = int(self.npts_var.get().strip())
            if n_final < 2:
                raise ValueError
        except Exception:
            messagebox.showerror("Input error", "Final points must be an integer >= 2.")
            return

        # Parse datasets
        datasets = []
        included = []

        for i, txt in enumerate(self.text_boxes):
            raw = txt.get("1.0", "end").strip()
            if not raw:
                datasets.append(None)
                included.append(False)
                continue

            parsed = parse_3col_text(raw)
            if parsed is None:
                messagebox.showerror("Parse error", f"Dataset {i+1}: Need >=3 numeric rows of 3 columns (strain,f,g).")
                return

            datasets.append(parsed)
            included.append(bool(self.include_vars[i].get()))

        # Determine which datasets are active (have data + checked)
        active = [(i, ds) for i, ds in enumerate(datasets) if (ds is not None and included[i])]
        if not active:
            messagebox.showwarning("No active data", "No dataset is both provided AND checked (Include in mean).")
            return

        # Global max strain among ACTIVE datasets (per your rule)
        max_strain = max(ds[0].max() for _, ds in active)
        if max_strain <= start_strain:
            messagebox.showerror("Range error", f"Global max strain ({max_strain:g}) must be > start strain ({start_strain:g}).")
            return

        # Common grid: union of strains from ACTIVE datasets within [start, max] + include start_strain
        all_x = np.concatenate([ds[0] for _, ds in active])
        common_x = np.unique(all_x[(all_x >= start_strain) & (all_x <= max_strain)])
        common_x = np.unique(np.concatenate([common_x, np.array([start_strain], dtype=float)]))
        common_x.sort()

        if len(common_x) < 3:
            messagebox.showerror("Grid error", "Not enough points in [start, max] to build mean spline (need >=3).")
            return

        # Interpolate each ACTIVE dataset onto common_x (NO trimming! keep full x so 0.0148 can be interpolated)
        F_rows = []
        G_rows = []
        active_names = []

        for idx, (x, f, g) in active:
            if len(x) < 3:
                # should not happen because parse requires >=3
                F_rows.append(np.full_like(common_x, np.nan))
                G_rows.append(np.full_like(common_x, np.nan))
                continue
            try:
                f_i = spline_interp(x, f, common_x)
                g_i = spline_interp(x, g, common_x)
            except Exception:
                f_i = np.full_like(common_x, np.nan)
                g_i = np.full_like(common_x, np.nan)

            F_rows.append(f_i)
            G_rows.append(g_i)
            active_names.append(f"Dataset{idx+1}")

        F_mat = np.vstack(F_rows)  # (n_active, n_common)
        G_mat = np.vstack(G_rows)

        # Mean ignoring NaNs
        mean_f = np.nanmean(F_mat, axis=0)
        mean_g = np.nanmean(G_mat, axis=0)

        # Final log-spaced points
        final_x = logspace_points(start_strain, max_strain, n_final)

        # Interp mean curve onto final_x:
        # We spline using only valid mean points (>=3), otherwise linear fallback, otherwise all NaN
        def interp_mean(cx, my, xq):
            valid = np.isfinite(my)
            cx2 = cx[valid]
            my2 = my[valid]
            if len(cx2) >= 3:
                return spline_interp(cx2, my2, xq)
            elif len(cx2) >= 2:
                return np.interp(xq, cx2, my2, left=np.nan, right=np.nan)
            else:
                return np.full_like(xq, np.nan, dtype=float)

        final_f = interp_mean(common_x, mean_f, final_x)
        final_g = interp_mean(common_x, mean_g, final_x)

        # Output
        self.out_text.delete("1.0", "end")
        self.out_text.insert("end", f"Active datasets included in mean: {', '.join(active_names)}\n")
        self.out_text.insert("end", f"Start strain = {start_strain:g}, Global max strain (active) = {max_strain:g}\n\n")

        self.out_text.insert("end", "=== Common strain grid (union) + mean(f), mean(g) ===\n")
        self.out_text.insert("end", "strain\tmean_f\tmean_g\n")
        for x, mf, mg in zip(common_x, mean_f, mean_g):
            self.out_text.insert("end", f"{x:.10g}\t{mf:.10g}\t{mg:.10g}\n")

        self.out_text.insert("end", "\n=== Final log-spaced points (N={}) + mean(f), mean(g) ===\n".format(n_final))
        self.out_text.insert("end", "strain_logN\tmean_f_logN\tmean_g_logN\n")
        for x, mf, mg in zip(final_x, final_f, final_g):
            self.out_text.insert("end", f"{x:.10g}\t{mf:.10g}\t{mg:.10g}\n")


if __name__ == "__main__":
    app = App()
    app.mainloop()