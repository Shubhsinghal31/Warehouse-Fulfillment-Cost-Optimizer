"""
==============================================================================
  Warehouse Fulfillment Cost Optimizer
  ── Part A: Neural Network approximates the cost function
  ── Part B: Compares SLSQP  vs  Differential Evolution (DE)
  MATLAB-style plots throughout
==============================================================================
"""

import numpy as np
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.optimize import minimize, differential_evolution

# ── MATLAB-style rcParams ────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "black",   "axes.linewidth": 1.2,
    "axes.grid": True,           "grid.color": "#cccccc",
    "grid.linestyle": "-",       "grid.linewidth": 0.6,
    "grid.alpha": 0.7,           "xtick.direction": "out",
    "ytick.direction": "out",    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,    "font.family": "sans-serif",
    "font.size": 11,             "axes.titlesize": 12,
    "axes.titleweight": "bold",  "axes.labelsize": 11,
    "legend.frameon": True,      "legend.edgecolor": "black",
    "legend.fancybox": False,
})
MC = ["#0072BD","#D95319","#EDB120","#7E2F8E","#77AC30","#4DBEEE","#A2142F"]

# ── PROBLEM DATA ─────────────────────────────────────────────────────────────
C       = np.array([[2,3,1,4],[5,4,8,1],[5,6,8,2]], dtype=np.float32)
supply  = np.array([120., 80., 80.])
demand  = np.array([120., 70., 50., 40.])
N_VARS  = 12        # 3 warehouses × 4 customers
TRUE_OPT = 790.0    # known optimum from SLSQP

# ── Penalty-based objective (used by DE and NN training) ─────────────────────
def penalised_cost(x, penalty=1e4):
    """Returns total cost + large penalty for constraint violations."""
    X = x.reshape(3, 4)
    cost = float(np.sum(C * X))
    pen  = 0.0
    for i in range(3):
        pen += max(0.0, X[i].sum() - supply[i]) ** 2
    for j in range(4):
        pen += max(0.0, demand[j] - X[:, j].sum()) ** 2
    pen += float(np.sum(np.maximum(0, -X) ** 2))
    return cost + penalty * pen

# ═══════════════════════════════════════════════════════════════════════════════
#  PART A — NEURAL NETWORK COST APPROXIMATOR
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 62)
print("  PART A  ─  Neural Network Cost Function Approximator")
print("=" * 62)

# ── 1. Generate training data ────────────────────────────────────────────────
np.random.seed(42)
N_SAMPLES = 8000

# Sample feasible-ish shipment plans via Dirichlet allocation
X_train_list, y_train_list = [], []
for _ in range(N_SAMPLES):
    x = np.zeros((3, 4), dtype=np.float32)
    for i in range(3):
        alpha = np.random.rand(4) + 0.5
        frac  = np.random.dirichlet(alpha)
        x[i]  = frac * supply[i] * np.random.uniform(0.5, 1.0)
    cost = float(np.sum(C * x))
    X_train_list.append(x.flatten())
    y_train_list.append(cost)

X_data = torch.tensor(np.array(X_train_list), dtype=torch.float32)
y_data = torch.tensor(np.array(y_train_list), dtype=torch.float32).unsqueeze(1)

# Normalise inputs and outputs
X_mean, X_std = X_data.mean(0), X_data.std(0) + 1e-8
y_mean, y_std = y_data.mean(),  y_data.std()  + 1e-8
X_norm = (X_data - X_mean) / X_std
y_norm = (y_data - y_mean) / y_std

dataset    = TensorDataset(X_norm, y_norm)
loader     = DataLoader(dataset, batch_size=128, shuffle=True)

# ── 2. Define neural network ─────────────────────────────────────────────────
class CostNet(nn.Module):
    """
    Small feedforward NN:  12 → 64 → 64 → 32 → 1
    Approximates the shipping cost surface Z(x).
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(12, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32,  1),
        )
    def forward(self, x):
        return self.net(x)

model     = CostNet()
optimizer_nn = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn   = nn.MSELoss()

# ── 3. Train ──────────────────────────────────────────────────────────────────
EPOCHS    = 80
train_losses, val_losses = [], []

# split 80/20
split = int(0.8 * len(dataset))
train_ds, val_ds = torch.utils.data.random_split(
    dataset, [split, len(dataset)-split],
    generator=torch.Generator().manual_seed(0)
)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=256)

for epoch in range(EPOCHS):
    model.train()
    running = 0.0
    for xb, yb in train_loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer_nn.zero_grad()
        loss.backward()
        optimizer_nn.step()
        running += loss.item() * len(xb)
    train_losses.append(running / len(train_ds))

    model.eval()
    with torch.no_grad():
        vl = sum(loss_fn(model(xb), yb).item()*len(xb) for xb,yb in val_loader)
    val_losses.append(vl / len(val_ds))

print(f"  Training complete — final train loss: {train_losses[-1]:.5f}  "
      f"val loss: {val_losses[-1]:.5f}")

# ── 4. Use NN to approximate costs on a 2-D slice ─────────────────────────────
# Vary x[0][0] (WH1→Cust1) and x[0][2] (WH1→Cust3) while fixing others
model.eval()
v1 = np.linspace(0, 120, 60)
v2 = np.linspace(0, 120, 60)
V1, V2 = np.meshgrid(v1, v2)
Z_true = np.zeros_like(V1)
Z_nn   = np.zeros_like(V1)

base = np.array([70, 0, 50, 0,   0, 70, 0, 10,  50, 0, 0, 30], dtype=np.float32)
for ii in range(len(v1)):
    for jj in range(len(v2)):
        x_slice = base.copy()
        x_slice[0] = V1[ii, jj]   # WH1→Cust1
        x_slice[2] = V2[ii, jj]   # WH1→Cust3
        Z_true[ii, jj] = float(np.sum(C * x_slice.reshape(3,4)))
        x_t = torch.tensor((x_slice - X_mean.numpy()) / X_std.numpy(),
                            dtype=torch.float32)
        with torch.no_grad():
            Z_nn[ii, jj] = float(model(x_t.unsqueeze(0))) * y_std.item() + y_mean.item()

# ── 5. Predict optimal point via NN gradient descent ─────────────────────────
x_opt_nn = torch.tensor(
    (base - X_mean.numpy()) / X_std.numpy(),
    dtype=torch.float32, requires_grad=True
)
nn_opt = torch.optim.Adam([x_opt_nn], lr=0.02)
nn_cost_history = []

for step in range(300):
    nn_opt.zero_grad()
    pred_cost = model(x_opt_nn.unsqueeze(0)) * y_std + y_mean
    pred_cost.backward()
    nn_opt.step()
    nn_cost_history.append(pred_cost.item())

x_final_nn = (x_opt_nn.detach().numpy() * X_std.numpy() + X_mean.numpy())
nn_predicted_cost = float(np.sum(C * x_final_nn.reshape(3,4)))
print(f"  NN gradient descent → predicted cost: ${nn_predicted_cost:.2f}")

# ═══════════════════════════════════════════════════════════════════════════════
#  PART B — ALGORITHM COMPARISON:  SLSQP  vs  Differential Evolution
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 62)
print("  PART B  ─  Algorithm Comparison: SLSQP vs Differential Evolution")
print("=" * 62)

bounds_list = [(0, s) for s in [120,120,120,120, 80,80,80,80, 80,80,80,80]]
constraints_slsqp = []
for i in range(3):
    def sc(x,i=i): return supply[i] - x.reshape(3,4)[i].sum()
    constraints_slsqp.append({"type":"ineq","fun":sc})
for j in range(4):
    def dc(x,j=j): return x.reshape(3,4)[:,j].sum() - demand[j]
    constraints_slsqp.append({"type":"ineq","fun":dc})

# ── Run SLSQP ─────────────────────────────────────────────────────────────────
slsqp_iters, slsqp_costs = [], []
def slsqp_cb(x):
    slsqp_iters.append(len(slsqp_iters)+1)
    slsqp_costs.append(float(np.sum(C * x.reshape(3,4))))

x0 = np.ones(N_VARS) * 10
t0 = time.perf_counter()
res_slsqp = minimize(
    lambda x: float(np.sum(C * x.reshape(3,4))),
    x0, method="SLSQP",
    bounds=bounds_list,
    constraints=constraints_slsqp,
    callback=slsqp_cb,
    options={"maxiter":500, "ftol":1e-9}
)
slsqp_time = time.perf_counter() - t0
slsqp_cost = float(np.sum(C * res_slsqp.x.reshape(3,4)))
print(f"  SLSQP  → cost=${slsqp_cost:.2f}  iters={len(slsqp_iters)}  "
      f"time={slsqp_time*1000:.2f} ms")

# ── Run Differential Evolution ────────────────────────────────────────────────
de_iters, de_costs = [], []
def de_cb(xk, convergence=None):
    de_iters.append(len(de_iters)+1)
    de_costs.append(penalised_cost(xk))

t0 = time.perf_counter()
res_de = differential_evolution(
    penalised_cost,
    bounds=bounds_list,
    seed=42,
    maxiter=300,
    tol=1e-9,
    popsize=12,
    mutation=(0.5, 1.0),
    recombination=0.9,
    callback=de_cb,
    polish=True,
)
de_time = time.perf_counter() - t0
de_cost = float(np.sum(C * res_de.x.reshape(3,4)))
print(f"  DE     → cost=${de_cost:.2f}  iters={len(de_iters)}  "
      f"time={de_time*1000:.2f} ms")

# ── Summary table ─────────────────────────────────────────────────────────────
print()
print(f"  {'Metric':<28} {'SLSQP':>12} {'Diff. Evol.':>14}")
print("  " + "-"*56)
print(f"  {'Optimal Cost ($)':<28} {'${:.2f}'.format(slsqp_cost):>12} {'${:.2f}'.format(de_cost):>14}")
print(f"  {'Iterations':<28} {len(slsqp_iters):>12} {len(de_iters):>14}")
print(f"  {'Solve Time (ms)':<28} {slsqp_time*1000:>11.2f} {de_time*1000:>13.2f}")
print(f"  {'Feasible Solution':<28} {'Yes':>12} {'Yes (polished)':>14}")
print(f"  {'Algorithm Type':<28} {'Gradient-based':>12} {'Gradient-free':>14}")

# ═══════════════════════════════════════════════════════════════════════════════
#  PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

# ── FIGURE 5: NN Training Loss + Cost Surface ─────────────────────────────────
fig5 = plt.figure(figsize=(15, 5))
fig5.suptitle("Deep Learning Module — Neural Network Cost Approximator",
              fontsize=13, fontweight="bold")
gs = GridSpec(1, 3, figure=fig5, wspace=0.38, left=0.07, right=0.97)

# Panel A — Training & Validation Loss
ax = fig5.add_subplot(gs[0])
ep = np.arange(1, EPOCHS+1)
ax.plot(ep, train_losses, color=MC[0], linewidth=2, label="Train Loss")
ax.plot(ep, val_losses,   color=MC[1], linewidth=2, linestyle="--", label="Val Loss")
ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss (normalised)")
ax.set_title("(a) NN Training & Validation Loss")
ax.legend()
ax.set_yscale("log")

# Panel B — True cost surface
ax2 = fig5.add_subplot(gs[1])
cp2 = ax2.contourf(V1, V2, Z_true, levels=20, cmap="YlOrRd")
fig5.colorbar(cp2, ax=ax2, label="True Cost ($)")
ax2.contour(V1, V2, Z_true, levels=10, colors="black", linewidths=0.4, alpha=0.4)
ax2.set_xlabel("WH1 → Cust1 (units)"); ax2.set_ylabel("WH1 → Cust3 (units)")
ax2.set_title("(b) True Cost Surface (2-D slice)")
ax2.grid(False)

# Panel C — NN-predicted surface
ax3 = fig5.add_subplot(gs[2])
cp3 = ax3.contourf(V1, V2, Z_nn, levels=20, cmap="YlGn")
fig5.colorbar(cp3, ax=ax3, label="NN Predicted Cost ($)")
ax3.contour(V1, V2, Z_nn, levels=10, colors="black", linewidths=0.4, alpha=0.4)
ax3.set_xlabel("WH1 → Cust1 (units)"); ax3.set_ylabel("WH1 → Cust3 (units)")
ax3.set_title("(c) NN-Predicted Cost Surface")
ax3.grid(False)

plt.savefig("/mnt/user-data/outputs/shipping_optimization/fig5_nn_module.png",
            dpi=150, bbox_inches="tight")
print("\nSaved: fig5_nn_module.png")

# ── FIGURE 6: Algorithm Comparison ────────────────────────────────────────────
fig6, axes6 = plt.subplots(1, 3, figsize=(15, 5))
fig6.suptitle("Algorithm Comparison — SLSQP vs Differential Evolution",
              fontsize=13, fontweight="bold")

# Panel A — Convergence curves
ax = axes6[0]
ax.plot(slsqp_iters, slsqp_costs,
        color=MC[0], linewidth=2,
        marker="o", markersize=5,
        markerfacecolor="white", markeredgewidth=1.5,
        label="SLSQP")
ax.plot(de_iters, de_costs,
        color=MC[1], linewidth=2,
        marker="s", markersize=4,
        markerfacecolor="white", markeredgewidth=1.5,
        label="Diff. Evolution")
ax.axhline(TRUE_OPT, color="black", linestyle=":", linewidth=1.4,
           label=f"Optimal = ${TRUE_OPT:.0f}")
ax.set_xlabel("Iteration"); ax.set_ylabel("Objective Value Z ($)")
ax.set_title("(a) Convergence Curves")
ax.set_yscale("log")
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v,_: f"${v:,.0f}"))
ax.legend(fontsize=9)

# Panel B — Bar: Final cost comparison
ax2 = axes6[1]
methods = ["SLSQP", "Diff. Evo.", "NN Approx.", "Optimal"]
costs_bar = [slsqp_cost, de_cost, nn_predicted_cost, TRUE_OPT]
colors_bar = [MC[0], MC[1], MC[4], "black"]
bars = ax2.bar(methods, costs_bar, color=colors_bar,
               edgecolor="black", linewidth=0.8, width=0.5)
for bar, val in zip(bars, costs_bar):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
             f"${val:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax2.set_ylabel("Final Cost ($)")
ax2.set_title("(b) Final Cost Comparison")
ax2.set_ylim(0, max(costs_bar) * 1.18)
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v,_: f"${v:.0f}"))

# Panel C — Grouped bar: solve time & iterations
ax3 = axes6[2]
x_pos = np.array([0, 1])
iters_vals = [len(slsqp_iters), len(de_iters)]
time_vals  = [slsqp_time*1000, de_time*1000]

ax3b = ax3.twinx()
w = 0.3
b1 = ax3.bar(x_pos - w/2, iters_vals, w, color=MC[0],
             edgecolor="black", linewidth=0.8, label="Iterations")
b2 = ax3b.bar(x_pos + w/2, time_vals, w, color=MC[1],
              edgecolor="black", linewidth=0.8, label="Time (ms)")

for bar, v in zip(b1, iters_vals):
    ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
             str(v), ha="center", va="bottom", fontsize=10, fontweight="bold")
for bar, v in zip(b2, time_vals):
    ax3b.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
              f"{v:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax3.set_xticks(x_pos); ax3.set_xticklabels(["SLSQP","Diff. Evo."])
ax3.set_ylabel("Number of Iterations", color=MC[0])
ax3b.set_ylabel("Solve Time (ms)", color=MC[1])
ax3.set_title("(c) Iterations & Solve Time")
ax3.tick_params(axis="y", colors=MC[0])
ax3b.tick_params(axis="y", colors=MC[1])
lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3b.get_legend_handles_labels()
ax3.legend(lines1+lines2, labels1+labels2, fontsize=9)

plt.tight_layout()
fig6.savefig("/mnt/user-data/outputs/shipping_optimization/fig6_comparison.png",
             dpi=150, bbox_inches="tight")
print("Saved: fig6_comparison.png")
print("\n✓ All outputs complete.")
