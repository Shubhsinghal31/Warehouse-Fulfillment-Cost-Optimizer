"""
==============================================================================
  Warehouse Fulfillment Cost Optimizer
  Method : Sequential Least Squares Programming (SLSQP) via scipy.optimize
==============================================================================
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── 1. PROBLEM DATA ──────────────────────────────────────────────────────────
#  3 warehouses (supply nodes)  →  4 customers (demand nodes)
#  Unit shipping cost matrix  C[i][j]  ($/unit)
C = np.array([
    [2.0, 3.0, 1.0, 4.0],   # Warehouse 1
    [5.0, 4.0, 8.0, 1.0],   # Warehouse 2
    [5.0, 6.0, 8.0, 2.0],   # Warehouse 3
])

supply    = np.array([120.0, 80.0, 80.0])   # units available at each warehouse
demand    = np.array([150.0, 70.0, 60.0])   # units required by each customer - fixed to balance
# NOTE: demand is adjusted below to balance the problem (sum supply = sum demand = 280)
demand    = np.array([120.0, 70.0, 50.0, 40.0])   # total = 280

n_src, n_dst = C.shape          # 3 warehouses, 4 customers
n_vars = n_src * n_dst          # 12 decision variables x[i,j]

# ── 2. OBJECTIVE FUNCTION ─────────────────────────────────────────────────────
iteration_costs = []   # record cost at each callback

def total_cost(x):
    """Scalar objective: sum of (cost * quantity) over all routes."""
    X = x.reshape(n_src, n_dst)
    return np.sum(C * X)

def callback(x):
    iteration_costs.append(total_cost(x))

# ── 3. CONSTRAINTS ────────────────────────────────────────────────────────────
constraints = []

# Supply constraints: sum over destinations ≤ supply[i]
for i in range(n_src):
    def supply_con(x, i=i):
        X = x.reshape(n_src, n_dst)
        return supply[i] - X[i, :].sum()
    constraints.append({"type": "ineq", "fun": supply_con})

# Demand constraints: sum over sources ≥ demand[j]
for j in range(n_dst):
    def demand_con(x, j=j):
        X = x.reshape(n_src, n_dst)
        return X[:, j].sum() - demand[j]
    constraints.append({"type": "ineq", "fun": demand_con})

# Non-negativity bounds
bounds = [(0, None)] * n_vars

# ── 4. SOLVE ──────────────────────────────────────────────────────────────────
x0 = np.ones(n_vars) * 10      # initial guess

result = minimize(
    total_cost,
    x0,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    callback=callback,
    options={"maxiter": 500, "ftol": 1e-9, "disp": True},
)

X_opt = result.x.reshape(n_src, n_dst)
optimal_cost = result.fun

print("\n" + "="*60)
print("  OPTIMAL SHIPPING PLAN")
print("="*60)
headers = ["Cust 1", "Cust 2", "Cust 3", "Cust 4", "Supply Used"]
print(f"{'':12}", end="")
for h in headers:
    print(f"{h:>10}", end="")
print()
for i in range(n_src):
    print(f"Warehouse {i+1}  ", end="")
    row_sum = 0
    for j in range(n_dst):
        v = X_opt[i, j]
        row_sum += v
        print(f"{v:>10.2f}", end="")
    print(f"{row_sum:>10.2f}")
print(f"\n  Total Minimum Shipping Cost: ${optimal_cost:,.2f}")
print(f"  Solver Status : {result.message}")
print("="*60)

# ── 5. VISUALISATIONS ─────────────────────────────────────────────────────────
DARK   = "#0d1117"
CARD   = "#161b22"
ACCENT = "#00d9a3"
GOLD   = "#f0c040"
RED    = "#ff6b6b"
BLUE   = "#4da6ff"
PURPLE = "#b57bee"
WHITE  = "#e6edf3"
GRAY   = "#8b949e"

fig = plt.figure(figsize=(18, 13), facecolor=DARK)
gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38,
               left=0.07, right=0.97, top=0.90, bottom=0.07)

# ── Subplot 1: Heatmap of optimal shipments ────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor(CARD)
im = ax1.imshow(X_opt, cmap="YlGn", aspect="auto")
for i in range(n_src):
    for j in range(n_dst):
        ax1.text(j, i, f"{X_opt[i,j]:.1f}", ha="center", va="center",
                 color="black" if X_opt[i,j] > 20 else WHITE, fontsize=11, fontweight="bold")
ax1.set_xticks(range(n_dst))
ax1.set_xticklabels([f"Cust {j+1}" for j in range(n_dst)], color=WHITE, fontsize=10)
ax1.set_yticks(range(n_src))
ax1.set_yticklabels([f"WH {i+1}" for i in range(n_src)], color=WHITE, fontsize=10)
ax1.set_title("Optimal Shipment Quantities\n(units per route)", color=ACCENT,
              fontsize=12, fontweight="bold", pad=10)
cbar = fig.colorbar(im, ax=ax1, fraction=0.046)
cbar.ax.yaxis.set_tick_params(color=WHITE)
plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=WHITE)

# ── Subplot 2: Cost heatmap ────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor(CARD)
route_cost = C * X_opt
im2 = ax2.imshow(route_cost, cmap="OrRd", aspect="auto")
for i in range(n_src):
    for j in range(n_dst):
        ax2.text(j, i, f"${route_cost[i,j]:.0f}", ha="center", va="center",
                 color="black" if route_cost[i,j] > 80 else WHITE, fontsize=11, fontweight="bold")
ax2.set_xticks(range(n_dst))
ax2.set_xticklabels([f"Cust {j+1}" for j in range(n_dst)], color=WHITE, fontsize=10)
ax2.set_yticks(range(n_src))
ax2.set_yticklabels([f"WH {i+1}" for i in range(n_src)], color=WHITE, fontsize=10)
ax2.set_title("Route Cost Breakdown ($)\n(cost × quantity)", color=GOLD,
              fontsize=12, fontweight="bold", pad=10)
cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046)
cbar2.ax.yaxis.set_tick_params(color=WHITE)
plt.setp(plt.getp(cbar2.ax.axes, "yticklabels"), color=WHITE)

# ── Subplot 3: Convergence curve ───────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_facecolor(CARD)
if len(iteration_costs) > 1:
    iters = range(1, len(iteration_costs) + 1)
    ax3.plot(iters, iteration_costs, color=ACCENT, linewidth=2.5, zorder=3)
    ax3.fill_between(iters, iteration_costs, min(iteration_costs),
                     alpha=0.18, color=ACCENT)
    ax3.axhline(optimal_cost, color=GOLD, linestyle="--", linewidth=1.5,
                label=f"Optimal: ${optimal_cost:,.0f}")
    ax3.scatter([len(iteration_costs)], [optimal_cost],
                color=GOLD, s=80, zorder=5)
    ax3.legend(facecolor=CARD, edgecolor=GRAY, labelcolor=WHITE, fontsize=9)
else:
    ax3.text(0.5, 0.5, "Converged in 1 step", transform=ax3.transAxes,
             ha="center", va="center", color=ACCENT, fontsize=13)
ax3.set_xlabel("Iteration", color=GRAY, fontsize=10)
ax3.set_ylabel("Total Cost ($)", color=GRAY, fontsize=10)
ax3.set_title("Convergence of SLSQP Solver", color=PURPLE,
              fontsize=12, fontweight="bold", pad=10)
ax3.tick_params(colors=GRAY)
for spine in ax3.spines.values():
    spine.set_edgecolor(GRAY)

# ── Subplot 4: Bar – supply utilisation ───────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
ax4.set_facecolor(CARD)
used = X_opt.sum(axis=1)
x_pos = np.arange(n_src)
bars_sup = ax4.bar(x_pos - 0.2, supply, 0.35, label="Capacity", color=BLUE, alpha=0.7)
bars_use = ax4.bar(x_pos + 0.2, used,   0.35, label="Used",     color=ACCENT, alpha=0.9)
ax4.set_xticks(x_pos)
ax4.set_xticklabels([f"WH {i+1}" for i in range(n_src)], color=WHITE)
ax4.set_ylabel("Units", color=GRAY)
ax4.set_title("Supply Capacity vs Utilisation", color=BLUE,
              fontsize=12, fontweight="bold", pad=10)
ax4.legend(facecolor=CARD, edgecolor=GRAY, labelcolor=WHITE)
ax4.tick_params(colors=GRAY)
for spine in ax4.spines.values():
    spine.set_edgecolor(GRAY)
for bar in bars_use:
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f"{bar.get_height():.0f}", ha="center", va="bottom",
             color=ACCENT, fontsize=9, fontweight="bold")

# ── Subplot 5: Bar – demand satisfaction ──────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
ax5.set_facecolor(CARD)
received = X_opt.sum(axis=0)
x_pos2 = np.arange(n_dst)
ax5.bar(x_pos2 - 0.2, demand,   0.35, label="Demand",   color=RED,  alpha=0.7)
ax5.bar(x_pos2 + 0.2, received, 0.35, label="Received", color=GOLD, alpha=0.9)
ax5.set_xticks(x_pos2)
ax5.set_xticklabels([f"Cust {j+1}" for j in range(n_dst)], color=WHITE)
ax5.set_ylabel("Units", color=GRAY)
ax5.set_title("Customer Demand vs Fulfilment", color=RED,
              fontsize=12, fontweight="bold", pad=10)
ax5.legend(facecolor=CARD, edgecolor=GRAY, labelcolor=WHITE)
ax5.tick_params(colors=GRAY)
for spine in ax5.spines.values():
    spine.set_edgecolor(GRAY)

# ── Subplot 6: Pie – cost distribution by warehouse ───────────────────────
ax6 = fig.add_subplot(gs[1, 2])
ax6.set_facecolor(CARD)
wh_costs = route_cost.sum(axis=1)
colors_pie = [BLUE, ACCENT, PURPLE]
wedges, texts, autotexts = ax6.pie(
    wh_costs, labels=[f"WH {i+1}\n${wh_costs[i]:.0f}" for i in range(n_src)],
    autopct="%1.1f%%", colors=colors_pie,
    startangle=140, pctdistance=0.75,
    wedgeprops={"edgecolor": DARK, "linewidth": 2}
)
for t in texts:      t.set_color(WHITE); t.set_fontsize(10)
for t in autotexts:  t.set_color(DARK);  t.set_fontsize(9); t.set_fontweight("bold")
ax6.set_title("Cost Share by Warehouse", color=PURPLE,
              fontsize=12, fontweight="bold", pad=10)

# ── Main title ────────────────────────────────────────────────────────────
fig.suptitle(
    f"Shipping Cost Minimization  ·  Optimal Total Cost: ${optimal_cost:,.2f}",
    color=WHITE, fontsize=16, fontweight="bold", y=0.96
)

out_path = "/mnt/user-data/outputs/shipping_optimization/results_dashboard.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK)
print(f"\nDashboard saved → {out_path}")
