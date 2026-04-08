"""
Run experiments: baseline vs rule-based metering.
Generates plots and prints metrics table.
"""
import numpy as np
import matplotlib.pyplot as plt
import json
from simulator import BottleneckSim, SimConfig

def run_one(metering: bool, seed: int):
    cfg = SimConfig(seed=seed, metering_enabled=metering)
    sim = BottleneckSim(cfg)
    metrics = sim.run()
    return sim, metrics

def average_over_seeds(metering: bool, seeds=(7, 11, 23, 41, 59)):
    all_m = []
    last_sim = None
    for s in seeds:
        sim, m = run_one(metering, s)
        all_m.append(m)
        last_sim = sim
    keys = all_m[0].keys()
    avg = {k: round(float(np.mean([m[k] for m in all_m])), 3) for k in keys}
    return avg, last_sim

print("Running baseline (no control) over 5 seeds...")
base_avg, base_sim = average_over_seeds(metering=False)
print("Baseline:", base_avg)

print("\nRunning rule-based metering over 5 seeds...")
met_avg, met_sim = average_over_seeds(metering=True)
print("Metered :", met_avg)

# ----- plots -----
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Space-time diagram baseline
axes[0, 0].imshow(base_sim.occupancy_history, aspect="auto", cmap="Greys", interpolation="nearest")
axes[0, 0].axvline(base_sim.cfg.bottleneck_start, color="red", linestyle="--", linewidth=1, label="Bottleneck")
axes[0, 0].set_title("Space-Time: Baseline (no control)")
axes[0, 0].set_xlabel("Cell (position along road)")
axes[0, 0].set_ylabel("Timestep")
axes[0, 0].legend(loc="upper right")

# Space-time diagram metered
axes[0, 1].imshow(met_sim.occupancy_history, aspect="auto", cmap="Greys", interpolation="nearest")
axes[0, 1].axvline(met_sim.cfg.bottleneck_start, color="red", linestyle="--", linewidth=1, label="Bottleneck")
axes[0, 1].axvline(met_sim.cfg.meter_position, color="blue", linestyle=":", linewidth=1, label="Meter")
axes[0, 1].set_title("Space-Time: Rule-Based Metering")
axes[0, 1].set_xlabel("Cell (position along road)")
axes[0, 1].set_ylabel("Timestep")
axes[0, 1].legend(loc="upper right")

# Throughput over time (rolling)
def rolling(x, w=30):
    x = np.array(x, dtype=float)
    if len(x) < w:
        return x
    c = np.cumsum(np.insert(x, 0, 0))
    return (c[w:] - c[:-w]) / w

axes[1, 0].plot(rolling(base_sim.throughput_per_step), label="Baseline", color="black")
axes[1, 0].plot(rolling(met_sim.throughput_per_step), label="Metered", color="blue")
axes[1, 0].set_title("Throughput over Time (30-step rolling avg)")
axes[1, 0].set_xlabel("Timestep")
axes[1, 0].set_ylabel("Vehicles exited per step")
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Bar chart of metrics
metrics_to_plot = ["throughput_per_step", "avg_wait_steps", "avg_max_congestion_len"]
labels = ["Throughput\n(veh/step)", "Avg Wait\n(steps)", "Max Congestion\n(cells)"]
base_vals = [base_avg[m] for m in metrics_to_plot]
met_vals = [met_avg[m] for m in metrics_to_plot]
x = np.arange(len(labels))
w = 0.35
axes[1, 1].bar(x - w/2, base_vals, w, label="Baseline", color="dimgray")
axes[1, 1].bar(x + w/2, met_vals, w, label="Metered", color="steelblue")
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(labels)
axes[1, 1].set_title("Metrics: Baseline vs Metered (5-seed avg)")
axes[1, 1].legend()
for i, (b, m) in enumerate(zip(base_vals, met_vals)):
    axes[1, 1].text(i - w/2, b, f"{b:.2f}", ha="center", va="bottom", fontsize=9)
    axes[1, 1].text(i + w/2, m, f"{m:.2f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig("results.png", dpi=130, bbox_inches="tight")
print("\nSaved results.png")

# Save metrics to json
out = {"baseline": base_avg, "metered": met_avg}
with open("metrics.json", "w") as f:
    json.dump(out, f, indent=2)
print("Saved metrics.json")
print(json.dumps(out, indent=2))
