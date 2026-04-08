"""
Generate an animated GIF showing baseline vs metered side-by-side.
Uses matplotlib animation saved as GIF via PillowWriter.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from simulator import BottleneckSim, SimConfig

# Run both sims, recording per-step full grid snapshots
def run_with_snapshots(metering: bool, seed: int = 7, n_steps: int = 400):
    cfg = SimConfig(seed=seed, metering_enabled=metering, n_steps=n_steps)
    sim = BottleneckSim(cfg)
    snapshots = []
    throughput_cum = []
    cum = 0
    for s in range(n_steps):
        sim.step(s)
        # snapshot: copy of grid (3 lanes x road_length), mark occupied
        snap = (sim.grid != -1).astype(np.int8)
        snapshots.append(snap)
        cum += sim.throughput_per_step[-1]
        throughput_cum.append(cum)
    return cfg, snapshots, throughput_cum

print("Running baseline for animation...")
cfg_b, snaps_b, cum_b = run_with_snapshots(metering=False)
print("Running metered for animation...")
cfg_m, snaps_m, cum_m = run_with_snapshots(metering=True)

n_frames = len(snaps_b)
# subsample for GIF size (every 3rd frame)
stride = 3
frame_indices = list(range(0, n_frames, stride))

fig, axes = plt.subplots(3, 1, figsize=(11, 6))
fig.patch.set_facecolor("white")

# Set up image plots for both sims
im_b = axes[0].imshow(snaps_b[0], aspect="auto", cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
axes[0].axvline(cfg_b.bottleneck_start, color="red", linestyle="--", linewidth=1.2)
axes[0].set_title("Baseline (no control)", fontsize=11)
axes[0].set_ylabel("Lane")
axes[0].set_yticks([0, 1, 2])

im_m = axes[1].imshow(snaps_m[0], aspect="auto", cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
axes[1].axvline(cfg_m.bottleneck_start, color="red", linestyle="--", linewidth=1.2, label="Bottleneck")
axes[1].axvline(cfg_m.meter_position, color="blue", linestyle=":", linewidth=1.2, label="Meter")
axes[1].set_title("Rule-Based Metering", fontsize=11)
axes[1].set_xlabel("Position along road (cells)")
axes[1].set_ylabel("Lane")
axes[1].set_yticks([0, 1, 2])
axes[1].legend(loc="upper left", fontsize=8)

# Cumulative throughput plot
line_b, = axes[2].plot([], [], color="black", label="Baseline", linewidth=1.8)
line_m, = axes[2].plot([], [], color="steelblue", label="Metered", linewidth=1.8)
axes[2].set_xlim(0, n_frames)
axes[2].set_ylim(0, max(max(cum_b), max(cum_m)) * 1.05)
axes[2].set_xlabel("Timestep")
axes[2].set_ylabel("Cumulative exited")
axes[2].set_title("Cumulative Throughput", fontsize=11)
axes[2].legend(loc="upper left", fontsize=9)
axes[2].grid(alpha=0.3)

step_text = fig.text(0.5, 0.97, "", ha="center", fontsize=10, fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.96])

def update(i):
    idx = frame_indices[i]
    im_b.set_data(snaps_b[idx])
    im_m.set_data(snaps_m[idx])
    line_b.set_data(range(idx + 1), cum_b[: idx + 1])
    line_m.set_data(range(idx + 1), cum_m[: idx + 1])
    step_text.set_text(f"Step {idx} / {n_frames}   |   Baseline exited: {cum_b[idx]}   Metered exited: {cum_m[idx]}")
    return im_b, im_m, line_b, line_m, step_text

print(f"Rendering {len(frame_indices)} frames...")
anim = animation.FuncAnimation(fig, update, frames=len(frame_indices), interval=60, blit=False)
writer = animation.PillowWriter(fps=15)
anim.save("simulation.gif", writer=writer, dpi=90)
print("Saved simulation.gif")
plt.close()
