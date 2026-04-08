"""
Bottleneck Traffic Simulator
============================
A Nagel-Schreckenberg-style multi-lane cellular automaton with a lane-merge
bottleneck. A 3-lane road narrows to 1 lane at a configurable position.

Vehicles:
  - have integer position (cell), velocity, lane, and an aggressiveness param
  - update via NS rules: accelerate, brake-for-gap, random-decel, move
  - perform lane changes when blocked, with safety checks
  - in the merge zone, lanes 0 and 1 must merge into lane 2 (the surviving lane)

Outputs per run:
  - throughput (vehicles exited / steps)
  - average waiting time (steps spent at v < v_max)
  - max congestion length (longest contiguous slow region upstream of bottleneck)
  - full space-time occupancy grid for plotting
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


# ---------- Configuration ----------
@dataclass
class SimConfig:
    road_length: int = 200          # cells
    n_lanes: int = 3                # lanes upstream of bottleneck
    bottleneck_start: int = 140     # cell at which lanes 0,1 disappear
    v_max: int = 5                  # cells per timestep
    p_slow: float = 0.20            # random deceleration probability
    arrival_rate: float = 0.75      # prob of spawning a vehicle per lane per step
    n_steps: int = 600
    seed: int = 7
    # control
    metering_enabled: bool = False
    meter_position: int = 30        # where the meter sits
    meter_period: int = 3           # release 1 vehicle every K steps when red


# ---------- Vehicle ----------
@dataclass
class Vehicle:
    vid: int
    pos: int
    lane: int
    v: int = 0
    aggressiveness: float = 0.5
    wait_steps: int = 0
    spawn_step: int = 0


# ---------- Simulator ----------
class BottleneckSim:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        # grid[lane, cell] = vehicle id or -1
        self.grid = -np.ones((cfg.n_lanes, cfg.road_length), dtype=int)
        self.vehicles: dict[int, Vehicle] = {}
        self.next_vid = 0
        self.exited = 0
        self.exit_wait_times: List[int] = []
        # space-time occupancy: store per step which cells are occupied (any lane)
        self.occupancy_history = np.zeros((cfg.n_steps, cfg.road_length), dtype=np.int8)
        self.throughput_per_step: List[int] = []
        self.meter_counter = 0
        self.meter_queue: List[Vehicle] = []  # vehicles waiting at meter

    # ---- helpers ----
    def lane_open(self, lane: int, cell: int) -> bool:
        """Is this (lane, cell) a valid road cell? Lanes 0,1 die at bottleneck_start."""
        if cell < 0 or cell >= self.cfg.road_length:
            return False
        if cell >= self.cfg.bottleneck_start and lane != self.cfg.n_lanes - 1:
            return False
        return True

    def gap_ahead(self, lane: int, cell: int) -> int:
        """Distance to next vehicle ahead in same lane (capped at v_max+1)."""
        for d in range(1, self.cfg.v_max + 2):
            c = cell + d
            if c >= self.cfg.road_length:
                return d  # treat exit as free
            if not self.lane_open(lane, c):
                return d - 1  # lane ends here, must merge before
            if self.grid[lane, c] != -1:
                return d - 1
        return self.cfg.v_max + 1

    def can_change_lane(self, veh: Vehicle, target_lane: int) -> bool:
        if target_lane < 0 or target_lane >= self.cfg.n_lanes:
            return False
        if not self.lane_open(target_lane, veh.pos):
            return False
        if self.grid[target_lane, veh.pos] != -1:
            return False
        # check gap behind in target lane
        for d in range(1, self.cfg.v_max + 1):
            c = veh.pos - d
            if c < 0:
                break
            if self.grid[target_lane, c] != -1:
                # require some gap proportional to (1 - aggressiveness)
                required = max(1, int((1 - veh.aggressiveness) * self.cfg.v_max))
                if d < required:
                    return False
                break
        # check gap ahead in target lane
        gap = self.gap_ahead(target_lane, veh.pos)
        if gap < 2:
            return False
        return True

    # ---- spawning ----
    def spawn(self, step: int):
        cfg = self.cfg
        for lane in range(cfg.n_lanes):
            if self.rng.random() < cfg.arrival_rate:
                # try to place at cell 0
                if self.grid[lane, 0] == -1 and self.lane_open(lane, 0):
                    veh = Vehicle(
                        vid=self.next_vid,
                        pos=0,
                        lane=lane,
                        v=cfg.v_max // 2,
                        aggressiveness=float(self.rng.uniform(0.3, 0.9)),
                        spawn_step=step,
                    )
                    self.grid[lane, 0] = veh.vid
                    self.vehicles[veh.vid] = veh
                    self.next_vid += 1

    # ---- metering ----
    def downstream_density(self) -> float:
        """Density in the merge approach zone (between meter and bottleneck)."""
        cfg = self.cfg
        zone = self.grid[:, cfg.meter_position : cfg.bottleneck_start]
        return float((zone != -1).sum()) / zone.size

    def apply_meter(self, step: int):
        """Rule-based metering: brake vehicles crossing the meter line ONLY when
        the downstream merge zone is congested. This smooths arrivals into the
        merge without strangling throughput."""
        cfg = self.cfg
        if not cfg.metering_enabled:
            return
        density = self.downstream_density()
        # threshold: if merge zone is more than 28% full, throttle
        if density < 0.28:
            return
        # cap velocity of vehicles within meter window (gentle hold-back)
        meter_lo = cfg.meter_position
        meter_hi = cfg.meter_position + 8
        # stronger throttle when more congested
        cap = 1 if density > 0.40 else 2
        for lane in range(cfg.n_lanes):
            for cell in range(meter_lo, meter_hi):
                vid = self.grid[lane, cell]
                if vid != -1:
                    veh = self.vehicles[vid]
                    veh.v = min(veh.v, cap)

    # ---- one timestep ----
    def step(self, step_idx: int):
        cfg = self.cfg

        # 1. spawn
        self.spawn(step_idx)

        # 2. metering
        self.apply_meter(step_idx)

        # 3. lane-change pass: vehicles approaching bottleneck must move to surviving lane
        # iterate from front to back so leaders move first
        order = sorted(self.vehicles.values(), key=lambda v: -v.pos)
        for veh in order:
            if veh.vid not in self.vehicles:
                continue
            # if in a dying lane and within merge approach, try to move toward surviving lane
            distance_to_bn = cfg.bottleneck_start - veh.pos
            if veh.lane != cfg.n_lanes - 1 and 0 < distance_to_bn < 60:
                # urgency rises as distance shrinks
                urgency = 1.0 - (distance_to_bn / 60.0)
                if self.rng.random() < (0.3 + 0.6 * urgency * veh.aggressiveness):
                    # try moving one lane toward surviving lane
                    target = veh.lane + 1 if veh.lane < cfg.n_lanes - 1 else veh.lane
                    if self.can_change_lane(veh, target):
                        self.grid[veh.lane, veh.pos] = -1
                        veh.lane = target
                        self.grid[veh.lane, veh.pos] = veh.vid
            # general "lane is slow, try to switch" behaviour upstream
            elif distance_to_bn >= 60:
                gap = self.gap_ahead(veh.lane, veh.pos)
                if gap < veh.v + 1 and self.rng.random() < 0.2:
                    for tgt in [veh.lane - 1, veh.lane + 1]:
                        if 0 <= tgt < cfg.n_lanes and self.can_change_lane(veh, tgt):
                            self.grid[veh.lane, veh.pos] = -1
                            veh.lane = tgt
                            self.grid[veh.lane, veh.pos] = veh.vid
                            break

        # 4. NS update: accelerate, brake, randomize, move
        exited_this_step = 0
        order = sorted(self.vehicles.values(), key=lambda v: -v.pos)
        for veh in order:
            if veh.vid not in self.vehicles:
                continue
            # if vehicle is in a lane that has died at this cell, force-stop and try merge
            if not self.lane_open(veh.lane, veh.pos):
                # this shouldn't happen but guard
                continue

            # accelerate
            veh.v = min(veh.v + 1, cfg.v_max)
            # brake for gap
            gap = self.gap_ahead(veh.lane, veh.pos)
            veh.v = min(veh.v, gap)
            # extra brake if approaching dying-lane end
            if veh.lane != cfg.n_lanes - 1:
                cells_to_lane_end = cfg.bottleneck_start - veh.pos - 1
                if cells_to_lane_end >= 0:
                    veh.v = min(veh.v, cells_to_lane_end)
            # random slowdown
            if veh.v > 0 and self.rng.random() < cfg.p_slow:
                veh.v -= 1
            # waiting tracker
            if veh.v < cfg.v_max:
                veh.wait_steps += 1
            # move
            new_pos = veh.pos + veh.v
            self.grid[veh.lane, veh.pos] = -1
            if new_pos >= cfg.road_length:
                # exited
                self.exited += 1
                exited_this_step += 1
                self.exit_wait_times.append(veh.wait_steps)
                del self.vehicles[veh.vid]
            else:
                veh.pos = new_pos
                self.grid[veh.lane, veh.pos] = veh.vid

        self.throughput_per_step.append(exited_this_step)

        # 5. record occupancy
        occ = (self.grid != -1).any(axis=0).astype(np.int8)
        self.occupancy_history[step_idx] = occ

    # ---- run ----
    def run(self):
        for s in range(self.cfg.n_steps):
            self.step(s)
        return self.metrics()

    # ---- metrics ----
    def metrics(self) -> dict:
        cfg = self.cfg
        total_throughput = self.exited
        avg_throughput = total_throughput / cfg.n_steps
        avg_wait = float(np.mean(self.exit_wait_times)) if self.exit_wait_times else 0.0
        # congestion length: longest run of occupied cells upstream of bottleneck (averaged across last 100 steps)
        window = self.occupancy_history[-100:]
        max_runs = []
        for row in window:
            run_len = 0
            best = 0
            for c in row[: cfg.bottleneck_start]:
                if c:
                    run_len += 1
                    best = max(best, run_len)
                else:
                    run_len = 0
            max_runs.append(best)
        avg_max_congestion = float(np.mean(max_runs))
        return {
            "total_exited": int(total_throughput),
            "throughput_per_step": round(avg_throughput, 4),
            "avg_wait_steps": round(avg_wait, 2),
            "avg_max_congestion_len": round(avg_max_congestion, 2),
        }
