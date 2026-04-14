#!/usr/bin/env python3
"""
tetlv_simulation_v2.py  -  SRv6 TE-TLV v2  AUTHENTIC simulation
-----------------------------------------------------------------
Authenticity principles:
  1. IOAM ECMP measured on REAL 50/50 routed packets - same method as TE-TLV
  2. Violations: both systems checked on ACTUAL latency vs TC_BUDGET_US
  3. Thompson Sampling (LRT): returns CONTINUOUS probability tu/(tu+tl)
  4. LCon: PROPORTIONAL to queue ratio - not a binary step function
  5. QoS model: TC5/TC6 (EF DSCP 46/48) hardware-protected under congestion;
     TC0/TC1 (BE) absorb excess first - authentic DSCP behaviour
  6. No display-packet weighting bias toward any TC
  7. TE-TLV advantage emerges naturally from better path knowledge, not tuning

Requirements: pip install matplotlib numpy
Run        : python3 tetlv_simulation_v2.py
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation
from collections import deque
import math, random, time

# ==================================================================
#  CONSTANTS
# ==================================================================
TC_DSCP      = {"TC0":0,"TC1":8,"TC2":16,"TC3":24,"TC4":32,"TC5":46,"TC6":48}
TCS          = list(TC_DSCP.keys())
TC_C         = ["#4caf50","#8bc34a","#ffc107","#ff9800","#f44336","#e91e63","#9c27b0"]
TC_BUDGET_US = {"TC0":500,"TC1":400,"TC2":300,"TC3":200,"TC4":120,"TC5":60,"TC6":80}
TC_ALGO      = {"TC0":"DWRR","TC1":"DWRR","TC2":"LConn","TC3":"RB",
                "TC4":"LCon","TC5":"LRT","TC6":"LRT"}

# Jumbo frame parameters (IEEE 802.3, 9000-byte payload)
JUMBO_FRAME_B  = 9000
SRV6_OH_B      = 40+56+40      # IPv6(40) + SRH(56) + TE-TLV(40)
FRAME_TOTAL_B  = JUMBO_FRAME_B + SRV6_OH_B   # 9136 bytes on wire
LINK_RATE_GBPS = 100
SER_DELAY_US   = FRAME_TOTAL_B * 8 / (LINK_RATE_GBPS * 1e9) * 1e6  # ~0.731 us

DWRR_QUANTUM = {"TC0": JUMBO_FRAME_B, "TC1": JUMBO_FRAME_B * 2}

QOS_PROT = {"TC0":1.40,"TC1":1.20,"TC2":1.00,"TC3":0.85,
            "TC4":0.65,"TC5":0.38,"TC6":0.42}

NODE_SID = {n: "fd00::" + n.replace("Server-","S").replace("P","")
            for n in ["Server-A","P1","P2","P3","P4","P5","P6","P7","P8","Server-B"]}
PATH_UPPER = ["Server-A","P1","P2","P3","P4","P5","Server-B"]
PATH_LOWER = ["Server-A","P1","P6","P7","P8","P5","Server-B"]
ALL_NODES  = list(dict.fromkeys(PATH_UPPER + PATH_LOWER))
CTRL_NODES = ["P2","P3","P4","P6","P7","P8"]
UPPER_CTRL = ["P2","P3","P4"]
LOWER_CTRL = ["P6","P7","P8"]
NODE_POS   = {
    "Server-A":(0,.5), "P1":(1,.5),
    "P2":(2,.82), "P3":(3,.82), "P4":(4,.82),
    "P6":(2,.18), "P7":(3,.18), "P8":(4,.18),
    "P5":(5,.5),  "Server-B":(6,.5)
}
EDGES_U    = list(zip(PATH_UPPER, PATH_UPPER[1:]))
EDGES_L    = list(zip(PATH_LOWER, PATH_LOWER[1:]))
PKT_BASE_NS= int(time.time() * 1e9)
N_PARTS    = 7

# ==================================================================
#  DATA CLASSES
# ==================================================================
class TETLVRecord:
    __slots__ = ("node","dscp","tc_name","queue_fill","ptp_in_ns",
                 "ptp_eg_ns","residence_us","drops","c_flag","budget_pct")
    def __init__(self, node, tc_name, qfill, ptp_in, ptp_eg, drops):
        self.node        = node
        self.dscp        = TC_DSCP[tc_name]
        self.tc_name     = tc_name
        self.queue_fill  = dict(qfill)
        self.ptp_in_ns   = ptp_in
        self.ptp_eg_ns   = ptp_eg
        self.residence_us= (ptp_eg - ptp_in) / 1000.0
        self.drops       = drops
        self.c_flag      = qfill.get("TC5",0) > .80 or qfill.get("TC6",0) > .80
        self.budget_pct  = self.residence_us / TC_BUDGET_US.get(tc_name, 200) * 100


class SRv6Packet:
    def __init__(self, pid, tc_name, seg_list):
        self.pid      = pid
        self.tc_name  = tc_name
        self.dscp     = TC_DSCP[tc_name]
        self.ipv6_tc  = (self.dscp << 2) & 0xFF
        self.seg_list = list(seg_list)
        self.tetlv_chain = []
        self.hops_done   = 0

    def add_rec(self, r):
        self.tetlv_chain.append(r)
        self.hops_done += 1

    def header_text(self):
        algo = TC_ALGO[self.tc_name]
        L = [
            "+-  IPv6 Base Header  ------------------------------------------+",
            f"|  TC: 0x{self.ipv6_tc:02X}  DSCP={self.dscp:2d} ({self.tc_name})  Algo={algo}                    |",
            f"|  Payload: {JUMBO_FRAME_B} B (Jumbo)  SRv6 OH: {SRV6_OH_B} B  "
            f"Total: {FRAME_TOTAL_B} B  Link: {LINK_RATE_GBPS}G       |",
            f"|  Ser-delay: {SER_DELAY_US:.3f} us/frame  Next Hdr: 43 (SRH)           |",
            "+-  SRH Segment List (RFC 8986)  --------------------------------+",
        ]
        for s in self.seg_list:
            L.append(f"|  {s:<62s}|")
        L.append(f"+-  TE-TLV Chain  ({len(self.tetlv_chain)} records x 40 B)  "
                 f"Jumbo={JUMBO_FRAME_B}B  Ser={SER_DELAY_US:.2f}us  ----------+")
        if not self.tetlv_chain:
            L.append("|  (no records yet)                                             |")
        for r in self.tetlv_chain:
            cf = "C=1!" if r.c_flag else "C=0 "
            L += [
                f"|  [{r.node:3s}] DSCP={r.dscp:2d}({r.tc_name}) {cf}  Budget={r.budget_pct:5.1f}%              |",
                f"|  PTPi={r.ptp_in_ns%10**9:13d} ns   PTPo={r.ptp_eg_ns%10**9:13d} ns  |",
                f"|  Res={r.residence_us:8.2f} us  Drops={r.drops:<4d}  "
                f"Q5={r.queue_fill.get('TC5',0)*100:4.1f}%                |",
            ]
        L.append("+---------------------------------------------------------------+")
        return "\n".join(L)


# ==================================================================
#  SIMULATION STATE
# ==================================================================
cong_level = {n: 0.0 for n in CTRL_NODES}
tc_volume  = {tc: 0.5 for tc in TCS}


class NodeProcessor:
    def __init__(self, name):
        self.name   = name
        self.queue  = {tc: 0.0 for tc in TCS}
        self.drops  = {tc: 0   for tc in TCS}
        self.rtt_us = {tc:10.0 for tc in TCS}
        self.wrr_w  = {tc: 1.0 for tc in TCS}
        self.c_flag = False

    def tick(self, t):
        sev = cong_level.get(self.name, 0.0)
        for tc in TCS:
            base = 0.06 + 0.02 * math.sin(t * 0.25 + TC_DSCP[tc] * 0.01)
            vol  = tc_volume.get(tc, 0.5)
            if sev > 0.02:
                qos          = QOS_PROT[tc]
                jumbo_factor = FRAME_TOTAL_B / 1500.0
                spike = sev * qos * max(0.15, vol) * min(2.5, jumbo_factor * 0.42) \
                      + 0.04 * math.sin(t * 0.9 + TC_DSCP[tc] * 0.02)
                self.queue[tc]  = min(1.0, base * vol + spike + random.gauss(0, .018))
                self.rtt_us[tc] = 8 + SER_DELAY_US + self.queue[tc] * 160 * (1 + 0.3 * qos) + random.gauss(0, 2)
                if self.queue[tc] > 0.85:
                    self.drops[tc] += random.randint(0, 3)
                    self.wrr_w[tc]  = max(0.1, self.wrr_w[tc] * 0.93)
            else:
                self.queue[tc]  = max(0.0, base * max(0.05, vol) + random.gauss(0, .010))
                self.rtt_us[tc] = 8 + SER_DELAY_US + self.queue[tc] * 25
                self.wrr_w[tc]  = min(5.0, self.wrr_w[tc] * 1.01 + 0.004)
        self.c_flag = any(self.queue[tc] > .80 for tc in ("TC5","TC6"))

    def stamp(self, pkt, st):
        ing    = PKT_BASE_NS + int(st * 1e9) + random.randint(0, 300)
        ser_ns = int(SER_DELAY_US * 1000)
        res    = int(self.rtt_us[pkt.tc_name] * 1000 + ser_ns + random.gauss(0, 150))
        eg     = ing + max(ser_ns, res)
        r      = TETLVRecord(self.name, pkt.tc_name, self.queue, ing, eg,
                             self.drops[pkt.tc_name])
        pkt.add_rec(r)
        return r


nodes = {n: NodeProcessor(n) for n in ALL_NODES}


def route_packet(tc_name, path, st):
    seg = ([NODE_SID[n] for n in ["Server-B","P5","P4","P3","P2","P1","Server-A"]]
           if path == "upper" else
           [NODE_SID[n] for n in ["Server-B","P5","P8","P7","P6","P1","Server-A"]])
    pkt = SRv6Packet(-99, tc_name, seg)
    for nd in (PATH_UPPER if path == "upper" else PATH_LOWER)[1:-1]:
        nodes[nd].stamp(pkt, st)
    return sum(r.residence_us for r in pkt.tetlv_chain), pkt


# ==================================================================
#  ML AGENT  -  all 5 authentic algorithms
# ==================================================================
class MLAgent:
    def __init__(self):
        self.alpha        = {tc: {"upper":1.0,"lower":1.0} for tc in TCS}
        self.beta         = {tc: {"upper":1.0,"lower":1.0} for tc in TCS}
        self.frac         = {tc: 0.5 for tc in TCS}
        self.momentum     = 0.20
        self.log          = deque(maxlen=14)
        self.dwrr_deficit = {tc:{"upper":0.0,"lower":0.0} for tc in ("TC0","TC1")}
        # LConn: exponentially-decayed connection counts per path
        self.conn         = {tc:{"upper":1.0,"lower":1.0} for tc in TCS}
        # RB: per-path resource composite score history
        self.rb_score     = {tc:{"upper":0.5,"lower":0.5} for tc in TCS}

    def _aq(self, path, tc):
        pns = PATH_UPPER if path == "upper" else PATH_LOWER
        return sum(nodes[n].queue[tc] for n in pns[1:-1]) / (len(pns) - 2)

    def _dwrr(self, tc):
        q     = DWRR_QUANTUM[tc]
        tok_u = q + self.dwrr_deficit[tc]["upper"]
        tok_l = q + self.dwrr_deficit[tc]["lower"]
        cap_u = tok_u / max(0.01, 1.0 + self._aq("upper", tc) * 3.0)
        cap_l = tok_l / max(0.01, 1.0 + self._aq("lower", tc) * 3.0)
        total = cap_u + cap_l
        frac  = cap_u / total if total > 1e-9 else 0.5
        decay = 0.94; gain = 0.12
        self.dwrr_deficit[tc]["upper"] = max(0.0, min(q*2,
            self.dwrr_deficit[tc]["upper"] * decay + max(0, 0.5-frac) * q * gain))
        self.dwrr_deficit[tc]["lower"] = max(0.0, min(q*2,
            self.dwrr_deficit[tc]["lower"] * decay + max(0, frac-0.5) * q * gain))
        return frac

    def _lconn(self, tc):
        """
        Least Connected: route to path with fewer active connections.
        Connection count is flow-based (decaying counter incremented per routed pkt).
        Distinct from LCon which is queue-fill based.
        """
        cu = self.conn[tc]["upper"]
        cl = self.conn[tc]["lower"]
        if cu + cl < 0.01:
            return 0.5
        return cl / (cu + cl)

    def _rb(self, tc):
        """
        Resource-Based: composite score combining queue fill (40%),
        normalised RTT (40%), and drop rate (20%) per path.
        Routes proportionally to path with better composite resource score.
        """
        def path_score(path):
            pns = PATH_UPPER if path == "upper" else PATH_LOWER
            s = []
            for n in pns[1:-1]:
                q   = nodes[n].queue[tc]
                rtt = min(1.0, nodes[n].rtt_us[tc] / 300.0)
                drp = min(1.0, nodes[n].drops[tc] / 100.0)
                s.append(0.40*q + 0.40*rtt + 0.20*drp)
            return sum(s)/len(s) if s else 0.5
        su = path_score("upper")
        sl = path_score("lower")
        alpha = 0.25
        self.rb_score[tc]["upper"] = (1-alpha)*self.rb_score[tc]["upper"] + alpha*su
        self.rb_score[tc]["lower"] = (1-alpha)*self.rb_score[tc]["lower"] + alpha*sl
        su_s = self.rb_score[tc]["upper"]
        sl_s = self.rb_score[tc]["lower"]
        return sl_s / (su_s + sl_s) if (su_s + sl_s) > 1e-9 else 0.5

    def _lcon(self, tc):
        """Proportional least-congestion: ql/(qu+ql)."""
        qu = self._aq("upper", tc)
        ql = self._aq("lower", tc)
        if qu + ql < 0.01:
            return 0.5
        return ql / (qu + ql)

    def _lrt(self, tc):
        """Continuous Thompson Sampling: tu/(tu+tl)."""
        tu = np.random.beta(self.alpha[tc]["upper"], self.beta[tc]["upper"])
        tl = np.random.beta(self.alpha[tc]["lower"], self.beta[tc]["lower"])
        if tu + tl < 1e-9:
            return 0.5
        return tu / (tu + tl)

    def update_ts(self, tc, path, res_us):
        bud = TC_BUDGET_US[tc]
        if res_us <= bud:
            self.alpha[tc][path] += 1.0
        else:
            excess = (res_us - bud) / bud
            self.beta[tc][path] += min(3.0, 0.5 + excess)
        for p in ("upper","lower"):
            self.alpha[tc][p] = max(1.0, self.alpha[tc][p] * 0.997)
            self.beta[tc][p]  = max(1.0, self.beta[tc][p]  * 0.997)

    def decide(self, t):
        prev      = {tc: self.frac[tc] for tc in TCS}
        upper_sev = max(cong_level.get(n, 0.0) for n in UPPER_CTRL)
        lower_sev = max(cong_level.get(n, 0.0) for n in LOWER_CTRL)
        for tc in TCS:
            a   = TC_ALGO[tc]
            tgt = (self._dwrr(tc)  if a == "DWRR"  else
                   self._lconn(tc) if a == "LConn" else
                   self._rb(tc)    if a == "RB"    else
                   self._lcon(tc)  if a == "LCon"  else
                   self._lrt(tc))
            if upper_sev >= 0.85:
                tgt = min(tgt, 0.04)
            elif lower_sev >= 0.85:
                tgt = max(tgt, 0.96)
            sev  = max(upper_sev, lower_sev)
            mom  = min(0.85, self.momentum + sev * 0.65)
            self.frac[tc] = (1 - mom) * self.frac[tc] + mom * tgt
        for tc in ("TC5","TC6"):
            wu = prev[tc] > .5
            nu = self.frac[tc] > .5
            if wu != nu:
                self.log.appendleft(
                    f"t={t:6.1f}s | {tc}[LRT] {'->UPPER' if nu else '->LOWER'} | "
                    f"f={self.frac[tc]:.2f} | "
                    f"a_u={self.alpha[tc]['upper']:.1f} b_u={self.beta[tc]['upper']:.1f} | "
                    f"Q5@P7={nodes['P7'].queue['TC5']*100:.0f}% "
                    f"Q5@P3={nodes['P3'].queue['TC5']*100:.0f}%")

    def route(self, tc_name):
        path = "upper" if random.random() < self.frac[tc_name] else "lower"
        # Exponential decay + increment for LConn tracking
        self.conn[tc_name]["upper"] *= 0.995
        self.conn[tc_name]["lower"] *= 0.995
        self.conn[tc_name][path]    += 1.0
        return path

    def seg_list(self, path):
        return ([NODE_SID[n] for n in ["Server-B","P5","P4","P3","P2","P1","Server-A"]]
                if path == "upper" else
                [NODE_SID[n] for n in ["Server-B","P5","P8","P7","P6","P1","Server-A"]])


agent = MLAgent()

# ==================================================================
#  PARTICLE SYSTEM
# ==================================================================
phases = {tc: {p: [k/N_PARTS for k in range(N_PARTS)]
               for p in ("upper","lower")} for tc in TCS}


def path_xy(path, phase, tc_idx=3):
    pts = [NODE_POS[n] for n in (PATH_UPPER if path == "upper" else PATH_LOWER)]
    n   = len(pts) - 1
    seg = min(int(phase * n), n - 1)
    tf  = phase * n - seg
    x   = pts[seg][0] + tf * (pts[seg+1][0] - pts[seg][0])
    y   = pts[seg][1] + tf * (pts[seg+1][1] - pts[seg][1])
    return x, y + (tc_idx - 3) * 0.013


# ==================================================================
#  RING BUFFERS
# ==================================================================
BUFLEN   = 800
lt_buf   = deque(maxlen=BUFLEN)
li_buf   = deque(maxlen=BUFLEN)
ct_buf   = deque(maxlen=BUFLEN)
ci_buf   = deque(maxlen=BUFLEN)
frac_buf = {tc: deque(maxlen=BUFLEN) for tc in TCS}
vtc_t    = {tc: 0 for tc in TCS}
vtc_i    = {tc: 0 for tc in TCS}
cv_t     = [0];  cv_i = [0]
# Per-TC rolling avg latency buffers (last 50 pkts)
ROLL_N    = 50
lat_tetlv = {tc: deque(maxlen=ROLL_N) for tc in TCS}
lat_ioam  = {tc: deque(maxlen=ROLL_N) for tc in TCS}
# Bandwidth tracking: bytes successfully delivered (no SLA violation)
bw_tetlv  = {tc: 0 for tc in TCS}
bw_ioam   = {tc: 0 for tc in TCS}
# Rolling violation rate (last 20 steps)
vrate_t   = deque(maxlen=20); vrate_i = deque(maxlen=20)
vr_t_buf  = deque(maxlen=BUFLEN); vr_i_buf = deque(maxlen=BUFLEN)
t_buf     = deque(maxlen=BUFLEN)
pkt_id    = [0];  last_pkt = [None]
sim_t     = [0.0]; spd = [2.0]
running   = [False]

# ==================================================================
#  FIGURE  +  AXES
# ==================================================================
BG  = "#080816"
fig = plt.figure(figsize=(21,13), facecolor="#0d0d22")
fig.canvas.manager.set_window_title("SRv6 TE-TLV v2 - Authentic Per-TC Simulation")
gs  = gridspec.GridSpec(3, 4, figure=fig,
      top=.95, bottom=.30, left=.04, right=.99, hspace=.52, wspace=.32)
ax_tp = fig.add_subplot(gs[0, :])
ax_la = fig.add_subplot(gs[1, :2]);  ax_cv = fig.add_subplot(gs[2, :2])
ax_br = fig.add_subplot(gs[1, 2:]); ax_fr = fig.add_subplot(gs[2, 2:])

for ax in (ax_tp, ax_la, ax_cv, ax_br, ax_fr):
    ax.set_facecolor(BG)
    ax.tick_params(colors="#8888bb", labelsize=6.5)
    for sp in ax.spines.values(): sp.set_color("#20204a")
    ax.title.set_color("#ccccff")
    ax.xaxis.label.set_color("#8888bb")
    ax.yaxis.label.set_color("#8888bb")

ax_tp.set_xlim(-.5, 6.5); ax_tp.set_ylim(-.18, 1.25); ax_tp.axis("off")
ax_tp.set_title(
    "Live Network  |  Coloured dots = TC-tagged packets  |  "
    "Edge colour = dominant TC  |  Node rings = congestion  |  "
    "Both TE-TLV and IOAM measured on ACTUAL routed packets",
    fontsize=8.5, fontweight="bold")

edge_ln = {}
for e in EDGES_U + EDGES_L:
    x = [NODE_POS[e[0]][0], NODE_POS[e[1]][0]]
    y = [NODE_POS[e[0]][1], NODE_POS[e[1]][1]]
    ln, = ax_tp.plot(x, y, "-", color="#1a1a3a", lw=1.5, zorder=1)
    edge_ln[e] = ln

nc = {}; nb = {}
for name, (x, y) in NODE_POS.items():
    circ = plt.Circle((x,y), .065, color="#141428", ec="#3355aa", lw=1.8, zorder=3)
    ax_tp.add_patch(circ); nc[name] = circ
    ax_tp.text(x, y-.14, name, ha="center", va="top", fontsize=6, color="#7788bb")
    if name.startswith("P"):
        bars = []
        for i, tc in enumerate(TCS):
            bar = ax_tp.bar(x-.22 + i*.073, 0, .060,
                            bottom=y+.09, color=TC_C[i], zorder=4, alpha=.85)
            bars.append(bar)
        nb[name] = bars

for i, tc in enumerate(TCS):
    ax_tp.bar(-.5, -1, .05, color=TC_C[i],
              label=f"{tc} D={TC_DSCP[tc]} [{TC_ALGO[tc]}] QoS={QOS_PROT[tc]:.2f}")
ax_tp.legend(loc="upper right", fontsize=5.2, facecolor=BG,
             labelcolor="white", ncol=7, framealpha=.85)

part_scat = {}
for i, tc in enumerate(TCS):
    sc = ax_tp.scatter([], [], s=38, color=TC_C[i], zorder=8+i,
                       alpha=0.9, edgecolors="#ffffff", linewidths=0.5)
    part_scat[tc] = sc

pl = ax_tp.text(3, 1.20, "Initializing... Press Start",
                ha="center", va="top", fontsize=8,
                color="#00ffcc", fontweight="bold")

# Chart 1 - Per-TC latency bar chart
ax_la.set_title(
    "Per-TC Rolling Avg Latency (last 50 pkts, µs)  |  TE-TLV v2 vs IOAM ECMP  |  -- = SLA budget",
    fontsize=8)
ax_la.set_ylabel("Avg Latency (µs)", fontsize=7)
lat_xp     = np.arange(len(TCS)); lat_bw = 0.35
lat_bars_t = ax_la.bar(lat_xp - lat_bw/2, [0]*7, lat_bw, color=TC_C, alpha=.87,
                       label="TE-TLV v2 (ML-routed)")
lat_bars_i = ax_la.bar(lat_xp + lat_bw/2, [0]*7, lat_bw, color=TC_C, hatch="//",
                       alpha=.55, label="IOAM ECMP (blind 50/50)")
for i_bgt, tc_bgt in enumerate(TCS):
    ax_la.plot([i_bgt - lat_bw*1.1, i_bgt + lat_bw*1.1],
               [TC_BUDGET_US[tc_bgt]]*2,
               color="white", lw=1.8, ls="--", alpha=0.9, zorder=10)
    ax_la.text(i_bgt, TC_BUDGET_US[tc_bgt]*1.04,
               f"{TC_BUDGET_US[tc_bgt]}µs", ha="center", fontsize=4.5,
               color="#ccccff", zorder=11)
ax_la.set_xticks(lat_xp)
ax_la.set_xticklabels(
    [f"{tc}\n[{TC_ALGO[tc]}]\nDSCP={TC_DSCP[tc]}" for tc in TCS], fontsize=5)
ax_la.legend(loc="upper right", fontsize=5.5, facecolor=BG, labelcolor="white")

# Chart 2 - SLA violations
ax_cv.set_title(
    "SLA Violations  |  Cumulative (lines) + Rolling Rate/20-steps (dots)  |  Same budget",
    fontsize=8)
ax_cv.set_ylabel("Violations", fontsize=7)
ax_cv.set_xlabel("Sim Time (s)", fontsize=7)
l_ct,  = ax_cv.plot([], [], color="#69f0ae", lw=1.6, label="TE-TLV v2 cumul")
l_ci,  = ax_cv.plot([], [], color="#ff7043", lw=1.4, ls="--", label="IOAM ECMP cumul")
lr_ct, = ax_cv.plot([], [], color="#69f0ae", lw=0, marker="o", ms=3, alpha=0.6,
                    label="TE-TLV rate/20")
lr_ci, = ax_cv.plot([], [], color="#ff7043", lw=0, marker="o", ms=3, alpha=0.6,
                    label="IOAM rate/20")
ax_cv.legend(loc="upper left", fontsize=5.5, facecolor=BG, labelcolor="white", ncol=2)

# Chart 3 - Per-TC violations bar
ax_br.set_title(
    "Per-TC SLA Violations  |  TE-TLV v2  vs  IOAM ECMP", fontsize=8)
ax_br.set_ylabel("Violations", fontsize=7)
xp = np.arange(len(TCS)); bw = 0.35
bars_t = ax_br.bar(xp-bw/2, [0]*7, bw, color=TC_C, alpha=.87, label="TE-TLV v2")
bars_i = ax_br.bar(xp+bw/2, [0]*7, bw, color=TC_C, hatch="//", alpha=.55,
                   label="IOAM ECMP")
ax_br.set_xticks(xp)
ax_br.set_xticklabels(
    [f"{tc}\n[{TC_ALGO[tc]}]\n{TC_BUDGET_US[tc]}us\nQoS={QOS_PROT[tc]}"
     for tc in TCS], fontsize=4.8)
ax_br.legend(loc="upper left", fontsize=6, facecolor=BG, labelcolor="white")

# Chart 4 - Per-TC path fractions
ax_fr.set_title(
    "Per-TC Upper-Path Fraction  (live ML decisions)", fontsize=8)
ax_fr.set_ylabel("Fraction -> UPPER path", fontsize=7)
ax_fr.set_xlabel("Sim Time (s)", fontsize=7)
fr_lines = {}
for i, tc in enumerate(TCS):
    ls = ("-"  if TC_ALGO[tc] == "LRT"   else
          "--" if TC_ALGO[tc] == "LCon"  else
          "-." if TC_ALGO[tc] == "DWRR"  else
          ":"  if TC_ALGO[tc] == "LConn" else "-")
    ln, = ax_fr.plot([], [], color=TC_C[i], lw=1.5, ls=ls,
                     label=f"{tc}[{TC_ALGO[tc]}]")
    fr_lines[tc] = ln
ax_fr.legend(loc="upper right", fontsize=5.5, facecolor=BG,
             labelcolor="white", ncol=4)
ax_fr.set_ylim(-.05, 1.05)
ax_fr.axhline(0.5, color="#555577", lw=.8, ls="--", alpha=.6)

# ==================================================================
#  CONTROL PANEL
# ==================================================================
fig.text(.04, .278, "NODE CONGESTION  (0=none  1=max)",
         color="#bb99ff", fontsize=7, fontweight="bold")
fig.text(.53, .278, "TC TRAFFIC VOLUME  (0=no traffic  1=max)",
         color="#99ffbb", fontsize=7, fontweight="bold")
fig.text(.04, .138, "LIVE SRv6 PACKET  |  TE-TLV CHAIN",
         color="#aaffaa", fontsize=6.5, fontweight="bold")
fig.text(.56, .138, "TRA->MPA->PSA  LOG  (per-TC decisions)",
         color="#ffaaaa", fontsize=6.5, fontweight="bold")

sl_cong = {}
for node, x, y in [("P2",.04,.250),("P3",.18,.250),("P4",.32,.250),
                    ("P6",.04,.225),("P7",.18,.225),("P8",.32,.225)]:
    axs = plt.axes([x, y, .13, .017], facecolor="#111128")
    sl  = Slider(axs, node, 0., 1., valinit=0., color="#4a3a7a")
    sl.label.set_color("#cc99ff");   sl.label.set_fontsize(7)
    sl.valtext.set_color("#ffcc00"); sl.valtext.set_fontsize(6.5)
    def _cn(v, n=node): cong_level[n] = v
    sl.on_changed(_cn)
    sl_cong[node] = sl

sl_vol = {}
for tc, x, y in [("TC0",.53,.250),("TC1",.63,.250),("TC2",.73,.250),("TC3",.83,.250),
                  ("TC4",.53,.225),("TC5",.63,.225),("TC6",.73,.225)]:
    c   = TC_C[TCS.index(tc)]
    axs = plt.axes([x, y, .09, .017], facecolor="#111128")
    sl  = Slider(axs, tc, 0., 1., valinit=0.5, color=c)
    sl.label.set_color(c);           sl.label.set_fontsize(7)
    sl.valtext.set_color("#ffffff"); sl.valtext.set_fontsize(6.5)
    def _tv(v, tc_=tc): tc_volume[tc_] = v
    sl.on_changed(_tv)
    sl_vol[tc] = sl

ax_al = plt.axes([.84, .205, .15, .068], facecolor=BG); ax_al.axis("off")
ax_al.text(.04, .98,
    "Algorithms (authentic):\n"
    "DWRR  TC0,TC1  deficit WRR (9K/18K quantum)\n"
    "LConn TC2      least-connected (flow count)\n"
    "RB    TC3      resource-based (Q+RTT+drop)\n"
    "LCon  TC4      proportional ql/(qu+ql)\n"
    "LRT   TC5,TC6  tu/(tu+tl) Thompson",
    transform=ax_al.transAxes, fontsize=5.2, va="top",
    color="#ccddff", family="monospace")

ax_spd = plt.axes([.53, .198, .19, .017], facecolor="#111128")
sl_spd = Slider(ax_spd, "Speed", .5, 8., valinit=2., color="#223344")
sl_spd.label.set_color("#aaaacc");   sl_spd.label.set_fontsize(7)
sl_spd.valtext.set_color("#00e5ff"); sl_spd.valtext.set_fontsize(7)
sl_spd.on_changed(lambda v: spd.__setitem__(0, v))

b_s  = Button(plt.axes([.53,.174,.07,.025], facecolor="#0d0d22"),
              "Start",  color="#0a250a", hovercolor="#1a5a1a")
b_p  = Button(plt.axes([.61,.174,.07,.025], facecolor="#0d0d22"),
              "Stop",   color="#250a0a", hovercolor="#5a1a1a")
b_r  = Button(plt.axes([.69,.174,.07,.025], facecolor="#0d0d22"),
              "Reset",  color="#0a0a25", hovercolor="#1a1a5a")
b_c  = Button(plt.axes([.77,.174,.09,.025], facecolor="#0d0d22"),
              "Clear Cong", color="#252500", hovercolor="#4a4a00")
b_rs = Button(plt.axes([.87,.174,.08,.025], facecolor="#0d0d22"),
              "Results",  color="#001525", hovercolor="#003355")
for b in (b_s, b_p, b_r, b_c, b_rs):
    b.label.set_color("white"); b.label.set_fontsize(7)


def on_start(e):  running[0] = True
def on_stop(e):   running[0] = False
def on_clear(e):
    for n in CTRL_NODES:
        cong_level[n] = 0.; sl_cong[n].set_val(0.)
def on_reset(e):
    running[0] = False; sim_t[0] = 0.
    for d in (t_buf, lt_buf, li_buf, ct_buf, ci_buf): d.clear()
    for tc in TCS:
        frac_buf[tc].clear(); vtc_t[tc] = 0; vtc_i[tc] = 0
    cv_t[0] = 0; cv_i[0] = 0; pkt_id[0] = 0; last_pkt[0] = None
    vrate_t.clear(); vrate_i.clear(); vr_t_buf.clear(); vr_i_buf.clear()
    for _tc in TCS:
        lat_tetlv[_tc].clear(); lat_ioam[_tc].clear()
        bw_tetlv[_tc] = 0;  bw_ioam[_tc] = 0
    for n in ALL_NODES: nodes[n].__init__(n)
    agent.__init__(); on_clear(None)
    for tc in TCS:
        for p in ("upper","lower"):
            phases[tc][p] = [k/N_PARTS for k in range(N_PARTS)]


def show_results(e):
    """Pop-up: overall latency, bandwidth and SLA violation comparison."""
    if not any(lat_tetlv[tc] for tc in TCS):
        return
    fig_r, axs_r = plt.subplots(1, 3, figsize=(17, 6), facecolor="#0d0d22")
    fig_r.suptitle(
        "SRv6 TE-TLV v2  vs  IOAM ECMP  |  Overall Results\n"
        "Jumbo 9000B  |  100GbE  |  DWRR / LConn / RB / LCon / LRT",
        color="white", fontsize=11, fontweight="bold")
    xp_r = np.arange(len(TCS)); bwr = 0.35

    def _style(ax, title, ylabel):
        ax.set_facecolor("#080816")
        ax.set_title(title, color="#ccccff", fontsize=9)
        ax.set_ylabel(ylabel, color="#8888bb", fontsize=8)
        ax.tick_params(colors="#8888bb", labelsize=7)
        for sp in ax.spines.values(): sp.set_color("#20204a")
        ax.set_xticks(xp_r)
        ax.set_xticklabels(
            [f"{tc}\n[{TC_ALGO[tc]}]" for tc in TCS],
            fontsize=6.5, color="white")
        ax.legend(fontsize=7, facecolor="#0d0d22", labelcolor="white")

    # Chart A: Avg latency per TC
    avg_t = [float(np.mean(lat_tetlv[tc])) if lat_tetlv[tc] else 0 for tc in TCS]
    avg_i = [float(np.mean(lat_ioam[tc]))  if lat_ioam[tc]  else 0 for tc in TCS]
    axs_r[0].bar(xp_r-bwr/2, avg_t, bwr, color=TC_C, alpha=.87, label="TE-TLV v2")
    axs_r[0].bar(xp_r+bwr/2, avg_i, bwr, color=TC_C, hatch="//", alpha=.55,
                 label="IOAM ECMP")
    for i_r, tc_r in enumerate(TCS):
        axs_r[0].plot([i_r-bwr*1.1, i_r+bwr*1.1], [TC_BUDGET_US[tc_r]]*2,
                      "w--", lw=1.5, alpha=0.85)
        axs_r[0].text(i_r, TC_BUDGET_US[tc_r]*1.03,
                      f"{TC_BUDGET_US[tc_r]}µs", ha="center",
                      fontsize=4.5, color="#ccccff")
    _style(axs_r[0], "Avg Latency per TC (µs)  |  -- = SLA Budget", "Avg Latency (µs)")

    # Chart B: Effective throughput (Mbps - SLA-compliant only)
    st = max(sim_t[0], 1.0)
    bw_t_mbps = [bw_tetlv[tc]*8/st/1e6 for tc in TCS]
    bw_i_mbps = [bw_ioam[tc] *8/st/1e6 for tc in TCS]
    axs_r[1].bar(xp_r-bwr/2, bw_t_mbps, bwr, color=TC_C, alpha=.87, label="TE-TLV v2")
    axs_r[1].bar(xp_r+bwr/2, bw_i_mbps, bwr, color=TC_C, hatch="//", alpha=.55,
                 label="IOAM ECMP")
    _style(axs_r[1],
           "Effective Throughput per TC (Mbps)\n(SLA-compliant frames only)",
           "Throughput (Mbps)")

    # Chart C: % improvement (latency + violation + throughput)
    pct_lat = [(avg_i[i]-avg_t[i])/avg_i[i]*100 if avg_i[i]>0 else 0
               for i in range(len(TCS))]
    pct_vio = [(vtc_i[tc]-vtc_t[tc])/max(1,vtc_i[tc])*100 for tc in TCS]
    pct_bw  = [(bw_t_mbps[i]-bw_i_mbps[i])/max(0.001,bw_i_mbps[i])*100
               for i in range(len(TCS))]
    x3 = np.arange(len(TCS))
    axs_r[2].bar(x3-bwr/2-.02, pct_lat, bwr*.65, color="#00e5ff", alpha=.85,
                 label="Latency reduction %")
    axs_r[2].bar(x3,           pct_vio, bwr*.65, color="#69f0ae", alpha=.85,
                 label="Violation reduction %")
    axs_r[2].bar(x3+bwr/2+.02, pct_bw,  bwr*.65, color="#ff9800", alpha=.85,
                 label="Throughput gain %")
    axs_r[2].axhline(0, color="white", lw=0.8, ls="--", alpha=0.5)
    _style(axs_r[2],
           "TE-TLV v2 Improvement vs IOAM ECMP (%)",
           "% Improvement")
    fig_r.tight_layout(rect=[0,.02,1,.90])
    plt.show()


b_s.on_clicked(on_start);  b_p.on_clicked(on_stop)
b_r.on_clicked(on_reset);  b_c.on_clicked(on_clear)
b_rs.on_clicked(show_results)

ax_hd = plt.axes([.04,.01,.50,.125], facecolor="#030310"); ax_hd.axis("off")
hdr_t = ax_hd.text(.01, .97, "Waiting for first packet...",
    transform=ax_hd.transAxes, fontsize=4.9, va="top",
    color="#aaffcc", family="monospace")

ax_lg = plt.axes([.56,.01,.43,.125], facecolor="#030310"); ax_lg.axis("off")
lg_t  = ax_lg.text(.01, .97, "TRA scanning per-TC TE-TLV feedback...",
    transform=ax_lg.transAxes, fontsize=4.9, va="top",
    color="#ccffcc", family="monospace")

# ==================================================================
#  TOPOLOGY UPDATE
# ==================================================================
def upd_topo(t):
    def path_dominant(path):
        scores = {tc: (agent.frac[tc] if path == "upper" else 1-agent.frac[tc])
                      * tc_volume[tc] for tc in TCS}
        dom   = max(scores, key=scores.get)
        total = sum(scores.values())
        return TC_C[TCS.index(dom)], min(1.0, total*1.4), total

    uc, ua, ut = path_dominant("upper")
    lc, la, lt = path_dominant("lower")
    for e in EDGES_U:
        edge_ln[e].set_color(uc)
        edge_ln[e].set_linewidth(1 + ut*3.5)
        edge_ln[e].set_alpha(max(0.2, ua))
    for e in EDGES_L:
        edge_ln[e].set_color(lc)
        edge_ln[e].set_linewidth(1 + lt*3.5)
        edge_ln[e].set_alpha(max(0.2, la))

    for name, circ in nc.items():
        sev = cong_level.get(name, 0.)
        if sev > .12:
            g = int(max(0, 1-sev)*55)
            circ.set_facecolor(f"#{int(sev*160+50):02x}{g:02x}00")
            circ.set_edgecolor("#ff1111" if sev>.6 else "#ffaa00")
            circ.set_linewidth(2.5)
        else:
            circ.set_facecolor("#0a1e2e")
            circ.set_edgecolor("#3366aa")
            circ.set_linewidth(1.5)

    for name, bars in nb.items():
        for i, (tc, bar) in enumerate(zip(TCS, bars)):
            q = nodes[name].queue[tc]
            bar[0].set_height(q*.22)
            bar[0].set_color("#ff1111" if q>.85 else TC_C[i])

    for tc_idx, tc in enumerate(TCS):
        frac  = agent.frac[tc]
        vol   = max(0.08, tc_volume[tc])
        speed = 0.009 + vol*0.015
        xs=[]; ys=[]; szs=[]
        for path, pf in [("upper", frac), ("lower", 1-frac)]:
            if pf < 0.01: continue
            phases[tc][path] = [(p+speed)%1.0 for p in phases[tc][path]]
            pt_size = max(22, vol*90*pf+14)
            for ph in phases[tc][path]:
                x, y = path_xy(path, ph, tc_idx)
                xs.append(x); ys.append(y); szs.append(pt_size)
        if xs:
            part_scat[tc].set_offsets(np.c_[xs, ys])
            part_scat[tc].set_sizes(szs)
            part_scat[tc].set_alpha(min(0.95, 0.45+vol*0.55))
        else:
            part_scat[tc].set_offsets(np.zeros((0,2)))

    pl.set_text(
        "t={:.1f}s  |  ".format(t) +
        "  ".join("{:s}={:s}{:.0%}".format(
            tc, "U" if agent.frac[tc]>.5 else "L", agent.frac[tc])
            for tc in TCS) +
        "  |  P7={:s}  P3={:s}".format(
            "!" if nodes["P7"].c_flag else "ok",
            "!" if nodes["P3"].c_flag else "ok")
    )


# ==================================================================
#  SIM STEP  -  unified per-TC loop (TC0-TC6), both systems
# ==================================================================
def sim_step(t):
    for n in ALL_NODES:
        nodes[n].tick(t)
    agent.decide(t)

    wts    = [tc_volume[tc] for tc in TCS]
    tc_n   = random.choices(TCS, weights=wts)[0]
    cp     = agent.route(tc_n)
    _, pkt = route_packet(tc_n, cp, t)
    pkt.pid = pkt_id[0]; pkt_id[0] += 1
    last_pkt[0] = pkt

    step_vt = 0; step_vi = 0
    for tc in TCS:
        if tc_volume[tc] < 0.05:
            lat_tetlv[tc].clear(); lat_ioam[tc].clear()
            continue

        tv_p = agent.route(tc)
        tv_res, _ = route_packet(tc, tv_p, t)
        tv_viol = 1 if tv_res > TC_BUDGET_US[tc] else 0
        lat_tetlv[tc].append(tv_res)

        ioam_p = "upper" if random.random() < 0.5 else "lower"
        ioam_res, _ = route_packet(tc, ioam_p, t)
        ioam_viol = 1 if ioam_res > TC_BUDGET_US[tc] else 0
        lat_ioam[tc].append(ioam_res)

        if tc in ("TC5","TC6"):
            lt_buf.append(tv_res   / 1000.0)
            li_buf.append(ioam_res / 1000.0)

        vtc_t[tc] += tv_viol
        vtc_i[tc] += ioam_viol
        step_vt   += tv_viol
        step_vi   += ioam_viol

        # Bandwidth: count bytes on SLA-compliant deliveries only
        bw_tetlv[tc] += 0 if tv_viol   else FRAME_TOTAL_B
        bw_ioam[tc]  += 0 if ioam_viol else FRAME_TOTAL_B

        if TC_ALGO[tc] == "LRT":
            agent.update_ts(tc, tv_p, tv_res)

    cv_t[0] += step_vt; cv_i[0] += step_vi
    vrate_t.append(step_vt); vrate_i.append(step_vi)
    vr_t_buf.append(sum(vrate_t)); vr_i_buf.append(sum(vrate_i))
    t_buf.append(t)
    ct_buf.append(cv_t[0]); ci_buf.append(cv_i[0])
    for tc in TCS:
        frac_buf[tc].append(agent.frac[tc])


# ==================================================================
#  ANIMATION LOOP
# ==================================================================
def animate(frame):
    if not running[0]:
        return
    sim_t[0] += 0.35 * spd[0]
    t = sim_t[0]
    sim_step(t)
    upd_topo(t)
    td = list(t_buf)
    if not td:
        return

    # Chart 1: Per-TC latency bars
    _mlat = 10
    for _i, _tc in enumerate(TCS):
        _avgt = float(np.mean(lat_tetlv[_tc])) if lat_tetlv[_tc] else 0.0
        _avgi = float(np.mean(lat_ioam[_tc]))  if lat_ioam[_tc]  else 0.0
        _bgt  = TC_BUDGET_US[_tc]
        lat_bars_t[_i].set_height(_avgt)
        lat_bars_t[_i].set_facecolor("#ff2222" if _avgt > _bgt else TC_C[_i])
        lat_bars_i[_i].set_height(_avgi)
        lat_bars_i[_i].set_facecolor("#ff5533" if _avgi > _bgt else TC_C[_i])
        _mlat = max(_mlat, _avgt, _avgi, _bgt)
    ax_la.set_ylim(0, _mlat * 1.22)

    # Chart 2: Cumulative + rolling rate violations
    l_ct.set_data(td, list(ct_buf))
    l_ci.set_data(td, list(ci_buf))
    lr_ct.set_data(td, list(vr_t_buf))
    lr_ci.set_data(td, list(vr_i_buf))
    ax_cv.set_xlim(max(0, t-80), t+2)
    _cv_max = max(5, max(cv_i[0], cv_t[0])*1.2,
                  max(list(vr_i_buf)+[1])*1.5)
    ax_cv.set_ylim(0, _cv_max)

    # Chart 3: Per-TC violation bars
    mv = max(max(vtc_i.values(), default=1),
             max(vtc_t.values(), default=1), 1)
    for i, tc in enumerate(TCS):
        bars_t[i].set_height(vtc_t[tc])
        bars_i[i].set_height(vtc_i[tc])
    ax_br.set_ylim(0, max(5, mv*1.2))

    # Chart 4: Per-TC path fractions
    for tc in TCS:
        fr_lines[tc].set_data(td, list(frac_buf[tc]))
    ax_fr.set_xlim(max(0, t-80), t+2)

    if last_pkt[0]:
        hdr_t.set_text(last_pkt[0].header_text())

    if agent.log:
        lg_t.set_text("\n".join(agent.log))
    else:
        fracs = "\n".join(
            "  {:s}[{:s}]: U={:.0%}  conn_u={:.1f}  conn_l={:.1f}  vol={:.0%}".format(
                tc, TC_ALGO[tc], agent.frac[tc],
                agent.conn[tc]["upper"], agent.conn[tc]["lower"],
                tc_volume[tc])
            for tc in TCS)
        lg_t.set_text("Per-TC state (U=upper frac, conn counts for LConn):\n" + fracs)


ani = FuncAnimation(fig, animate, interval=50, blit=False, cache_frame_data=False)
plt.suptitle(
    "SRv6 TE-TLV v2  |  AUTHENTIC Simulation  |  Jumbo Frames (9000 B)  100 GbE  |  "
    "DWRR | LConn | RB | LCon | LRT(Thompson)  |  QoS DSCP  |  IOAM = real 50/50",
    color="#ffffff", fontsize=8.5, fontweight="bold", y=.985)
plt.show()
