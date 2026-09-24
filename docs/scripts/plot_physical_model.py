#!/usr/bin/env python
"""Schematic system/free-body figure of the physical model:
   (a) the 9-DOF seated upper-limb kinematic chain,
   (b) the rock-on-stick contact free-body diagram,
   (c) the key equations.
A stylised first draft -- swap panel (a) for a rendered real pose later if wanted.
Output: docs/assets/figures/physical_model.{png,pdf}
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon
import numpy as np

# --- palette -----------------------------------------------------------------
BONE = "#2b2b2b"; JOINT = "#E69F00"; NORMAL = "#D55E00"; FRIC = "#0072B2"
VEL = "#009E73"; RAIL = "#7a5230"; ROCK = "#888888"; MUT = "#666666"

fig = plt.figure(figsize=(15.5, 8.2))
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.15], height_ratios=[1.35, 1.0],
                      wspace=0.10, hspace=0.18)
axA = fig.add_subplot(gs[:, 0])   # kinematic chain (tall, left)
axB = fig.add_subplot(gs[0, 1])   # contact FBD (right top)
axC = fig.add_subplot(gs[1, 1])   # equations (right bottom)
for ax in (axA, axB, axC):
    ax.set_axis_off()

# =============================================================================
# (a) KINEMATIC CHAIN
# =============================================================================
axA.set_xlim(-0.8, 2.7); axA.set_ylim(-0.6, 3.1); axA.set_aspect("equal")
axA.set_title("(a)  9-DOF seated upper limb", loc="left", fontsize=13, weight="bold")

# joint positions (side view, stylised, roughly to human proportions)
J = {
    "pelvis":   (0.00, 0.00),
    "thoracic": (0.06, 1.25),
    "thorax":   (0.12, 1.92),
    "shoulder": (0.50, 1.86),
    "elbow":    (1.18, 1.34),
    "wrist":    (1.78, 0.96),
    "hand":     (2.02, 0.82),
}
# spine + head
axA.plot(*zip(J["pelvis"], J["thoracic"], J["thorax"]), "-", color=BONE, lw=6, solid_capstyle="round", zorder=2)
axA.add_patch(Circle((0.17, 2.36), 0.20, fc="none", ec=BONE, lw=3, zorder=2))  # head
# clavicle + arm bones
axA.plot(*zip(J["thorax"], J["shoulder"]), "-", color=BONE, lw=5, solid_capstyle="round", zorder=2)
axA.plot(*zip(J["shoulder"], J["elbow"]), "-", color=BONE, lw=6, solid_capstyle="round", zorder=2)
axA.plot(*zip(J["elbow"], J["wrist"]), "-", color=BONE, lw=6, solid_capstyle="round", zorder=2)
axA.plot(*zip(J["wrist"], J["hand"]), "-", color=BONE, lw=5, solid_capstyle="round", zorder=2)

# seat / fixed base at pelvis
axA.plot([-0.55, 0.55], [-0.28, -0.28], "-", color=MUT, lw=2)
for x in np.linspace(-0.5, 0.45, 7):
    axA.plot([x, x - 0.12], [-0.28, -0.45], "-", color=MUT, lw=1)
axA.add_patch(Polygon([(0.0, 0.0), (-0.12, -0.28), (0.12, -0.28)], closed=True, fc=MUT, ec=MUT, zorder=3))
axA.text(0.0, -0.55, "pinned (seated) base", ha="center", va="top", fontsize=9, color=MUT)

# the stick + rock at the hand
axA.add_patch(FancyBboxPatch((1.55, 0.55), 1.05, 0.13, boxstyle="round,pad=0.0,rounding_size=0.06",
                             fc="#d8c39a", ec=RAIL, lw=1.5, zorder=1))
axA.text(2.55, 0.61, "stick", ha="left", va="center", fontsize=9, color=RAIL)
axA.add_patch(Polygon([(1.90, 0.70), (2.14, 0.70), (2.08, 0.86), (1.96, 0.90)], closed=True,
                      fc=ROCK, ec=BONE, lw=1.2, zorder=3))

# joints: circle + label + DOF
def joint(name, xy, label, dof, dxy=(0.10, 0.10), ha="left"):
    axA.add_patch(Circle(xy, 0.058, fc=JOINT, ec=BONE, lw=1.4, zorder=5))
    axA.annotate(f"{label}\n{dof}", xy, xytext=(xy[0] + dxy[0], xy[1] + dxy[1]),
                 fontsize=9.5, ha=ha, va="center", color=BONE,
                 bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=JOINT, lw=1.0, alpha=0.95))

joint("thoracic", J["thoracic"], "thoracic", "1 DOF (tracked)", dxy=(-0.15, 0.05), ha="right")
joint("clavicle", (0.31, 1.90), "clavicle", "1 DOF", dxy=(-0.12, 0.28), ha="right")
joint("shoulder", J["shoulder"], "shoulder", "3 DOF (Z,X,Y)", dxy=(0.02, 0.34), ha="center")
joint("elbow", J["elbow"], "elbow", "2 DOF (Z,Y)", dxy=(0.14, 0.20))
joint("wrist", J["wrist"], "wrist", "2 DOF (Z,X)", dxy=(0.14, 0.10))
axA.annotate("rock\n(welded, 0 DOF)", J["hand"], xytext=(2.18, 0.30), fontsize=9.5, ha="left",
             va="center", color=BONE, arrowprops=dict(arrowstyle="->", color=BONE, lw=1.1),
             bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=ROCK, lw=1.0))

# gravity + DOF caption
axA.add_patch(FancyArrowPatch((-0.55, 2.7), (-0.55, 2.2), arrowstyle="-|>", mutation_scale=16,
                              color=MUT, lw=2))
axA.text(-0.5, 2.45, r"$g=-9.81\,\hat z$", fontsize=11, color=MUT, va="center")
axA.text(-0.78, -0.05, r"$u\in\mathbb{R}^{9}$ joint torques" "\n" r"$x=(q,v)\in\mathbb{R}^{18}$",
         fontsize=10, color=BONE, va="top",
         bbox=dict(boxstyle="round,pad=0.3", fc="#f4f4f4", ec=MUT, lw=1.0))

# =============================================================================
# (b) CONTACT FREE-BODY DIAGRAM
# =============================================================================
axB.set_xlim(0, 10); axB.set_ylim(0, 6.2); axB.set_aspect("equal")
axB.set_title("(b)  rock–stick contact (1-D moving rail)", loc="left", fontsize=13, weight="bold")

# stick (cylinder cross-section) + rail axis
axB.add_patch(FancyBboxPatch((0.6, 2.2), 8.8, 1.15, boxstyle="round,pad=0.0,rounding_size=0.5",
                             fc="#d8c39a", ec=RAIL, lw=2))
axB.annotate("", (0.9, 2.78), (9.1, 2.78), arrowprops=dict(arrowstyle="<->", color=RAIL, lw=1.4, ls=(0, (4, 3))))
axB.text(0.75, 2.05, r"$p_{\mathrm{start}}$", ha="center", va="top", fontsize=9, color=RAIL)
axB.text(9.25, 2.05, r"$p_{\mathrm{end}}$", ha="center", va="top", fontsize=9, color=RAIL)
axB.text(9.5, 2.9, "rail axis", ha="left", va="center", fontsize=9, color=RAIL)
axB.text(1.0, 2.78, r"$r\!\approx\!2$ cm", ha="left", va="center", fontsize=8.5, color=RAIL)

# rock pressed on top; contact point P at (5, 3.35)
Px, Py = 5.0, 3.35
axB.add_patch(Polygon([(4.1, 3.35), (5.9, 3.35), (5.55, 4.55), (4.55, 4.75), (4.05, 4.05)],
                      closed=True, fc=ROCK, ec=BONE, lw=1.6, zorder=3))
axB.text(5.0, 4.15, "rock", ha="center", va="center", fontsize=10, color="white", weight="bold")
axB.add_patch(Circle((Px, Py), 0.10, fc=BONE, ec="white", lw=1, zorder=6))
axB.text(Px + 0.15, Py + 0.05, "P", fontsize=10, color=BONE, va="bottom")

def arrow(x0, y0, x1, y1, color, lw=2.6, ms=20):
    axB.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms,
                                  color=color, lw=lw, zorder=7))

# contact frame n_hat (up), t_hat (along stick), l_hat (into page)
arrow(Px, Py, Px, Py + 1.35, MUT, lw=1.6, ms=14); axB.text(Px + 0.12, Py + 1.4, r"$\hat n$", color=MUT, fontsize=11)
arrow(Px, Py, Px + 1.35, Py, MUT, lw=1.6, ms=14); axB.text(Px + 1.4, Py + 0.12, r"$\hat t$", color=MUT, fontsize=11)
axB.add_patch(Circle((Px + 0.0, Py - 0.0), 0.0))  # spacer
axB.text(Px - 0.55, Py + 0.55, r"$\hat l\,\otimes$", color=MUT, fontsize=10)

# normal press f_n : rock pushes DOWN into stick
arrow(Px, Py + 0.05, Px, Py - 1.15, NORMAL); axB.text(Px + 0.18, Py - 0.7, r"$f_n$  (press = dual $\lambda_n$)",
                                                      color=NORMAL, fontsize=11, va="center")
# sliding velocity v_c : along +t
arrow(Px - 0.1, Py + 0.75, Px + 1.7, Py + 0.75, VEL); axB.text(Px + 1.8, Py + 0.75, r"$v_c$", color=VEL, fontsize=12, va="center")
# friction opposes slide : along -t
arrow(Px + 0.1, Py + 0.35, Px - 1.6, Py + 0.35, FRIC)
axB.text(Px - 1.7, Py + 0.35, r"$f_{\mathrm{fric}}=-\mu f_n\,\hat v_c$", color=FRIC, fontsize=11, ha="right", va="center")
# lateral deviation
arrow(Px, Py - 0.15, Px + 0.9, Py - 0.75, "#9467bd", lw=1.8, ms=14)
axB.text(Px + 0.95, Py - 0.85, r"rail_lat ($\hat l$)", color="#9467bd", fontsize=8.5, va="top")

axB.text(0.5, 5.9, "1-D holonomic: normal pinned, tool slides along $\\hat t$", fontsize=9.5,
         color=MUT, va="top")

# =============================================================================
# (c) EQUATIONS
# =============================================================================
axC.set_xlim(0, 1); axC.set_ylim(0, 1)
axC.set_title("(c)  the model, in equations", loc="left", fontsize=13, weight="bold")
eqs = [
    (r"Constrained dynamics:", r"$M(q)\ddot q + h(q,\dot q) = \tau_{\mathrm{act}} + J_c(q)^\top \lambda$"),
    (r"Actuation split (control $\neq$ muscle torque):", r"$\tau_{\mathrm{act}} = u + J_c(q)^\top f_{\mathrm{fric}}$"),
    (r"Coulomb friction ($\mu=0.3$):", r"$f_{\mathrm{fric}} = -\,\mu\, f_n\, v_c/\| v_c\|$"),
    (r"Press is a free contact dual (recovered):", r"$f_n = \lambda_n$"),
    (r"Contact features:", r"$\frac{1}{2}(f_n-f_{\mathrm{tgt}})^2,\;\; \frac{1}{2}(f_n-F_{\max})^2 \;\Rightarrow\; f_n^\star=\frac{w_{\mathrm{cap}}}{w_{\mathrm{pf}}+w_{\mathrm{cap}}}F_{\max}$"),
    (r"Time-varying IRL cost:", r"$C(t)=\sum_i w_i(t)\,\phi_i(x,u),\quad w(t)=\sum_k B_k(t)\,\theta_k$"),
]
y = 0.90
for lab, eq in eqs:
    axC.text(0.02, y, lab, fontsize=10.0, color=MUT, va="top")
    axC.text(0.06, y - 0.055, eq, fontsize=12.5, color=BONE, va="top")
    y -= 0.165

fig.suptitle("The physical model — seated human scraping a stone tool along a wooden stick",
             fontsize=15, weight="bold", y=0.985)

out = "docs/assets/figures"
os.makedirs(out, exist_ok=True)
fig.savefig(f"{out}/physical_model.png", dpi=200, bbox_inches="tight", facecolor="white")
fig.savefig(f"{out}/physical_model.pdf", bbox_inches="tight", facecolor="white")
print("wrote", f"{out}/physical_model.png", "and .pdf")
