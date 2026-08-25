"""Time-varying docs animation: the cost W(t) CHANGES over the stroke — early
progress-driven (glide toward the goal), late force-driven (press & scrape),
the analogy to the arm's approach→scrape. Panels: MuJoCo playback, side view +
penetration, emergent force, and the time-varying weights W(t)."""
import os
os.environ['MUJOCO_GL'] = 'egl'
import numpy as np, sys, mujoco
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import imageio.v2 as imageio
sys.path.insert(0, 'src')
import toy_box_slide_irl as toy

MU = toy.MU
xml = (toy._XML.replace('solref="0.4 1"', 'solref="0.1 1"')
       .replace('damping="150.0"', 'damping="800.0"')
       .replace('<worldbody>', '''<asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.86 0.86 0.9" rgb2="0.78 0.78 0.84" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="10 10" reflectance="0.0"/>
  </asset>
  <worldbody>
    <light name="key" directional="true" pos="0.3 -0.6 1.5" dir="-0.15 0.35 -1" diffuse="0.55 0.55 0.55" specular="0.15 0.15 0.15"/>
    <light name="fill" directional="true" pos="-0.5 0.4 1.2" dir="0.4 -0.3 -1" diffuse="0.35 0.35 0.35"/>''')
       .replace('rgba="0.9 0.9 0.9 1"', 'material="grid"'))
m = mujoco.MjModel.from_xml_string(xml)
gf = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'foot')
gl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'floor')

T = 240; dt = m.opt.timestep
tc = np.linspace(0, 1, T)
s = 1.0 / (1.0 + np.exp(-5.0 * (tc - 0.45)))          # smooth glide→press transition
push_c = 32 - 16 * s                                   # glide fast → slow
press_c = -(22 + 20 * s)                               # moderate → heavy press
# the TRUE time-varying cost that this behaviour optimises (wstar_smooth story)
W_eff = 0.05 * np.ones(T); W_force = 0.0 + 2.0 * s; W_prog = 2.0 - 1.7 * s

d = mujoco.MjData(m)
for _ in range(80):
    d.ctrl[0] = 0.0; d.ctrl[1] = press_c[0]; mujoco.mj_step(m, d)
Q, fN, PEN = [], [], []
for t in range(T):
    d.ctrl[0] = push_c[t]; d.ctrl[1] = press_c[t]; mujoco.mj_step(m, d)
    n = 0.0; pen = 0.0
    for ci in range(d.ncon):
        c = d.contact[ci]
        if {c.geom1, c.geom2} == {gf, gl}:
            fb = np.zeros(6); mujoco.mj_contactForce(m, d, ci, fb); n += max(0., fb[0]); pen = max(pen, -float(c.dist))
    Q.append(d.qpos.copy()); fN.append(n); PEN.append(pen)
Q = np.array(Q); fN = np.array(fN); PEN = np.array(PEN)
CUT = int((fN > 1e-6).sum()) if (fN <= 1e-6).any() else T
CUT = min(CUT, T - 4)
Q, fN, PEN = Q[:CUT], fN[:CUT], PEN[:CUT] * 1000.0
tt = (np.arange(T) * dt)[:CUT]
W_eff, W_force, W_prog = W_eff[:CUT], W_force[:CUT], W_prog[:CUT]
fF = MU * fN; X = Q[:, 0]; Z = 0.05 + Q[:, 1]
print(f'[tv-anim] travel={X.max()-X.min():.2f}m  N {fN.min():.0f}-{fN.max():.0f}N  step|df|={np.abs(np.diff(fN)).mean():.3f}N  frames={CUT}')

ren = mujoco.Renderer(m, height=360, width=480)
cam = mujoco.MjvCamera(); cam.azimuth = 90; cam.elevation = -14; cam.distance = 0.82
cam.lookat[:] = [0.5 * (X.min() + X.max()), 0.0, 0.04]
OK_B, OK_N, OK_F, OK_P = '#0072B2', '#D55E00', '#009E73', '#CC79A7'; HB = 0.05
os.makedirs('docs/assets/figures/toy_box', exist_ok=True)

frames = []
for t in range(0, CUT, 4):
    d.qpos[:] = Q[t]; d.qvel[:] = 0.0; mujoco.mj_forward(m, d)
    ren.update_scene(d, cam); img = ren.render()
    fig = plt.figure(figsize=(11.2, 8.6))
    gs = fig.add_gridspec(3, 2, height_ratios=[2.35, 1.0, 1.0], hspace=0.6, wspace=0.16)
    a_mj = fig.add_subplot(gs[0, 0]); a_sc = fig.add_subplot(gs[0, 1])
    a_f = fig.add_subplot(gs[1, :]); a_w = fig.add_subplot(gs[2, :])
    a_mj.imshow(img); a_mj.axis('off'); a_mj.set_title('MuJoCo playback', fontsize=11)
    # side view
    xr = (X.min() - 0.08, X.max() + 0.1)
    a_sc.axhline(0.0, color='#555', lw=2); a_sc.fill_between(xr, -0.06, 0.0, color='#ececec')
    a_sc.add_patch(Rectangle((X[t] - HB, Z[t] - HB), 2 * HB, 2 * HB, fc=OK_B, ec='#03395e', lw=2, alpha=.9))
    cx, cy = X[t], Z[t] - HB
    a_sc.plot([cx], [cy], 'o', ms=9, color=OK_N, zorder=5); a_sc.plot(X[:t + 1], Z[:t + 1] - HB, ':', color='#aaa', lw=1)
    a_sc.annotate('', xy=(cx, cy), xytext=(cx, cy + 0.006 + fN[t] / 6000.0), arrowprops=dict(arrowstyle='-|>', color=OK_N, lw=3))
    a_sc.annotate('', xy=(cx - 0.004 - fF[t] / 6000.0, cy), xytext=(cx, cy), arrowprops=dict(arrowstyle='-|>', color=OK_F, lw=3))
    a_sc.text(0.5, 0.90, f'into surface: {PEN[t]:.2f} mm', transform=a_sc.transAxes, ha='center', color=OK_P, fontsize=11, weight='bold')
    phase = 'GLIDE (progress-driven)' if s[:CUT][t] < 0.5 else 'PRESS (force-driven)'
    a_sc.text(0.5, 0.03, phase, transform=a_sc.transAxes, ha='center', color='#333', fontsize=10, weight='bold')
    a_sc.set_xlim(*xr); a_sc.set_ylim(-0.06, 0.24); a_sc.set_aspect('equal')
    a_sc.set_title('side view — single contact point', fontsize=11); a_sc.set_xlabel('slide direction →  x (m)'); a_sc.set_yticks([])
    # force
    a_f.plot(tt[:t + 1], fN[:t + 1], color=OK_N, lw=2.3, label='press (normal)')
    a_f.plot(tt[:t + 1], fF[:t + 1], color=OK_F, lw=2.1, label='friction = μ·N')
    a_f.plot([tt[t]], [fN[t]], 'o', color=OK_N, ms=5)
    a_f.set_xlim(0, tt[-1]); a_f.set_ylim(0, fN.max() * 1.25); a_f.set_ylabel('force (N)')
    a_f.grid(alpha=.3); a_f.legend(loc='upper left', fontsize=8, ncol=2); a_f.set_title('emergent contact force', fontsize=10)
    # W(t) — the time-varying cost
    a_w.plot(tt[:t + 1], W_prog[:t + 1], color='#009E73', lw=2.3, label='progress (pace)')
    a_w.plot(tt[:t + 1], W_force[:t + 1], color=OK_N, lw=2.3, label='force (press)')
    a_w.plot(tt[:t + 1], W_eff[:t + 1], color='#333', lw=2.1, label='effort')
    a_w.set_xlim(0, tt[-1]); a_w.set_ylim(-0.1, 2.2); a_w.set_xlabel('time (s)'); a_w.set_ylabel('cost weight')
    a_w.grid(alpha=.3); a_w.legend(loc='center left', fontsize=8, ncol=3); a_w.set_title('time-varying cost  W(t)', fontsize=10)
    fig.canvas.draw()
    fr = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3]
    frames.append(fr); plt.close(fig)
frames += [frames[-1]] * 8
imageio.mimsave('docs/assets/figures/toy_box/toy_box_tv.gif', frames, fps=18, loop=0)
print('[tv-anim] saved docs/assets/figures/toy_box/toy_box_tv.gif  (%d frames)' % len(frames))
