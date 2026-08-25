"""Docs animation of the toy. Four synced panels:
  • MuJoCo playback (rendered, rotating camera, lit)   • side view + penetration
  • emergent contact force (press + friction)          • the controls (push/press)
Smooth control → clean single-point force; realistic mm-scale penetration."""
import os
os.environ['MUJOCO_GL'] = 'egl'
import numpy as np, sys, mujoco
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import imageio.v2 as imageio
sys.path.insert(0, 'src')
import toy_box_slide_irl as toy

MU = toy.MU
# stiffer contact (mm penetration, matches the IRL's solref_time=0.1) + heavier
# vertical damping (no bounce) + lights + textured floor for depth.
xml = (toy._XML
       .replace('solref="0.4 1"', 'solref="0.1 1"')
       .replace('damping="150.0"', 'damping="800.0"')
       .replace('<worldbody>', '''<asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.86 0.86 0.9" rgb2="0.78 0.78 0.84" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="10 10" reflectance="0.0"/>
  </asset>
  <worldbody>
    <light name="key" directional="true" pos="0.3 -0.6 1.5" dir="-0.15 0.35 -1" diffuse="0.55 0.55 0.55" specular="0.15 0.15 0.15"/>
    <light name="fill" directional="true" pos="-0.5 0.4 1.2" dir="0.4 -0.3 -1" diffuse="0.35 0.35 0.35"/>''')
       .replace('friction="0.6 0.005 0.0001"/>\n  </worldbody>',
                'friction="0.6 0.005 0.0001" material="grid"/>\n  </worldbody>', 1))
# add material to the floor plane (first geom)
xml = xml.replace('rgba="0.9 0.9 0.9 1"', 'material="grid"')

m = mujoco.MjModel.from_xml_string(xml)
gf = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'foot')
gl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'floor')

T = 280; dt = m.opt.timestep; tt = np.arange(T) * dt
tc = np.linspace(0, 1, T)
press_c = -(16 + 18 * np.sin(np.pi * tc))
push_c = 24 + 4 * np.sin(np.pi * tc)
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
Q = np.array(Q); fN = np.array(fN); PEN = np.array(PEN) * 1000.0
CUT = 268   # stop before the step-271 lift-off/bounce (the press has eased by then)
Q, fN, PEN = Q[:CUT], fN[:CUT], PEN[:CUT]
tt = tt[:CUT]; push_c = push_c[:CUT]; press_c = press_c[:CUT]; T = CUT
fF = MU * fN; X = Q[:, 0]; Z = 0.05 + Q[:, 1]
print(f'[anim] travel={X.max()-X.min():.2f}m  N {fN.min():.0f}-{fN.max():.0f}N  pen {PEN.min():.2f}-{PEN.max():.2f}mm  step|df|={np.abs(np.diff(fN)).mean():.3f}N')

ren = mujoco.Renderer(m, height=360, width=480)
# STATIONARY camera framing the whole slide — the box moves across a fixed view.
cam = mujoco.MjvCamera()
cam.azimuth = 90; cam.elevation = -14; cam.distance = 0.82
cam.lookat[:] = [0.5 * (X.min() + X.max()), 0.0, 0.04]
OK_B, OK_N, OK_F, OK_P = '#0072B2', '#D55E00', '#009E73', '#CC79A7'
HB = 0.05
os.makedirs('docs/assets/figures/toy_box', exist_ok=True)
idxs = list(range(0, T, 4))
frames = []
for k, t in enumerate(idxs):
    d.qpos[:] = Q[t]; d.qvel[:] = 0.0; mujoco.mj_forward(m, d)
    ren.update_scene(d, cam); img = ren.render()

    fig = plt.figure(figsize=(11.2, 8.6))
    gs = fig.add_gridspec(3, 2, height_ratios=[2.35, 1.0, 1.0], hspace=0.6, wspace=0.16)
    a_mj = fig.add_subplot(gs[0, 0]); a_sc = fig.add_subplot(gs[0, 1])
    a_f = fig.add_subplot(gs[1, :]);  a_u = fig.add_subplot(gs[2, :])

    a_mj.imshow(img); a_mj.axis('off'); a_mj.set_title('MuJoCo playback', fontsize=11)

    xr = (X.min() - 0.08, X.max() + 0.1)
    a_sc.axhline(0.0, color='#555', lw=2)
    a_sc.fill_between(xr, -0.06, 0.0, color='#ececec')
    a_sc.add_patch(Rectangle((X[t] - HB, Z[t] - HB), 2 * HB, 2 * HB, fc=OK_B, ec='#03395e', lw=2, alpha=.9))
    cx, cy = X[t], Z[t] - HB
    a_sc.plot([cx], [cy], 'o', ms=9, color=OK_N, zorder=5)
    a_sc.plot(X[:t + 1], Z[:t + 1] - HB, ':', color='#aaa', lw=1)
    a_sc.annotate('', xy=(cx, cy), xytext=(cx, cy + 0.006 + fN[t] / 6000.0),
                  arrowprops=dict(arrowstyle='-|>', color=OK_N, lw=3))
    a_sc.annotate('', xy=(cx - 0.004 - fF[t] / 6000.0, cy), xytext=(cx, cy),
                  arrowprops=dict(arrowstyle='-|>', color=OK_F, lw=3))
    a_sc.text(cx + .006, cy + .034, f'press {fN[t]:.0f} N', color=OK_N, fontsize=9, weight='bold')
    a_sc.text(cx - .10, cy - .03, f'friction {fF[t]:.0f} N', color=OK_F, fontsize=9, weight='bold')
    a_sc.text(0.5, 0.90, f'into surface: {PEN[t]:.2f} mm', transform=a_sc.transAxes,
              ha='center', color=OK_P, fontsize=11, weight='bold')
    a_sc.set_xlim(*xr); a_sc.set_ylim(-0.06, 0.24); a_sc.set_aspect('equal')
    a_sc.set_title('side view — single contact point', fontsize=11)
    a_sc.set_xlabel('slide direction →  x (m)'); a_sc.set_yticks([])

    a_f.plot(tt[:t + 1], fN[:t + 1], color=OK_N, lw=2.3, label='press (normal)')
    a_f.plot(tt[:t + 1], fF[:t + 1], color=OK_F, lw=2.1, label='friction = μ·N')
    a_f.plot([tt[t]], [fN[t]], 'o', color=OK_N, ms=5)
    a_f.set_xlim(0, tt[-1]); a_f.set_ylim(0, fN.max() * 1.25)
    a_f.set_ylabel('force (N)'); a_f.grid(alpha=.3); a_f.legend(loc='upper right', fontsize=8, ncol=2)
    a_f.set_title('emergent contact force', fontsize=10)

    a_u.plot(tt[:t + 1], push_c[:t + 1], color='#333', lw=2.1, label='push (forward)')
    a_u.plot(tt[:t + 1], -press_c[:t + 1], color=OK_P, lw=2.1, label='press (down)')
    a_u.set_xlim(0, tt[-1]); a_u.set_ylim(0, 40)
    a_u.set_xlabel('time (s)'); a_u.set_ylabel('control'); a_u.grid(alpha=.3)
    a_u.legend(loc='upper right', fontsize=8, ncol=2); a_u.set_title('controls', fontsize=10)

    fig.canvas.draw()
    fr = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    fr = fr.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3]
    frames.append(fr); plt.close(fig)

frames += [frames[-1]] * 8
out = 'docs/assets/figures/toy_box/toy_box_slide.gif'
imageio.mimsave(out, frames, fps=18, loop=0)
print(f'[anim] saved {out}  ({len(frames)} frames)')
