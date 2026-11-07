"""Simulacion 3D: habitacion 6x4.5x2.6 m, 5 celulares a distintas alturas, eco de 6 superficies."""
import sys, numpy as np
sys.argv = ["x"]; exec(open("server.py").read().rsplit("asyncio.run(main())", 1)[0])
fs = 48000; Wr, Hr, Zr = 6.0, 4.5, 2.6
pts = np.array([[1.0, 1.0, 0.3], [5.0, 1.2, 1.8], [5.2, 3.6, 0.9], [0.8, 3.4, 2.0], [3.0, 2.2, 1.2]]); n = len(pts)
rng = np.random.default_rng(3)
c = chirp(fs); L = int(((n + 1) * GAP + OFFS[-1] + CHIRP_S + 1.5) * fs)
def images(p):
    return [np.array([-p[0], p[1], p[2]]), np.array([2*Wr - p[0], p[1], p[2]]), np.array([p[0], -p[1], p[2]]),
            np.array([p[0], 2*Hr - p[1], p[2]]), np.array([p[0], p[1], -p[2]]), np.array([p[0], p[1], 2*Zr - p[2]])]
person = np.array([[3.8, 3.0, z] for z in np.arange(0.2, 1.8, 0.2)])
box = np.array([[x, 0.6, z] for x in np.arange(1.5, 2.6, 0.3) for z in (0.2, 0.6)])
wall = np.array([[5.95, y, z] for y in np.arange(0.5, 4.2, 0.5) for z in np.arange(0.3, 2.4, 0.5)])
SC = np.vstack([person, box, wall])
def run_round():
    off = rng.uniform(0, 0.3, n); lat = 0.45; recs = {}
    for x in range(n):
        r = rng.normal(0, 0.003, L)
        for y in range(n):
            e = (y + 1) * GAP + lat + off[y] - off[x]
            d = np.linalg.norm(pts[x] - pts[y])
            paths = [(d, (1.0 if x == y else 0.04 / (0.3 + d)))] + [(np.linalg.norm(q - pts[y]) + np.linalg.norm(q - pts[x]), 0.012 / (0.3 + np.linalg.norm(q - pts[y])) / (0.3 + np.linalg.norm(q - pts[x]))) for q in SC]
            if x == y: paths[0] = (0.0, 1.0)
            for pl, amp in paths:
                for o in OFFS:
                    st = int((e + o + pl / C) * fs); r[st:st + len(c)] += amp * c
        recs[f"n{x}"] = (fs, r.astype(np.float32))
    return recs
order = [f"n{i}" for i in range(n)]; slots = {x: i for i, x in enumerate(order)}
for _ in range(4):
    res = process(run_round(), order, slots)
true = np.linalg.norm(pts[:, None] - pts[None], axis=2)
print("error max dist (m):", np.nanmax(np.abs(np.array(res["dist_m"]) - true)))
g = res["map"]; print("voxels:", g["n"], "grid", g["nx"], g["ny"], g["nz"], "planar:", g["planar"])
ref = np.array([state["refpos"][x] for x in order]); Pm, Rm = ref.mean(0), pts.mean(0)
best = None
for sgn in (1, -1):   # el eje vertical puede salir invertido
    M = ref.copy(); M[:, 2] *= sgn
    U, _, Vt = np.linalg.svd((M - Pm * [1, 1, 1] * 0 - M.mean(0)).T @ (pts - Rm)); Rot = U @ Vt
    err = np.linalg.norm((M - M.mean(0)) @ Rot + Rm - pts, axis=1).mean()
    if best is None or err < best[0]: best = (err, sgn, Rot, M.mean(0))
print("error posicion 3D (m):", round(float(best[0]), 3))
_, sgn, Rot, Mm = best
raw = np.frombuffer(base64.b64decode(g["b64"]), np.uint8).reshape(-1, 4)
Q = np.stack([g["x0"] + (raw[:, 0] + .5) * g["res"], g["y0"] + (raw[:, 1] + .5) * g["res"], g["z0"] + (raw[:, 2] + .5) * g["res"]], 1)
Q[:, 2] *= sgn; Q = (Q - Mm) @ Rot + Rm
dsc = np.min(np.linalg.norm(Q[:, None] - SC[None], axis=2), axis=1)
dper = np.min(np.linalg.norm(Q[:, None] - person[None], axis=2), axis=1)
top = np.argsort(-raw[:, 3])[:300]
print("top300: a <0.35 m de algun objeto:", round(float((dsc[top] < .35).mean()), 2), " | a <0.35 m de la persona:", round(float((dper[top] < .35).mean()), 2))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
top = np.argsort(-raw[:, 3])[:700]
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), facecolor="k")
for a_, (i, j, t) in zip(ax, [(0, 1, "planta (x,y)"), (0, 2, "frente (x,z)"), (1, 2, "lado (y,z)")]):
    a_.set_facecolor("k"); a_.scatter(Q[top, i], Q[top, j], c=raw[top, 3], s=9, cmap="cool")
    a_.scatter(SC[:, i], SC[:, j], facecolors="none", edgecolors="y", s=40, lw=.6)
    a_.plot(pts[:, i], pts[:, j], "w^"); a_.set_aspect("equal"); a_.set_title(t, color="w")
plt.savefig("/tmp/sim3d.png", dpi=70, facecolor="k")
