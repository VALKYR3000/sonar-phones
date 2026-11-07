#!/usr/bin/env python3
"""Sonar con varios celulares (BeepBeep). Uso: python3 server.py [--interval 10]"""
import argparse, asyncio, base64, json, os, ssl, subprocess, sys, time
import numpy as np
from aiohttp import web, WSMsgType
from scipy.signal import butter, sosfiltfilt, fftconvolve, hilbert, find_peaks

HERE = os.path.dirname(os.path.abspath(__file__))
F0, F1, CHIRP_S = 10000.0, 14000.0, 0.05
OFFS = [0.0, 0.1, 0.4, 0.9, 1.1]   # regla de Golomb: tren de 5 chirps sin repeticiones de intervalo
GAP = 2.0          # segundos entre turnos de emision
WIN = 0.95         # ventana de busqueda +-s de los chirps de otros (latencias distintas entre celulares)
OWN_WIN = 1.6      # ventana del chirp propio (la latencia de salida llega a ~1 s)
MASK = 1.25        # s alrededor del tren propio que se ignoran al buscar a los otros
MAX_D = 20.0       # distancias mayores se descartan
MIN_SNR = 35.0     # pico del tren / mediana (el ruido llega a ~20; las senales reales a >60)
GXY, GXY_PLANAR, MARGIN, ZMARGIN, MAXPTS = 72, 150, 3.5, 1.5, 9000
POW = 1.8          # contraste del mapa
C = 343.0          # m/s (ajustable con --temp)
DEBUG = False
BLOCK_PTS = 5000

nodes = {}         # ws -> {"name":..}
state = {"last": None, "busy": False, "round": 0, "pending": {}, "lhist": {}, "block": None, "hist": {},
         "refpos": {}, "grid": None, "map": None}


def chirp(fs):
    t = np.arange(int(CHIRP_S * fs)) / fs
    k = (F1 - F0) / CHIRP_S
    return np.sin(2 * np.pi * (F0 * t + 0.5 * k * t * t)) * np.hanning(len(t))


def envelope(rec, fs):
    sos = butter(6, 8000, "highpass", fs=fs, output="sos")
    x = sosfiltfilt(sos, rec)
    cor = fftconvolve(x, chirp(fs)[::-1], mode="same")
    return np.abs(hilbert(cor))


def train_score(env, fs):
    """Suma la envolvente en los desfases del tren: s[t] = sum_r env[t + o_r]."""
    s = np.zeros_like(env)
    for o in OFFS:
        n = int(round(o * fs))
        s[:len(s) - n] += env[n:]
    return s


def peak(s, med, center, half, fs, min_snr=0.0):
    """Primer pico fuerte de la ventana (el sonido directo llega antes que los ecos)."""
    a, b = max(0, int((center - half) * fs)), min(len(s), int((center + half) * fs))
    if b - a < 10:
        return None
    seg = s[a:b]
    mx = float(seg.max())
    if mx < min_snr * med:
        return None
    i = int(np.argmax(seg > 0.6 * mx))
    j = min(len(seg), i + int(0.002 * fs))
    return a + i + int(np.argmax(seg[i:j]))


def mds(D, dim=2):
    n = len(D)
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    w, v = np.linalg.eigh(B)
    o = np.argsort(w)[::-1][:dim]
    return v[:, o] * np.sqrt(np.maximum(w[o], 0))


def align(P, R):
    """Procrustes ortogonal (con reflexion): lleva P al marco de R."""
    Pc, Rc = P - P.mean(0), R - R.mean(0)
    U, _, Vt = np.linalg.svd(Pc.T @ Rc)
    return Pc @ (U @ Vt) + R.mean(0)


def embed(D, dim=3):
    """Posiciones desde distancias con huecos: relleno por camino minimo + SMACOF ponderado."""
    n = len(D)
    W = ~np.isnan(D)
    np.fill_diagonal(W, False)
    F = np.where(np.isnan(D), np.inf, D)
    for k in range(n):
        F = np.minimum(F, F[:, [k]] + F[[k], :])
    if np.isinf(F).any():
        return None
    X = mds(F, dim)
    if X.shape[1] < dim:
        X = np.pad(X, ((0, 0), (0, dim - X.shape[1])))
    w = W.astype(float)
    d0 = np.where(W, D, 0.0)
    V = np.diag(w.sum(1)) - w
    Vp = np.linalg.pinv(V)
    for _ in range(300):
        dist = np.linalg.norm(X[:, None] - X[None], axis=2)
        B = -w * d0 / np.maximum(dist, 1e-9)
        np.fill_diagonal(B, 0)
        np.fill_diagonal(B, -B.sum(1))
        X = Vp @ B @ X
    return X


def align_from(P, idxs, R):
    """Procrustes usando solo los nodos en idxs (comunes con la referencia R); aplica a todos."""
    Pm, Rm = P[idxs].mean(0), R.mean(0)
    U, _, Vt = np.linalg.svd((P[idxs] - Pm).T @ (R - Rm))
    return (P - Pm) @ (U @ Vt) + Rm


def make_grid(ref, planar=False):
    lo, hi = ref.min(0), ref.max(0)
    c = (lo + hi) / 2
    half = max((hi - lo)[:2].max() / 2 + MARGIN, MARGIN + 1)
    if planar:      # celulares a la misma altura: mapa plano (una capa) de mayor resolucion
        res = 2 * half / GXY_PLANAR
        return {"x0": float(c[0] - half), "y0": float(c[1] - half), "z0": float(-res / 2), "res": float(res),
                "nx": GXY_PLANAR, "ny": GXY_PLANAR, "nz": 1}
    res = 2 * half / GXY
    z0, z1 = lo[2] - ZMARGIN, hi[2] + ZMARGIN
    return {"x0": float(c[0] - half), "y0": float(c[1] - half), "z0": float(z0), "res": float(res),
            "nx": GXY, "ny": GXY, "nz": int(max(8, np.ceil((z1 - z0) / res)))}


def backproject(pos, order, idx, S, FS):
    g = state["grid"]
    xs = g["x0"] + (np.arange(g["nx"]) + .5) * g["res"]
    ys = g["y0"] + (np.arange(g["ny"]) + .5) * g["res"]
    zs = g["z0"] + (np.arange(g["nz"]) + .5) * g["res"]
    PX, PY, PZ = [a.astype(np.float32) for a in np.meshgrid(xs, ys, zs, indexing="ij")]
    R = [np.sqrt((PX - p[0]) ** 2 + (PY - p[1]) ** 2 + (PZ - p[2]) ** 2) for p in pos]
    acc = np.zeros(PX.shape, np.float32)
    for a, x in enumerate(order):          # receptor
        s, med = S[x]
        fs = FS[x]
        for b, y in enumerate(order):      # emisor
            t0 = idx[x].get(y)
            if t0 is None:
                continue
            dxy = 0.0 if a == b else float(np.linalg.norm(pos[a] - pos[b]))
            ex = R[a] + R[b] - dxy
            samp = (t0 + ex / C * fs).astype(np.int64)
            ok = (ex > 0.25) & (samp < len(s))
            v = np.zeros(PX.shape, np.float32)
            v[ok] = s[samp[ok]] / med
            acc += np.log1p(np.maximum(v - 2.0, 0))
    return acc


def pack_points(m):
    """Voxels fuertes -> bytes [i,j,k,valor] (los MAXPTS mas intensos)."""
    ijk = np.argwhere(m > 0.10)
    if len(ijk) == 0:
        return b"", 0
    val = m[tuple(ijk.T)]
    top = np.argsort(-val)[:MAXPTS]
    arr = np.concatenate([ijk[top], (val[top] * 255)[:, None]], axis=1).astype(np.uint8)
    return arr.tobytes(), len(top)


def link_attenuation(order, idx, S, D):
    """Atenuacion (dB) del sonido directo en cada enlace respecto a su nivel habitual (sala libre)."""
    out = {}
    for i, a in enumerate(order):
        for j in range(i + 1, len(order)):
            b = order[j]
            d = D[i, j]
            if np.isnan(d) or d < 0.2:
                continue
            ats, calib = [], False
            for x, y in ((a, b), (b, a)):
                t1 = idx[y].get(y)
                if t1 is None:
                    continue
                t0 = idx[x].get(y)
                h = state["lhist"].setdefault((x, y), [])
                if t0 is None:           # no se oye el directo: probablemente bloqueado
                    lng = None
                else:
                    lng = float(np.log(S[x][0][t0] / S[y][0][t1] * d))
                if len(h) >= 8:
                    base = float(np.percentile(h, 80))
                    ats.append(15.0 if lng is None else float(np.clip(8.686 * (base - lng), 0, 20)))
                else:
                    calib = True
                if lng is not None:
                    h.append(lng)
                    del h[:-40]
            out[(i, j)] = {"att": float(np.mean(ats)) if ats else 0.0, "calib": calib or not ats}
    return out


def block_map(pos, links):
    """Tomografia: cada enlace bloqueado ilumina la franja entre sus dos celulares (elipse delgada)."""
    g = state["grid"]
    xs = g["x0"] + (np.arange(g["nx"]) + .5) * g["res"]
    ys = g["y0"] + (np.arange(g["ny"]) + .5) * g["res"]
    zs = g["z0"] + (np.arange(g["nz"]) + .5) * g["res"]
    PX, PY, PZ = [a.astype(np.float32) for a in np.meshgrid(xs, ys, zs, indexing="ij")]
    R = [np.sqrt((PX - p[0]) ** 2 + (PY - p[1]) ** 2 + (PZ - p[2]) ** 2) for p in pos]
    num = np.zeros(PX.shape, np.float32)
    den = np.zeros(PX.shape, np.float32)
    for (i, j), L in links.items():
        d = float(np.linalg.norm(pos[i] - pos[j]))
        ex = R[i] + R[j] - d
        w = np.exp(-0.5 * (ex / 0.15) ** 2)
        num += w * L["att"]
        den += w
    return np.where(den > 0.3, num / np.maximum(den, 1e-6), 0) / 12.0


def pack_block(m):
    ijk = np.argwhere(m > 0.15)
    if len(ijk) == 0:
        return b"", 0
    val = np.clip(m[tuple(ijk.T)], 0, 1)
    top = np.argsort(-val)[:BLOCK_PTS]
    return np.concatenate([ijk[top], (val[top] * 255)[:, None]], axis=1).astype(np.uint8).tobytes(), len(top)


def process(recs, order, slots):
    """recs: {name: (fs, float32 array)}; order: nombres que respondieron; slots: turno de cada nombre."""
    corto = [x for x in order if len(recs[x][1]) < ((slots[x] + 1) * GAP + 1.5) * recs[x][0]]
    if corto:
        print("grabacion demasiado corta, se ignora:", corto, flush=True)
    order = [x for x in order if x not in corto]
    n = len(order)
    if n < 2:
        return {"t": time.time(), "nodes": order, "dist_m": [], "pos_m": None, "echo_dist_m": {}, "map": None}
    idx, S, FS = {}, {}, {}
    for x in order:
        fs, rec = recs[x]
        env = envelope(rec, fs)
        sc = train_score(env, fs)
        med = float(np.median(sc))
        S[x], FS[x] = (sc, med), fs
        c0 = (slots[x] + 1) * GAP
        a0, b0 = max(0, int((c0 - OWN_WIN) * fs)), min(len(sc), int((c0 + OWN_WIN) * fs))
        own = a0 + int(np.argmax(sc[a0:b0]))     # el propio es siempre el mas fuerte (parlante pegado al mic)
        idx[x] = {x: own}
        scm = sc.copy()                          # sin los residuos del tren propio
        scm[max(0, own - int(MASK * fs)):own + int(MASK * fs)] = 0
        for y in order:
            if y != x:
                idx[x][y] = peak(scm, med, own / fs + (slots[y] - slots[x]) * GAP, WIN, fs, MIN_SNR)
    # distancias BeepBeep, con mediana de las ultimas 9 rondas (por nombre)
    for i, a in enumerate(order):
        for j in range(i + 1, n):
            b = order[j]
            try:
                v = abs((idx[a][b] - idx[a][a]) / FS[a] + (idx[b][a] - idx[b][b]) / FS[b]) * C / 2
                if v <= MAX_D:
                    k = tuple(sorted((a, b)))
                    state["hist"][k] = (state["hist"].get(k, []) + [v])[-9:]
            except (TypeError, KeyError):
                pass
    D = np.full((n, n), np.nan)
    for i in range(n):
        D[i, i] = 0
        for j in range(i + 1, n):
            h = state["hist"].get(tuple(sorted((order[i], order[j]))))
            if h:
                D[i, j] = D[j, i] = float(np.median(h))
    pos = None
    if n == 2 and not np.isnan(D[0, 1]):
        pos = np.array([[0.0, 0.0, 0.0], [D[0, 1], 0.0, 0.0]])
    elif n >= 3 and (n >= 4 or not np.isnan(D).any()):   # con 3 hacen falta las 3 distancias
        pos = embed(D, 3)
        if pos is not None and pos[:, 2].std() < 0.12:   # casi coplanares: z no es fiable
            pos[:, 2] = 0.0
    links = link_attenuation(order, idx, S, D)
    out = {"t": time.time(), "nodes": order, "dist_m": np.round(D, 2).tolist(), "pos_m": None,
           "echo_dist_m": {}, "map": None, "block": None,
           "links": [{"a": order[i], "b": order[j], "att": round(L["att"], 1), "calib": L["calib"]}
                     for (i, j), L in links.items()]}
    if pos is not None:
        planar = bool(np.all(pos[:, 2] == 0))
        ref = state["refpos"]
        common = [i for i, x in enumerate(order) if x in ref]
        fresh = not ref or len(common) < 2 or state.get("planar") != planar
        if not fresh:
            al = align_from(pos, common, np.array([ref[order[i]] for i in common]))
            if np.linalg.norm(al[common] - np.array([ref[order[i]] for i in common]), axis=1).mean() > 0.4:
                fresh = True                  # la geometria cambio (distancia corregida o celular movido)
            else:
                pos = al
                for i, x in enumerate(order):     # nodos nuevos se integran al marco existente
                    ref.setdefault(x, pos[i].copy())
        if fresh:
            state.update(refpos={x: pos[i].copy() for i, x in enumerate(order)}, grid=make_grid(pos, planar),
                         map=None, block=None, planar=planar)
        out["pos_m"] = pos.round(2).tolist()
        acc = backproject(pos, order, idx, S, FS)
        acc = np.clip(acc / (np.percentile(acc, 99.7) + 1e-9), 0, 1) ** POW
        state["map"] = acc if state["map"] is None else 0.65 * state["map"] + 0.35 * acc
        bm = block_map(pos, links)
        state["block"] = bm if state["block"] is None or state["block"].shape != bm.shape else 0.5 * state["block"] + 0.5 * bm
        rawb, kb = pack_block(state["block"])
        out["block"] = {"n": kb, "b64": base64.b64encode(rawb).decode()}
        raw, k = pack_points(state["map"])
        out["map"] = dict(state["grid"], n=k, b64=base64.b64encode(raw).decode(),
                          planar=planar)
    for x in order:
        sc, med = S[x]
        p0 = idx[x][x]
        if p0 is None:
            continue
        fs = FS[x]
        a, b = p0 + int(0.0008 * fs), p0 + int(0.045 * fs)
        pk, _ = find_peaks(sc[a:b], height=max(0.15 * sc[p0], 6 * med), distance=int(0.0005 * fs))
        out["echo_dist_m"][x] = [round(float((p + a - p0) / fs * C / 2), 2) for p in pk][:8]
    return out


async def measure():
    if state["busy"]:
        return {"error": "ocupado"}
    order = [ws for ws in nodes if not ws.closed]
    if len(order) < 2:
        return {"error": "se necesitan >=2 celulares conectados"}
    state["busy"] = True
    try:
        names = [nodes[ws]["name"] for ws in order]
        slots = {nm: k for k, nm in enumerate(names)}
        loop = asyncio.get_running_loop()
        state["round"] += 1
        state["pending"] = {"want": set(names), "recs": {}, "fut": loop.create_future(), "round": state["round"]}
        dur = (len(order) + 1) * GAP + OFFS[-1] + CHIRP_S + 1.5
        for k, ws in enumerate(order):
            await ws.send_json({"type": "go", "slot": k, "gap": GAP, "dur": dur,
                                "f0": F0, "f1": F1, "chirp": CHIRP_S, "offs": OFFS, "round": state["round"]})
        try:
            await asyncio.wait_for(state["pending"]["fut"], timeout=dur + 8)
        except asyncio.TimeoutError:
            pass
        recs = dict(state["pending"]["recs"])
        got = [nm for nm in names if nm in recs]
        if len(got) < len(names):
            print("sin respuesta:", [nm for nm in names if nm not in recs], flush=True)
        if len(got) < 2:
            return {"error": "menos de 2 nodos respondieron", "sin_respuesta": [nm for nm in names if nm not in recs]}
        DEBUG and np.savez(os.path.join(HERE, "last_rec.npz"), **{k: v[1] for k, v in recs.items()}, fs=list(recs.values())[0][0])
        res = await loop.run_in_executor(None, process, recs, got, slots)
        state["last"] = res
        return res
    except Exception as e:
        print("error en ronda:", repr(e), flush=True)
        return {"error": repr(e)}
    finally:
        state["busy"] = False


async def ws_handler(req):
    ws = web.WebSocketResponse(max_msg_size=64 * 1024 * 1024, heartbeat=15)
    await ws.prepare(req)
    nodes[ws] = {"name": f"cel{len(nodes)+1}-{req.remote}", "meta": None}
    ua = req.headers.get("User-Agent", "")
    nodes[ws]["ua"] = ua
    print("conectado", nodes[ws]["name"], "total", len(nodes), "|", ua[ua.find("(") + 1: ua.find(")")], flush=True)
    try:
        async for m in ws:
            p = state["pending"]
            if m.type == WSMsgType.TEXT:
                d = json.loads(m.data)
                if d.get("type") == "meta":
                    nodes[ws]["meta"] = d
                elif d.get("type") in ("ack", "err"):
                    print("nodo", nodes[ws]["name"], d, flush=True)
                elif d.get("type") == "hello" and d.get("cid"):
                    nodes[ws]["cid"] = d["cid"]
                    for w2 in [w for w, v in nodes.items() if w is not ws and v.get("cid") == d["cid"]]:
                        await w2.close()          # misma pagina conectada dos veces: queda la nueva
                    if d.get("name"):
                        base, nm, i = d["name"], d["name"], 1
                        used = {v["name"] for k, v in nodes.items() if k is not ws}
                        while nm in used:
                            i += 1
                            nm = f"{base}{i}"
                        nodes[ws]["name"] = nm
                elif d.get("type") == "hello" and d.get("name"):
                    base, nm, i = d["name"], d["name"], 1
                    used = {v["name"] for k, v in nodes.items() if k is not ws}
                    while nm in used:
                        i += 1
                        nm = f"{base}{i}"
                    nodes[ws]["name"] = nm
            elif m.type == WSMsgType.BINARY and p and nodes[ws]["meta"] and nodes[ws]["meta"].get("round") == p.get("round"):
                nm = nodes[ws]["name"]
                fs = nodes[ws]["meta"]["fs"]
                p["recs"][nm] = (fs, np.frombuffer(m.data, dtype=np.float32))
                if p["want"] <= set(p["recs"]) and not p["fut"].done():
                    p["fut"].set_result(True)
    finally:
        nodes.pop(ws, None)
        print("desconectado; total", len(nodes))
    return ws


async def api_measure(req):
    return web.json_response(await measure())


async def api_calibrate(req):
    state["lhist"] = {}
    state["block"] = None
    return web.json_response({"ok": True, "msg": "recalibrando: deja la sala libre unas 8 rondas"})


async def api_last(req):
    return web.json_response(state["last"] or {})


async def loop_task(interval):
    while True:
        await asyncio.sleep(interval)
        r = await measure()
        print(json.dumps(r) if "error" in r else f"ok {r['nodes']} dist={r['dist_m']}")


def ensure_cert(ip):
    cert, key = os.path.join(HERE, "cert.pem"), os.path.join(HERE, "key.pem")
    if not os.path.exists(cert):
        subprocess.check_call(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "365",
                               "-keyout", key, "-out", cert, "-subj", "/CN=sonar",
                               "-addext", f"subjectAltName=IP:{ip},DNS:localhost"],
                              stderr=subprocess.DEVNULL)
    return cert, key


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8443)
    ap.add_argument("--interval", type=float, default=0, help="medir cada N s (0 = manual)")
    ap.add_argument("--debug", action="store_true", help="guarda last_rec.npz")
    ap.add_argument("--temp", type=float, default=20, help="temperatura C")
    a = ap.parse_args()
    global C, DEBUG
    DEBUG = a.debug
    C = 331.3 + 0.606 * a.temp
    ip = subprocess.check_output("hostname -I", shell=True, text=True).split()[0]
    cert, key = ensure_cert(ip)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert, key)
    app = web.Application()
    app.add_routes([web.get("/", lambda r: web.FileResponse(os.path.join(HERE, "static/index.html"), headers={"Cache-Control": "no-store"})),
                    web.get("/ws", ws_handler),
                    web.get("/radar", lambda r: web.FileResponse(os.path.join(HERE, "static/radar.html"))),
                    web.get("/api/nodes", lambda r: web.json_response([v["name"] for v in nodes.values()])), web.get("/api/measure", api_measure),
                    web.get("/api/last", api_last), web.get("/api/calibrate", api_calibrate)])
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "0.0.0.0", a.port, ssl_context=ctx).start()
    print(f"Abre en cada celular: https://{ip}:{a.port}  (acepta la advertencia del certificado)")
    print(f"Medir: curl -k https://localhost:{a.port}/api/measure")
    if a.interval:
        asyncio.create_task(loop_task(a.interval))
    await asyncio.Event().wait()

asyncio.run(main())
