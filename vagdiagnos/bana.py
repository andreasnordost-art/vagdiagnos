"""Radier ur mätbilens GNSS-bana: kurvaturmodellen anpassas till koordinaterna.

Motorn bestämmer kurvans form (övergångar, cirkelbågar och var de börjar och slutar) ur PMS
kurvatur, som mäts med gyro. Här behålls den formen, men varje cirkelbågas kurvatur bestäms om så
att den integrerade banan (riktning och läge) följer mätbilens GNSS-koordinater genom hela kurvan
och 100 m raklinje på var sida. Det är en oberoende skattning som utnyttjar alla punkter i kurvan,
också övergångarna, och inte bara de få punkterna i en kort cirkelbåge. För sammansatta kurvor
bestäms alla bågar samtidigt, så den totala vinkeländringen fördelas rätt mellan dem.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

MARGINAL = 100.0   # m raklinje före och efter kurvan som binder riktningen
DS = 1.0           # m integrationssteg


def kurvgrupper(hor: pd.DataFrame):
    """Dela kurvtabellen i grupper: en enkel kurva eller alla delar i en sammansatt kurva."""
    grupper, akt = [], []
    for i, c in hor.sort_values("start").iterrows():
        fl = str(c.get("flaggor", "") or "")
        if "del " in fl and akt and "del 1 av" not in fl:
            akt.append(i)
        else:
            if akt:
                grupper.append(akt)
            akt = [i]
    if akt:
        grupper.append(akt)
    return grupper


def _knutar(rader, kc):
    """Knutpunkter (s, k) för gruppens kurvaturprofil; k i 1/m."""
    s = [rader[0].start]; k = [0.0]
    for j, c in enumerate(rader):
        s += [c.karna_start, c.karna_slut]; k += [kc[j], kc[j]]
    s.append(rader[-1].slut); k.append(0.0)
    s = np.maximum.accumulate(np.asarray(s, float))
    return s, np.asarray(k, float)


def _passa(rader, ref_s, ref_x, ref_y, a, b, kc_init, fria):
    """Anpassa kurvaturen för bågarna i `fria` (övriga hålls fasta) till GNSS-punkterna i [a, b].

    Returnerar (kc, sigma_kc för fria, rms) eller None."""
    sel = (ref_s >= a) & (ref_s <= b) & np.isfinite(ref_x) & np.isfinite(ref_y)
    if sel.sum() < len(fria) + 6:
        return None
    ps, px, py = ref_s[sel], ref_x[sel], ref_y[sel]
    if np.max(np.diff(ps)) > 150:
        return None                      # lucka i GNSS-punkterna
    # startriktning och läge: linje genom de första punkterna
    f0 = np.arange(len(ps)) < max(3, int((ps < rader[0].start).sum()))
    kx, ky = np.polyfit(ps[f0], px[f0], 1), np.polyfit(ps[f0], py[f0], 1)
    th0 = float(np.arctan2(ky[0], kx[0])); x00 = float(np.polyval(kx, ps[0])); y00 = float(np.polyval(ky, ps[0]))
    ss = np.arange(ps[0] - 40, ps[-1] + 40 + DS, DS)
    kc_init = np.asarray(kc_init, float); fria = list(fria)

    def bana(p, kc):
        th, x0, y0, e, dl = p[0], p[1], p[2], p[3], p[4]
        ks, kk = _knutar(rader, kc)
        k = np.interp(ss, ks, kk, left=0.0, right=0.0)
        d = DS * (1 + e)                  # skillnad mellan PMS löpande längd och verklig båglängd
        # PMS: positiv kurvatur = vänsterkurva = moturs i (E, N)
        theta = th + np.r_[0, np.cumsum((k[1:] + k[:-1]) / 2 * d)]
        x = x0 + np.r_[0, np.cumsum(np.cos((theta[1:] + theta[:-1]) / 2) * d)]
        y = y0 + np.r_[0, np.cumsum(np.sin((theta[1:] + theta[:-1]) / 2) * d)]
        return np.interp(ps + dl, ss, x), np.interp(ps + dl, ss, y)   # dl: koordinaternas läge längs vägen

    def kc_av(q):
        kc = kc_init.copy(); kc[fria] = q; return kc

    def res(p):
        x, y = bana(p[:5], kc_av(p[5:]))
        return np.r_[x - px, y - py]

    bas_lo = [-np.inf, -np.inf, -np.inf, -0.03, -30]
    try:
        # steg 1: läge, riktning och längdskala med motorns kurvatur; steg 2: också kurvaturen fri
        r1 = least_squares(lambda q: res(np.r_[q, kc_init[fria]]), np.r_[th0, x00, y00, 0.0, 0.0],
                           x_scale=[0.01, 1, 1, 0.001, 5], bounds=(bas_lo, [-v for v in bas_lo]))
        lo = np.r_[bas_lo, np.full(len(fria), -np.inf)]
        r = least_squares(res, np.r_[r1.x, kc_init[fria]], x_scale=np.r_[0.01, 1, 1, 0.001, 5, np.abs(kc_init[fria]) + 1e-5],
                          bounds=(lo, -lo))
    except Exception:
        return None
    rms = float(np.sqrt(np.mean(r.fun ** 2)))
    try:
        J = r.jac; dof = max(len(r.fun) - len(r.x), 1)
        cov = np.linalg.pinv(J.T @ J) * (np.sum(r.fun ** 2) / dof)
        sk = np.sqrt(np.clip(np.diag(cov)[5:], 0, None))
    except Exception:
        sk = np.full(len(fria), np.nan)
    return kc_av(r.x[5:]), sk, rms


def anpassa_grupp(rader, ref_s, ref_x, ref_y):
    """Anpassa bågarnas kurvatur till GNSS-punkterna. Returnerar lista med (R, sigma_rel, rms) per båge.

    Först anpassas hela gruppen samtidigt. Följer banan inte punkterna (rms över 1 m), vilket händer
    när motorns kurvform är fel någonstans i en lång sammansatt kurva, anpassas varje båge för sig
    inom sin egen del av kurvan, med grannbågarna fasta."""
    n = len(rader)
    kc0 = np.array([c.tecken / c.radie for c in rader])
    g = _passa(rader, ref_s, ref_x, ref_y, rader[0].start - MARGINAL, rader[-1].slut + MARGINAL, kc0, range(n))
    res = []
    if g is not None:
        kc, sk, rms = g
        res = [(kc[j], sk[j], rms) for j in range(n)]
    if n > 1 and (g is None or g[2] > 1.0):
        kc_bas = g[0] if g is not None else kc0
        for j, c in enumerate(rader):
            lok = _passa(rader, ref_s, ref_x, ref_y, c.start - 40, c.slut + 40, kc_bas, [j])
            if lok is not None and (not res or lok[2] < res[j][2]):
                if not res:
                    res = [(np.nan, np.nan, np.inf)] * n
                res[j] = (lok[0][j], lok[1][0], lok[2])
    if not res:
        return None
    ut = []
    for j, c in enumerate(rader):
        kc, s_kc, rms = res[j]
        if not np.isfinite(kc) or np.sign(kc) != c.tecken or abs(kc) < 1e-6:
            ut.append((np.nan, np.nan, rms)); continue
        ut.append((float(1 / abs(kc)), float(s_kc / abs(kc)), rms))
    return ut


def radier_ur_bana(res) -> pd.DataFrame:
    """Kör anpassningen för alla kurvor i ett Vagresultat. Kolumner: kurva_nr, R_bana, rel_bana, rms_bana."""
    hor = res.horisontal
    ref = res.ref
    rs, rx, ry = np.asarray(ref.s, float), np.asarray(ref.x, float), np.asarray(ref.y, float)
    rows = []
    for g in kurvgrupper(hor):
        rader = [hor.loc[i] for i in g]
        f = anpassa_grupp(rader, rs, rx, ry)
        for c, v in zip(rader, f or [(np.nan, np.nan, np.nan)] * len(rader)):
            rows.append(dict(kurva_nr=int(c.kurva_nr), R_bana=v[0], rel_bana=v[1], rms_bana=v[2], n_grupp=len(rader)))
    return pd.DataFrame(rows)
