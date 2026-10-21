"""Parametriska geometrimodeller anpassade till PMS 20-metersmedelvärden.

PMS redovisar kurvatur och lutning som medelvärde över varje mätintervall.
Modellerna beskriver den underliggande kontinuerliga geometrin och
jämförs med data genom att integreras över exakt samma intervall. Därmed
återvinns cirkelbågens radie även när bågen är kortare än 20 m eller
omges av långa övergångskurvor.

Horisontalt: kurvatur k(s) är trapetsformad
    0 -> (klotoid L1) -> kc (cirkelbåge Lc) -> (klotoid L2) -> 0
Vertikalt: lutning g(s) är en ramp
    g1 -> (vertikalkurva L, linjär lutningsändring) -> g2
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


# ------------------------------------------------------------ hjälpfunktioner
def _int_ramp(s, s0, L):
    """Integral från -inf till s av en enhetsramp 0->1 över [s0, s0+L], därefter 1."""
    s = np.asarray(s, float)
    if L <= 1e-9:
        return np.clip(s - s0, 0, None)
    u = np.clip(s - s0, 0, L)
    return u * u / (2 * L) + np.clip(s - s0 - L, 0, None)


def trapets_int(s, s0, L1, Lc, L2, kc):
    """Integral av trapetskurvatur från -inf till s."""
    s1 = s0 + L1 + Lc
    return kc * (_int_ramp(s, s0, L1) - _int_ramp(s, s1, L2))


def trapets_medel(a, b, p):
    s0, L1, Lc, L2, kc = p
    return (trapets_int(b, s0, L1, Lc, L2, kc) - trapets_int(a, s0, L1, Lc, L2, kc)) / (b - a)


def trapets_varde(s, p):
    s0, L1, Lc, L2, kc = p
    s = np.asarray(s, float)
    up = np.clip((s - s0) / max(L1, 1e-9), 0, 1) if L1 > 1e-9 else (s >= s0).astype(float)
    s1 = s0 + L1 + Lc
    dn = np.clip((s - s1) / max(L2, 1e-9), 0, 1) if L2 > 1e-9 else (s >= s1).astype(float)
    return kc * (up - dn)


def ramp_medel(a, b, p):
    g1, s0, L, dg = p
    return g1 + dg * (_int_ramp(b, s0, L) - _int_ramp(a, s0, L)) / (b - a)


def ramp_varde(s, p):
    g1, s0, L, dg = p
    return g1 + dg * np.clip((np.asarray(s, float) - s0) / max(L, 1e-9), 0, 1)


# ------------------------------------------------------------ anpassningar
def anpassa_trapets(a, b, y, w=None):
    """Anpassa trapetsmodell till intervallmedelvärden y över [a, b].

    Returnerar dict med parametrar, radie, residual-RMS och osäkerhet.
    """
    a = np.asarray(a, float); b = np.asarray(b, float); y = np.asarray(y, float)
    w = np.ones_like(y) if w is None else np.asarray(w, float)
    tecken = np.sign(np.sum(y * (b - a))) or 1.0
    yy = y * tecken
    # Startgissning: kärnan där kurvaturen överstiger halva maximum
    sm = np.convolve(yy, np.ones(3) / 3, mode="same")
    peak = max(sm.max(), 1e-3)
    over = np.where(sm >= 0.5 * peak)[0]
    i0, i1 = over[0], over[-1]
    lo, hi = a.min(), b.max()
    best = None
    for frac in (0.25, 0.5, 0.75):
        Lc0 = max(b[i1] - a[i0], 5.0) * frac
        side = max((b[i1] - a[i0] - Lc0) / 2, 5.0)
        x0 = np.array([a[i0] - side * 0.5, side, Lc0, side, peak * 1.1])
        lb = [lo - 200, 0.0, 0.0, 0.0, 0.0]
        ub = [hi, 1000.0, 3000.0, 1000.0, 2000.0]
        x0 = np.clip(x0, np.array(lb) + 1e-6, np.array(ub) - 1e-6)

        def res(p):
            return (trapets_medel(a, b, p) - yy) * w

        try:
            r = least_squares(res, x0, bounds=(lb, ub), x_scale=[20, 20, 20, 20, max(peak, 1)])
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    p = best.x
    resid = best.fun / np.maximum(w, 1e-9)
    rms = float(np.sqrt(np.mean(resid ** 2)))
    # Kovarians för kc (parameter 4) ur Jacobianen
    sig_kc = np.nan
    try:
        J = best.jac
        dof = max(len(y) - 5, 1)
        cov = np.linalg.pinv(J.T @ J) * (2 * best.cost / dof)
        sig_kc = float(np.sqrt(max(cov[4, 4], 0)))
    except Exception:
        pass
    kc = p[4] * tecken
    return dict(s0=p[0], L1=p[1], Lc=p[2], L2=p[3], kc=kc, tecken=int(tecken),
                start=p[0], slut=p[0] + p[1] + p[2] + p[3],
                karna_start=p[0] + p[1], karna_slut=p[0] + p[1] + p[2],
                radie=10000 / abs(kc) if abs(kc) > 1e-9 else np.inf,
                sigma_kc=sig_kc, rms=rms, n=len(y))


def anpassa_ramp(a, b, y, w=None):
    """Anpassa lutningsramp (vertikalkurva) till intervallmedel av lutning (%)."""
    a = np.asarray(a, float); b = np.asarray(b, float); y = np.asarray(y, float)
    w = np.ones_like(y) if w is None else np.asarray(w, float)
    lo, hi = a.min(), b.max()
    g1 = float(np.mean(y[:2])); g2 = float(np.mean(y[-2:]))
    best = None
    for frac in (0.2, 0.5, 0.8):
        L0 = (hi - lo) * frac
        x0 = np.array([g1, lo + (hi - lo - L0) / 2, L0, g2 - g1])
        lb = [-20, lo - 100, 1.0, -30]; ub = [20, hi, 5000, 30]
        x0 = np.clip(x0, np.array(lb) + 1e-6, np.array(ub) - 1e-6)
        try:
            r = least_squares(lambda p: (ramp_medel(a, b, p) - y) * w, x0, bounds=(lb, ub),
                              x_scale=[1, 20, 20, 1])
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    g1, s0, L, dg = best.x
    resid = best.fun / np.maximum(w, 1e-9)
    rms = float(np.sqrt(np.mean(resid ** 2)))
    sig = dict(L=np.nan, dg=np.nan)
    try:
        J = best.jac; dof = max(len(y) - 4, 1)
        cov = np.linalg.pinv(J.T @ J) * (2 * best.cost / dof)
        sig = dict(L=float(np.sqrt(max(cov[2, 2], 0))), dg=float(np.sqrt(max(cov[3, 3], 0))))
    except Exception:
        pass
    rv = 100 * L / abs(dg) if abs(dg) > 1e-9 else np.inf
    # relativ osäkerhet i Rv = sqrt((sL/L)^2 + (sdg/dg)^2)
    rel = np.sqrt((sig["L"] / max(L, 1e-9)) ** 2 + (sig["dg"] / max(abs(dg), 1e-9)) ** 2)
    return dict(g1=g1, g2=g1 + dg, s0=s0, L=L, dg=dg, start=s0, slut=s0 + L,
                typ="konvex" if dg < 0 else "konkav", radie=rv, rel_sigma=float(rel),
                rms=rms, n=len(y))


# ------------------------------------------------------------ flera cirkelbågar (korgbågar)
def _pl_int(s, t, v):
    """Integral från t[0] till s av styckvis linjär funktion (knutar t, värden v), 0 utanför."""
    s = np.asarray(s, float)[:, None]
    t = np.asarray(t, float); v = np.asarray(v, float)
    a = t[:-1]; L = np.maximum(t[1:] - a, 1e-9)
    va = v[:-1]; dv = v[1:] - va
    u = np.clip(s - a, 0, L)
    return (va * u + dv * u * u / (2 * L)).sum(axis=1)


def _knutar(p, n):
    t = p[0] + np.r_[0, np.cumsum(np.abs(p[1:2 * n + 2]))]
    v = np.zeros(2 * n + 2)
    v[1:-1] = np.repeat(p[2 * n + 2:], 2)
    return t, v


def flerbage_medel(a, b, p, n):
    t, v = _knutar(p, n)
    return (_pl_int(b, t, v) - _pl_int(a, t, v)) / (b - a)


def _startvarden(a, b, sm, n, f, rng):
    lo, hi = a.min(), b.max()
    if f == 0:
        kant = np.linspace(lo, hi, n + 1)[1:-1]
    elif f == 1 and n > 1:
        mid = (a + b) / 2
        hopp = np.abs(np.diff(sm))
        idx = np.sort(np.argsort(-hopp)[:n - 1])
        kant = (mid[idx] + mid[idx + 1]) / 2
    else:
        kant = np.sort(rng.uniform(lo, hi, n - 1))
    delar = np.r_[lo, kant, hi]
    c0 = []
    for i in range(n):
        m = (a >= delar[i]) & (b <= delar[i + 1])
        c0.append(max(np.max(sm[m]) if m.any() else sm.max(), 0.5))
    d0 = np.full(2 * n + 1, (hi - lo) / (3 * n + 1))
    return np.r_[lo, d0, c0]


def anpassa_flerbage(a, b, y, nmax=3, nmin=1, forsok=6, start=None):
    """Anpassa nmin..nmax cirkelbågar med mellanliggande övergångar; välj antal med BIC.

    start: parametervektor från en tidigare anpassning (används då som enda startvärde).
    """
    a = np.asarray(a, float); b = np.asarray(b, float); y = np.asarray(y, float)
    o = np.argsort(a); a, b, y = a[o], b[o], y[o]
    tecken = np.sign(np.sum(y * (b - a))) or 1.0
    yy = y * tecken; N = len(y)
    lo, hi = a.min(), b.max(); span = hi - lo
    sm = np.convolve(yy, np.ones(3) / 3, mode="same")
    resultat = []
    for n in range(nmin, nmax + 1):
        if N < 3 * n + 3:
            break
        lb = np.r_[lo - 100, np.zeros(2 * n + 1), np.zeros(n)]
        ub_d = np.full(2 * n + 1, span + 50.0)
        ub_d[0::2] = min(500.0, span + 50.0)   # övergångar (klotoider) högst 500 m
        ub = np.r_[hi, ub_d, np.full(n, 250.0)]   # R >= 40 m
        starter = [np.asarray(start, float)] if start is not None else \
            [_startvarden(a, b, sm, n, f, np.random.default_rng(100 * n + f)) for f in range(forsok)]
        best = None
        for x0 in starter:
            x0 = np.clip(x0, lb + 1e-6, ub - 1e-6)
            try:
                r = least_squares(lambda p: flerbage_medel(a, b, p, n) - yy, x0, bounds=(lb, ub),
                                  x_scale=np.r_[20, np.full(2 * n + 1, 20), np.full(n, 5)], max_nfev=400)
            except Exception:
                continue
            if best is None or r.cost < best.cost:
                best = r
        if best is None:
            continue
        rss = 2 * best.cost; k = 3 * n + 2
        bic = N * np.log(max(rss / N, 1e-9)) + k * np.log(N)
        resultat.append((bic, n, best))
    if not resultat:
        return None
    resultat.sort(key=lambda x: x[0])
    bic, n, best = resultat[0]
    # Slå ihop: om två intilliggande bågar har nästan samma radie (< 30 % skillnad) väljs
    # närmast enklare modell, eftersom 20 m-data inte kan skilja dem åt.
    while n > 1:
        c = best.x[2 * n + 2:]
        kv = [max(c[i], c[i + 1]) / max(min(c[i], c[i + 1]), 1e-9) for i in range(n - 1)]
        if min(kv) >= 1.3:
            break
        enklare = [r for r in resultat if r[1] == n - 1]
        if not enklare:
            break
        bic, n, best = enklare[0]
    t, v = _knutar(best.x, n)
    dof = max(N - (3 * n + 2), 1)
    try:
        cov = np.linalg.pinv(best.jac.T @ best.jac) * (2 * best.cost / dof)
    except Exception:
        cov = np.full((len(best.x), len(best.x)), np.nan)
    bagar = []
    for i in range(n):
        ci = best.x[2 * n + 2 + i]
        sig = float(np.sqrt(max(cov[2 * n + 2 + i, 2 * n + 2 + i], 0)))
        ks, ke = t[2 * i + 1], t[2 * i + 2]
        bagar.append(dict(kc=ci * tecken, sigma_kc=sig, karna_start=ks, karna_slut=ke, Lc=ke - ks,
                          L_in=t[2 * i + 1] - t[2 * i], L_ut=t[2 * i + 3] - t[2 * i + 2],
                          radie=10000 / abs(ci) if abs(ci) > 1e-9 else np.inf))
    rms = float(np.sqrt(2 * best.cost / N))
    return dict(n=n, bagar=bagar, start=t[0], slut=t[-1], rms=rms, tecken=int(tecken), x=best.x)
