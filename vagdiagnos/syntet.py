"""Syntetiska vägar med känd geometri, mätta som PMS gör (20 m medelvärden).

Används som facit: motorn ska återvinna de radier som lagts in.
"""
from __future__ import annotations

import numpy as np

from .modell import trapets_varde, ramp_varde


def pms_medel(fn, start, slut, steg=0.25):
    """Medelvärde av fn över varje intervall (numerisk integration)."""
    out = np.empty(len(start))
    for i, (a, b) in enumerate(zip(start, slut)):
        s = np.arange(a, b, steg) + steg / 2
        out[i] = fn(s).mean()
    return out


def intervall(langd, fas=0.0, steg=20.0):
    a = np.arange(fas, langd - steg, steg)
    return a, a + steg


def horisontalkurva(R, A, Lc, tecken=1, s0=200.0):
    """Klotoid (parameter A) – cirkelbåge (Lc) – klotoid. Returnerar trapetsparametrar."""
    Lk = A * A / R if A else 0.0
    return (s0, Lk, Lc, Lk, tecken * 10000.0 / R)


def simulera_horisontal(R, A, Lc, brus=0.8, fas=0.0, rng=None, langd=None):
    rng = rng or np.random.default_rng(0)
    p = horisontalkurva(R, A, Lc)
    total = p[0] + p[1] + p[2] + p[3] + 200 if langd is None else langd
    a, b = intervall(total, fas)
    y = pms_medel(lambda s: trapets_varde(s, p), a, b) + rng.normal(0, brus, len(a))
    return a, b, y, p


def simulera_vertikal(Rv, g1, g2, brus=0.1, fas=0.0, rng=None):
    rng = rng or np.random.default_rng(0)
    dg = g2 - g1
    L = Rv * abs(dg) / 100
    p = (g1, 200.0, L, dg)
    total = 400 + L
    a, b = intervall(total, fas)
    y = pms_medel(lambda s: ramp_varde(s, p), a, b) + rng.normal(0, brus, len(a))
    return a, b, y, p
