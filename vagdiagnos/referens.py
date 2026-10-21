"""Linjär referering längs mätbilens uppmätta färdlinje.

Riktning 1:s (Med) startpunkter och löpande längd bildar referenslinjen.
Riktning 2:s (Mot) punkter projiceras vinkelrätt på den, så att båda
riktningarnas mätvärden hamnar på samma längdmätning.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


class Referenslinje:
    def __init__(self, s: np.ndarray, x: np.ndarray, y: np.ndarray, steg_m: float = 1.0):
        o = np.argsort(s)
        s, x, y = np.asarray(s, float)[o], np.asarray(x, float)[o], np.asarray(y, float)[o]
        ok = np.isfinite(x) & np.isfinite(y)
        self.s, self.x, self.y = s[ok], x[ok], y[ok]
        # Förtätning i steg_m för snabb närmaste-punkt-sökning
        ss = np.arange(self.s[0], self.s[-1] + steg_m, steg_m)
        self.ts = ss
        self.tx = np.interp(ss, self.s, self.x)
        self.ty = np.interp(ss, self.s, self.y)
        self.tree = cKDTree(np.c_[self.tx, self.ty])

    def projicera(self, x, y, max_avstand_m=60.0):
        """Returnerar (s, sidoavstånd). NaN där punkten ligger längre bort än max_avstand_m."""
        d, i = self.tree.query(np.c_[x, y])
        s = self.ts[i].astype(float)
        # Förfina mot lokal segmentriktning
        j0 = np.clip(i - 1, 0, len(self.ts) - 1); j1 = np.clip(i + 1, 0, len(self.ts) - 1)
        dx = self.tx[j1] - self.tx[j0]; dy = self.ty[j1] - self.ty[j0]
        L = np.hypot(dx, dy); L[L == 0] = 1
        t = ((np.asarray(x) - self.tx[i]) * dx + (np.asarray(y) - self.ty[i]) * dy) / L
        s = s + t * (self.ts[j1] - self.ts[j0]) / np.maximum(L, 1e-9)
        sido = ((np.asarray(x) - self.tx[i]) * dy - (np.asarray(y) - self.ty[i]) * dx) / L
        s[d > max_avstand_m] = np.nan
        return s, sido

    def punkt(self, s):
        return np.interp(s, self.ts, self.tx), np.interp(s, self.ts, self.ty)

    @property
    def langd(self):
        return float(self.s[-1])
