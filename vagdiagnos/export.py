"""Export av analysresultat till kompakt JSON för vägrapporten (webb) och CSV."""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from . import vgu
from .diagnos import celler, kontroller, HASTIGHETER, DOMANER, OK, BRIST, OSAKER, EJ

KOD = {OK: "O", BRIST: "B", OSAKER: "U", EJ: "-"}


def _r(x, n=0):
    if x is None:
        return None
    try:
        if not math.isfinite(float(x)):
            return None
    except (TypeError, ValueError):
        return None
    return round(float(x), n) if n else int(round(float(x)))


def vagnamn(vn):
    return f"E{vn}" if vn in (4, 10, 12, 14, 16, 18, 20, 22, 45, 65) else f"Väg {vn}"


def bygg(res, lan_namn=""):
    # R < 50 m är mätbilens svängrörelse i korsning/cirkulation, inte vägens kurva
    res.horisontal = res.horisontal[res.horisontal.radie >= 50].reset_index(drop=True)
    from .diagnos import vidga_osakerhet
    res.horisontal = vidga_osakerhet(res.horisontal, getattr(res, 'gnss', None), getattr(res, 'flerar', None), getattr(res, 'bana', None))
    from .diagnos import markera_takfall
    res.horisontal = markera_takfall(res.horisontal, getattr(res, 'tp', None))
    from .diagnos import fusera_vertikal
    res.vertikal = fusera_vertikal(res.vertikal, getattr(res, 'vdtm', None))
    # Vertikalradie < 200 m eller lutning > 12 % är mätfel (t.ex. vid broskarvar eller korsningar)
    v = res.vertikal
    if len(v):
        res.vertikal = v[(v.radie >= 200) & (v.lutning_in.abs() <= 12) & (v.lutning_ut.abs() <= 12)].reset_index(drop=True)
    cel, stat = celler(res)
    # status per objekt (kurva/krön) och kontroll: {objekt: {domän: {std: sträng per hastighet}}}
    objst = {}
    for dom, a, b, V, std, st, varde, krav, obj in kontroller(res):
        objst.setdefault(obj, {}).setdefault(dom, {}).setdefault(std, {})[str(V)] = KOD[st]
    out = dict(vn=int(res.vagnummer), namn=vagnamn(int(res.vagnummer)), lan=lan_namn,
               langd=_r(res.langd), cell=20, matdatum=res.kvalitet.get("matdatum", []),
               kvalitet={k: (_r(v, 3) if isinstance(v, float) else v) for k, v in res.kvalitet.items()
                         if k != "matdatum"})
    # Karta: punkter var 20:e meter (SWEREF 99 TM)
    s = np.arange(0, res.langd + 1, 20.0)
    x, y = res.ref.punkt(s)
    out["xy"] = [[_r(a), _r(b)] for a, b in zip(x, y)]
    out["skyltad"] = [_r(v) for v in cel.skyltad]
    out["data"] = "".join("1" if d else "0" for d in cel.data)
    for std in ("ny", "om", "tdok"):
        out[f"saker_{std}"] = [_r(v) for v in cel[f"saker_{std}"]]
        out[f"mojlig_{std}"] = [_r(v) for v in cel[f"mojlig_{std}"]]
        # status per domän och hastighet: en teckensträng per cell
        out[f"st_{std}"] = {str(V): {d: "".join(KOD[x] for x in stat[(d, std, V)]) for d in DOMANER}
                            for V in HASTIGHETER}
    # Horisontalkurvor
    hk = []
    tv = res.tvarfall.set_index("kurva_nr") if len(res.tvarfall) else pd.DataFrame()
    for c in res.horisontal.itertuples():
        t = tv.loc[c.kurva_nr] if c.kurva_nr in tv.index else None
        mid = (c.karna_start + c.karna_slut) / 2
        h = res.hastighet
        hs = h[(h.start <= mid) & (h.slut > mid)].hastighet
        sk = _r(hs.iloc[0]) if len(hs) and pd.notna(hs.iloc[0]) else None
        vt = h[(h.start <= mid) & (h.slut > mid)].vagtyp.astype(str)
        ff = bool(len(vt)) and vt.str.startswith("Motor").any()
        krav_min = krav_norm = None
        if sk:
            krav_min, krav_norm = vgu.skevning_krav(c.radie, int(min(max(round(sk / 10) * 10, 60), 120)), ff)
        vt2 = h[(h.start <= mid) & (h.slut > mid)].vagtyp.astype(str)
        motesfri = bool(len(vt2)) and (("mötesfri" in vt2.iloc[0]) or vt2.iloc[0].startswith("Motor"))
        hk.append(dict(nr=int(c.kurva_nr), st=objst.get(f"H{c.kurva_nr}", {}), motesfri=motesfri, s0=_r(c.start), s1=_r(c.slut), k0=_r(c.karna_start), k1=_r(c.karna_slut),
                       R=_r(c.radie), Rl=_r(c.radie_lag), Rh=_r(c.radie_hog), R1=_r(c.radie_r1), R2=_r(c.radie_r2),
                       rikt=c.riktning, flagg=c.flaggor, sk=sk,
                       e1=_r(t.tvarfall_in_r1, 1) if t is not None else None,
                       e2=_r(t.tvarfall_in_r2, 1) if t is not None else None,
                       ue=_r(1.645 * t.u_tvarfall, 1) if t is not None else None,
                       emin=krav_min, enorm=krav_norm,
                       rikt_krav=vgu.horisontal_krav(sk)[0] if sk else None,
                       grans_krav=vgu.horisontal_krav(sk)[1] if sk else None))
    out["hk"] = hk
    vk = []
    for v in res.vertikal.itertuples():
        vk.append(dict(nr=int(v.vertikal_nr), st=objst.get(f"V{v.vertikal_nr}", {}), s0=_r(v.start), s1=_r(v.slut), typ=v.typ, R=_r(v.radie),
                       Rl=_r(v.radie_lag), Rh=_r(v.radie_hog), i1=_r(v.lutning_in, 2), i2=_r(v.lutning_ut, 2),
                       flagg=v.flaggor))
    out["vk"] = vk
    # Profil var 10:e meter
    p = res.profil
    ps = np.arange(0, res.langd + 1, 10.0)
    z = np.interp(ps, p.s, p.z)
    dat = np.interp(ps, p.s, p.data.astype(float), left=0, right=0) > 0.5
    out["prof"] = dict(steg=10, z=[_r(a, 2) if d else None for a, d in zip(z, dat)])
    # Stoppsikt per riktning var 10:e meter (riktning 2 redovisad på riktning 1:s längdmätning)
    sik = {}
    for r in (1, 2):
        q = res.sikt[res.sikt.riktning == r]
        sik[str(r)] = dict(s=[_r(a) for a in q.s], t=[_r(a) for a in q.tillganglig],
                           g=[_r(a, 1) for a in q.lutning_fram])
    out["sikt"] = sik
    return out, cel


def skriv(res, path, lan_namn=""):
    data, cel = bygg(res, lan_namn)
    txt = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    path.write_text(txt, encoding="utf-8")
    return data, cel
