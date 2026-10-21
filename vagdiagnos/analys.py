"""Geometrianalys av en väg från PMS-exporter för båda riktningarna.

Flöde:
 1. Välj senaste mätvärde per läge i K1 för respektive riktning.
 2. Lägg riktning 2 på riktning 1:s längdmätning (projektion av uppmätta koordinater).
 3. Horisontalkurvor: hitta kurvor i den sammanvägda kurvatursignalen och anpassa
    klotoid–cirkelbåge–klotoid till båda riktningarnas 20-metersmedelvärden.
 4. Tvärfall: inåtriktat tvärfall per körfält i kurvans cirkelbåge.
 5. Längdprofil: höjd ur mätbilens Z-koordinat och backighet (minsta kvadrat).
 6. Vertikalkurvor: anpassa lutningsramp till backigheten.
 7. Stoppsikt: tillgänglig sikt över längdprofilen i båda riktningarna.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsqr

from . import vgu
from .modell import anpassa_ramp, anpassa_flerbage
from .pms import Export, valj_senaste
from .referens import Referenslinje

KORSNING_K = 200.0          # |kurvatur| över detta (R < 50 m) = svängrörelse i korsning/cirkulation
KURVA_MIN_K = 2.0           # R <= 5000 m räknas som kurva
VERT_MAX_R = 30000.0


@dataclass
class Vagresultat:
    vagnummer: int
    langd: float
    ref: Referenslinje
    signal: pd.DataFrame          # per 20 m-intervall (riktning 1-ram)
    horisontal: pd.DataFrame
    tvarfall: pd.DataFrame
    vertikal: pd.DataFrame
    profil: pd.DataFrame          # per 2 m
    sikt: pd.DataFrame            # per 10 m och riktning
    hastighet: pd.DataFrame       # skyltad hastighet per intervall
    kvalitet: dict = field(default_factory=dict)


# ------------------------------------------------------------------ steg 1–2
def _lagg_pa_ref(ref: Referenslinje, m2: pd.DataFrame):
    s, sido = ref.projicera(m2.e.to_numpy(), m2.n.to_numpy())
    L = (m2.slut - m2.start).to_numpy(float)
    # Robust kontroll: s ska avta linjärt med riktning 2:s löpande längd
    ok = np.isfinite(s)
    if ok.sum() > 10:
        A = np.c_[m2.start.to_numpy(float)[ok], np.ones(ok.sum())]
        coef, *_ = np.linalg.lstsq(A, s[ok], rcond=None)
        for _ in range(3):
            pred = A @ coef; r = s[ok] - pred
            keep = np.abs(r) < max(100, 4 * np.median(np.abs(r)) + 1)
            coef, *_ = np.linalg.lstsq(A[keep], s[ok][keep], rcond=None)
        pred_all = m2.start.to_numpy(float) * coef[0] + coef[1]
        bad = ~np.isfinite(s) | (np.abs(s - pred_all) > 150)
        s[bad] = np.nan
    out = m2.copy()
    out["a"] = s - L
    out["b"] = s
    out["sido_m"] = sido
    return out


# ------------------------------------------------------------------ steg 3
def _kurvkandidater(sig: pd.DataFrame):
    """Kandidatfönster i sammanvägd kurvatur (riktning 1-ram)."""
    k = sig.k_mix.to_numpy()
    ks = pd.Series(k).rolling(3, center=True, min_periods=1).mean().to_numpy()
    a = sig.a.to_numpy(); b = sig.b.to_numpy()
    n = len(k); i = 0; out = []
    while i < n:
        if not np.isfinite(ks[i]) or abs(ks[i]) < KURVA_MIN_K:
            i += 1; continue
        tecken = np.sign(ks[i]); j = i
        while j + 1 < n and np.isfinite(ks[j + 1]) and np.sign(ks[j + 1]) == tecken and abs(ks[j + 1]) >= KURVA_MIN_K * 0.5:
            j += 1
        # utvidga till teckenbyte eller nära noll (klotoider), högst 150 m
        lo = i
        while lo - 1 >= 0 and np.isfinite(ks[lo - 1]) and np.sign(ks[lo - 1]) == tecken and abs(ks[lo - 1]) > 0.3 and a[i] - a[lo - 1] < 150:
            lo -= 1
        hi = j
        while hi + 1 < n and np.isfinite(ks[hi + 1]) and np.sign(ks[hi + 1]) == tecken and abs(ks[hi + 1]) > 0.3 and b[hi + 1] - b[j] < 150:
            hi += 1
        topp = np.nanmax(np.abs(ks[i:j + 1]))
        langd = b[j] - a[i]
        if langd >= 40 or topp >= 10:
            out.append((max(lo - 1, 0), min(hi + 1, n - 1), int(tecken)))
        i = hi + 1
    return out


def _horisontal(sig, r1, r2, brus_k):
    rows = []
    nr = 0
    for lo, hi, tecken in _kurvkandidater(sig):
        A0, B0 = sig.a.iloc[lo], sig.b.iloc[hi]
        def urval(d):
            return d[(d.a >= A0 - 1) & (d.b <= B0 + 1) & (d.k.abs() < KORSNING_K) & ~d.korsningszon]
        d1 = urval(r1); d2 = urval(r2)
        a = np.r_[d1.a, d2.a]; b = np.r_[d1.b, d2.b]; y = np.r_[d1.k, -d2.k]
        if len(y) < 4:
            continue
        langd_f = B0 - A0
        f = anpassa_flerbage(a, b, y, nmax=int(min(4, 1 + langd_f // 250)))
        if f is None:
            continue
        n = f["n"]
        f1 = anpassa_flerbage(d1.a, d1.b, d1.k, nmax=n, nmin=n, start=f["x"]) if len(d1) >= 3 * n + 3 else None
        f2 = anpassa_flerbage(d2.a, d2.b, -d2.k, nmax=n, nmin=n, start=f["x"]) if len(d2) >= 3 * n + 3 else None
        for i, bg in enumerate(f["bagar"]):
            R = bg["radie"]
            if not np.isfinite(R) or R > 5000 or R < 50:
                continue  # R < 50 m: svängrörelse i korsning, inte vägens kurva
            kc = abs(bg["kc"]); flaggor = []
            if len(d1) < 3 or len(d2) < 3:
                flaggor.append("endast en riktning")
            if n > 1:
                flaggor.append(f"del {i + 1} av {n} i sammansatt kurva")
            rel_fit = min(bg["sigma_kc"] / kc, 0.5) if np.isfinite(bg["sigma_kc"]) else 0.2
            R1 = f1["bagar"][i]["radie"] if f1 else np.nan
            R2 = f2["bagar"][i]["radie"] if f2 else np.nan
            rel_dir = min(abs(R1 - R2) / (2 * R), 0.5) if np.isfinite(R1) and np.isfinite(R2) else 0.15
            golv = 0.05 if bg["Lc"] >= 40 else 0.10
            if bg["Lc"] + bg["L_in"] + bg["L_ut"] < 40:
                golv = 0.25; flaggor.append("kort kurva, radie svårbestämd")
            if f["rms"] > 2.5 * brus_k:
                golv = max(golv, 0.15); flaggor.append("oregelbunden kurvatur")
            rel = math.sqrt(rel_fit ** 2 + rel_dir ** 2 + golv ** 2)
            nr += 1
            rows.append(dict(
                kurva_nr=nr, start=bg["karna_start"] - bg["L_in"], slut=bg["karna_slut"] + bg["L_ut"],
                karna_start=bg["karna_start"], karna_slut=bg["karna_slut"], tecken=int(np.sign(bg["kc"])),
                riktning="vänster" if bg["kc"] > 0 else "höger", radie=R, radie_r1=R1, radie_r2=R2,
                osakerhet_rel=rel, radie_lag=R / (1 + 1.645 * rel), radie_hog=R * (1 + 1.645 * rel),
                overgang_in=bg["L_in"], cirkelbage=bg["Lc"], overgang_ut=bg["L_ut"], rms=f["rms"],
                n_r1=len(d1), n_r2=len(d2), flaggor="; ".join(flaggor)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ steg 4
def _tvarfall(hor, r1, r2):
    rows = []
    for c in hor.itertuples():
        ks, ke = c.karna_start, c.karna_slut
        if ke - ks < 40:
            mid = (ks + ke) / 2; ks, ke = mid - 20, mid + 20
        res = dict(kurva_nr=c.kurva_nr)
        for namn, d, sign in (("r1", r1, 1), ("r2", r2, -1)):
            q = d[(d.b > ks) & (d.a < ke) & d.tvarfall.notna()]
            # Inåt: PMS-kurvatur positiv = vänsterkurva; tvärfall negativt = lägre på höger sida.
            # I körfältets egen riktning är kurvans tecken c.tecken*sign.
            e_in = (c.tecken * sign) * q.tvarfall.to_numpy()
            if len(e_in):
                med = float(np.median(e_in))
                mad = float(np.median(np.abs(e_in - med))) * 1.4826
                u = math.sqrt(0.2 ** 2 + (mad / math.sqrt(len(e_in))) ** 2)
            else:
                med, u = np.nan, np.nan
            res[f"tvarfall_in_{namn}"] = med; res[f"u_{namn}"] = u; res[f"n_{namn}"] = len(e_in)
            res[f"variation_{namn}"] = float(q.tvarfallsvariation.median()) if len(q) else np.nan
        vals = [(res["tvarfall_in_r1"], res["u_r1"]), (res["tvarfall_in_r2"], res["u_r2"])]
        vals = [v for v in vals if np.isfinite(v[0])]
        if vals:
            e, u = min(vals, key=lambda v: v[0])
        else:
            e, u = np.nan, np.nan
        res["tvarfall_in_min"] = e; res["u_tvarfall"] = u
        rows.append(res)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ steg 5
def _profil(sig, r1, r2, steg=2.0, sigma_z=0.15, sigma_g=0.15, glatt=0.02):
    """Höjdprofil på riktning 1-ram ur Z-punkter och backighet (gles minsta kvadrat)."""
    s0 = np.floor(np.nanmin(sig.a) / steg) * steg
    s1 = np.ceil(np.nanmax(sig.b) / steg) * steg
    x = np.arange(s0, s1 + steg, steg); N = len(x)
    rows, cols, vals, rhs = [], [], [], []
    r = 0

    def interp_row(s):
        t = (s - s0) / steg; i = int(np.clip(np.floor(t), 0, N - 2)); f = t - i
        return [i, i + 1], [1 - f, f]

    obs_z = []
    for d, sign in ((r1, 1), (r2, -1)):
        pts = d[np.isfinite(d.z) & np.isfinite(d.a)]
        s_start = pts.a.to_numpy() if sign == 1 else pts.b.to_numpy()
        for s, z in zip(s_start, pts.z.to_numpy()):
            obs_z.append((s, z))
        for a, b, g in zip(pts.a.to_numpy(), pts.b.to_numpy(), sign * pts.lutning.to_numpy()):
            if not np.isfinite(g) or b - a < 5:
                continue
            ia, wa = interp_row(a); ib, wb = interp_row(b)
            for i, w in zip(ib, wb):
                rows.append(r); cols.append(i); vals.append(w / (b - a) / (sigma_g / 100))
            for i, w in zip(ia, wa):
                rows.append(r); cols.append(i); vals.append(-w / (b - a) / (sigma_g / 100))
            rhs.append(g / 100 / (sigma_g / 100)); r += 1
    zref = np.median([z for _, z in obs_z])
    for s, z in obs_z:
        idx, w = interp_row(s)
        for i, ww in zip(idx, w):
            rows.append(r); cols.append(i); vals.append(ww / sigma_z)
        rhs.append((z - zref) / sigma_z); r += 1
    # glättning: andra differens (krökning) liten
    for i in range(1, N - 1):
        rows += [r, r, r]; cols += [i - 1, i, i + 1]
        vals += [1 / glatt, -2 / glatt, 1 / glatt]; rhs.append(0.0); r += 1
    A = sparse.csr_matrix((vals, (rows, cols)), shape=(r, N))
    h = lsqr(A, np.array(rhs), atol=1e-10, btol=1e-10, iter_lim=20000)[0] + zref
    # residualer för Z-punkter
    zres = np.array([z - np.interp(s, x, h) for s, z in obs_z])
    prof = pd.DataFrame(dict(s=x, z=h))
    prof["lutning"] = np.gradient(h, x) * 100
    # täckning: markera noder utan mätdata inom 30 m
    have = np.zeros(N, bool)
    for d in (r1, r2):
        for a, b in zip(d.a.to_numpy(), d.b.to_numpy()):
            if np.isfinite(a):
                have[int(max((a - s0) / steg, 0)):int(min((b - s0) / steg + 1, N))] = True
    prof["data"] = have
    return prof, float(np.sqrt(np.mean(zres ** 2)))


# ------------------------------------------------------------------ steg 6
def _vertikal(sig, r1, r2, brus_g):
    g = sig.g_mix.to_numpy()
    mid = ((sig.a + sig.b) / 2).to_numpy()
    # lokal lutningsändring per meter ur glidande linjär anpassning över 5 intervall
    c = np.full(len(g), np.nan)
    for i in range(2, len(g) - 2):
        w = slice(i - 2, i + 3)
        if np.isfinite(g[w]).all() and (mid[w][-1] - mid[w][0]) < 120:
            c[i] = np.polyfit(mid[w], g[w], 1)[0]
    thr = 100 / VERT_MAX_R  # %/m
    rows = []; i = 0; n = len(c)
    while i < n:
        if not np.isfinite(c[i]) or abs(c[i]) < thr:
            i += 1; continue
        tecken = np.sign(c[i]); j = i
        while True:
            while j + 1 < n and np.isfinite(c[j + 1]) and np.sign(c[j + 1]) == tecken and abs(c[j + 1]) >= thr:
                j += 1
            # slå ihop med nästa del av samma tecken om glappet är högst 3 intervall utan teckenbyte
            k = j + 1
            while k < n and k <= j + 4 and np.isfinite(c[k]) and abs(c[k]) < thr and np.sign(c[k]) != -tecken:
                k += 1
            if k < n and k <= j + 4 and np.isfinite(c[k]) and np.sign(c[k]) == tecken and abs(c[k]) >= thr:
                j = k
                continue
            break
        lo = max(i - 3, 0); hi = min(j + 3, n - 1)
        A0, B0 = sig.a.iloc[lo], sig.b.iloc[hi]
        d1 = r1[(r1.a >= A0 - 1) & (r1.b <= B0 + 1) & r1.lutning.notna() & ~r1.korsningszon]
        d2 = r2[(r2.a >= A0 - 1) & (r2.b <= B0 + 1) & r2.lutning.notna() & ~r2.korsningszon]
        a = np.r_[d1.a, d2.a]; b = np.r_[d1.b, d2.b]; y = np.r_[d1.lutning, -d2.lutning]
        o = np.argsort(a); a, b, y = a[o], b[o], y[o]
        i = j + 1
        if len(y) < 5:
            continue
        f = anpassa_ramp(a, b, y)
        if abs(f["dg"]) < 0.5 or f["radie"] > VERT_MAX_R:
            continue
        if f["L"] < 20 and abs(f["dg"]) < 2.0:
            continue  # för kort för att upplösas ur 20 m-data och för liten lutningsändring för att vara en verklig brytpunkt
        flaggor = []
        if len(d1) < 3 or len(d2) < 3:
            flaggor.append("endast en riktning")
        if f["L"] < 30:
            flaggor.append("kort vertikalkurva, radie svårbestämd")
        golv = 0.25 if f["L"] < 30 else 0.10
        if f["rms"] > 2.5 * brus_g:
            golv = max(golv, 0.2); flaggor.append("oregelbunden profil")
        rel = math.sqrt(min(f["rel_sigma"], 1.0) ** 2 + golv ** 2)
        rows.append(dict(start=f["start"], slut=f["slut"], typ=f["typ"], radie=f["radie"],
                         osakerhet_rel=rel, radie_lag=f["radie"] / (1 + 1.645 * rel),
                         radie_hog=f["radie"] * (1 + 1.645 * rel), lutning_in=f["g1"],
                         lutning_ut=f["g2"], langd=f["L"], rms=f["rms"], n=len(y),
                         flaggor="; ".join(flaggor)))
    v = pd.DataFrame(rows)
    if len(v):
        v = v.sort_values("start").reset_index(drop=True)
        v.insert(0, "vertikal_nr", np.arange(1, len(v) + 1))
    return v


# ------------------------------------------------------------------ steg 7
def _sikt(prof, steg_oga=10.0, max_sikt=400.0):
    """Tillgänglig stoppsikt längs profilen (endast vertikal skymning)."""
    s = prof.s.to_numpy(); z = prof.z.to_numpy(); ds = s[1] - s[0]
    M = int(max_sikt / ds)
    oh, hh = vgu.OGONHOJD_M, vgu.HINDERHOJD_M
    out = []
    for riktning, sgn in ((1, 1), (2, -1)):
        ss = s if sgn == 1 else s[::-1]; zz = z if sgn == 1 else z[::-1]
        dat = prof.data.to_numpy() if sgn == 1 else prof.data.to_numpy()[::-1]
        step = int(steg_oga / ds)
        for i in range(0, len(ss) - 2, step):
            j1 = min(i + M, len(ss) - 1)
            if j1 - i < 5 or not dat[i]:
                continue
            d = np.abs(ss[i + 1:j1 + 1] - ss[i])
            ze = zz[i] + oh
            terr = (zz[i + 1:j1 + 1] - ze) / d
            obj = (zz[i + 1:j1 + 1] + hh - ze) / d
            cmax = np.maximum.accumulate(terr)
            # objekt vid j syns om dess riktning ligger över all mellanliggande terräng
            prev = np.r_[-np.inf, cmax[:-1]]
            synlig = obj >= prev
            bort = np.where(~synlig)[0]
            avail = d[bort[0]] if len(bort) else (d[-1] if j1 == i + M else np.nan)
            begransad = len(bort) > 0
            # lutning i färdriktningen över närmaste 150 m
            k2 = min(i + int(150 / ds), len(ss) - 1)
            lut = 100 * (zz[k2] - zz[i]) / max(abs(ss[k2] - ss[i]), 1)
            out.append(dict(riktning=riktning, s=ss[i], tillganglig=avail,
                            begransad_av_profil=begransad, lutning_fram=lut,
                            vid_vagslut=(j1 < i + M and not begransad)))
    return pd.DataFrame(out)


# ------------------------------------------------------------------ huvud
def analysera(e1: Export, e2: Export) -> Vagresultat:
    assert e1.riktning == 1 and e2.riktning == 2 and e1.vagnummer == e2.vagnummer
    m1, bort1 = valj_senaste(e1.matning)
    m2, bort2 = valj_senaste(e2.matning)
    ref = Referenslinje(m1.start.to_numpy(float), m1.e.to_numpy(), m1.n.to_numpy())
    r1 = m1.copy(); r1["a"] = r1.start.astype(float); r1["b"] = r1.slut.astype(float); r1["sido_m"] = 0.0
    r2 = _lagg_pa_ref(ref, m2)
    r2 = r2[np.isfinite(r2.a)].sort_values("a").reset_index(drop=True)

    # Korsnings-/cirkulationszoner: där vägdata saknar hastighetsgräns (cirkulationsplatsens
    # ring tillhör annan länk) samt där mätbilen svänger skarpt (R < 50 m), med 40 m marginal.
    vd = e1.vagdata
    zoner = [(float(r.start) - 80, float(r.slut) + 80) for r in vd.itertuples()
             if not (pd.notna(pd.to_numeric(r.hastighet, errors="coerce")) and 0 < float(r.hastighet) <= 130)]
    for d in (r1, r2):
        for a, b in zip(d.a[d.k.abs() >= KORSNING_K], d.b[d.k.abs() >= KORSNING_K]):
            zoner.append((a - 40, b + 40))
    def i_zon(a, b):
        m = np.zeros(len(a), bool)
        for z0, z1 in zoner:
            m |= (b > z0) & (a < z1)
        return m
    r1["korsningszon"] = i_zon(r1.a.to_numpy(), r1.b.to_numpy())
    r2["korsningszon"] = i_zon(r2.a.to_numpy(), r2.b.to_numpy())

    # Sammanvägd signal på riktning 1:s intervall
    sig = r1[["a", "b", "k", "tvarfall", "lutning", "datum"]].copy()
    mid1 = ((sig.a + sig.b) / 2).to_numpy()
    mid2 = ((r2.a + r2.b) / 2).to_numpy()
    def till1(v):
        out = np.interp(mid1, mid2, v, left=np.nan, right=np.nan)
        # ingen interpolation över luckor > 30 m i riktning 2
        idx = np.searchsorted(mid2, mid1)
        gap = np.r_[np.inf, np.diff(mid2), np.inf][idx]
        out[gap > 30] = np.nan
        return out
    sig["k2"] = -till1(r2.k.to_numpy()); sig["g2"] = -till1(r2.lutning.to_numpy())
    k1 = sig.k.where(sig.k.abs() < KORSNING_K); k2 = sig.k2.where(sig.k2.abs() < KORSNING_K)
    sig["k_mix"] = pd.concat([k1, k2], axis=1).mean(axis=1)
    sig["g_mix"] = pd.concat([sig.lutning, sig.g2], axis=1).mean(axis=1)
    sig["korsning"] = r1.korsningszon.to_numpy()
    sig.loc[sig.korsning, ["k_mix", "g_mix"]] = np.nan

    # Brus skattas ur skillnaden mellan riktningarna där båda finns
    dk = (sig.k - sig.k2).dropna(); dg = (sig.lutning - sig.g2).dropna()
    brus_k = float(1.4826 * np.median(np.abs(dk - dk.median())) / math.sqrt(2)) if len(dk) else 1.0
    brus_g = float(1.4826 * np.median(np.abs(dg - dg.median())) / math.sqrt(2)) if len(dg) else 0.1

    hor = _horisontal(sig, r1, r2, brus_k)
    tv = _tvarfall(hor, r1, r2) if len(hor) else pd.DataFrame()
    prof, z_rms = _profil(sig, r1, r2)
    ver = _vertikal(sig, r1, r2, brus_g)
    sikt = _sikt(prof)

    # skyltad hastighet (riktning 1:s vägdata)
    v = e1.vagdata.rename(columns={"StartLL": "start", "SlutLL": "slut"})
    hast = v[["start", "slut", "hastighet", "vagtyp"]].copy()
    hast["hastighet"] = pd.to_numeric(hast.hastighet, errors="coerce")
    hast.loc[(hast.hastighet <= 0) | (hast.hastighet > 130), "hastighet"] = np.nan

    # Teckenkontroll av tvärfall (ska i median luta inåt i tydliga kurvor)
    tydliga = tv.merge(hor[["kurva_nr", "radie"]], on="kurva_nr")
    tydliga = tydliga[tydliga.radie < 800]
    tecken_ok = float(np.nanmedian(tydliga.tvarfall_in_min)) if len(tydliga) else np.nan

    kval = dict(brus_kurvatur=brus_k, brus_lutning=brus_g, z_rms_m=z_rms,
                bortvalda_r1=len(bort1), bortvalda_r2=len(bort2),
                andel_r2_projicerad=float(len(r2) / max(len(m2), 1)),
                median_inat_tvarfall_tydliga_kurvor=tecken_ok,
                matdatum=sorted({str(d)[:10] for d in pd.concat([m1.datum, m2.datum]).dropna().unique()}),
                langd=ref.langd)
    return Vagresultat(e1.vagnummer, ref.langd, ref, sig, hor, tv, ver, prof, sikt, hast, kval)
