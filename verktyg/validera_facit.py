"""Validering mot relationshandling: E12 Västra länken, Umeå.

1. Kör motorn på PMS-data för sträckan.
2. Bygg facitets kurvatur och simulera PMS 20 m-medelvärden.
3. Hitta förskjutning (och skala) mellan ritningens km och PMS löpande längd.
4. Jämför horisontalradier, tvärfall, vertikalradier och höjder.
"""
import sys, pickle
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vagdiagnos.pms import las_export
from vagdiagnos.analys import analysera
from facit import e12_vastra_lanken as F

D = Path('/home/claude/data/lan_24/vag_12')
e1 = las_export(D / 'export_r1_16000_27000.xlsx'); e2 = las_export(D / 'export_r2_426153_438153.xlsx')
res = analysera(e1, e2)
pickle.dump(res, open('/home/claude/work/e12.pkl', 'wb'))


def k_facit(s):
    """Kurvatur i PMS-konvention (positiv = vänster) längs ritningens km."""
    k = np.zeros_like(s, float)
    el = F.PLAN
    for i, (a, b, typ, v) in enumerate(el):
        m = (s >= a) & (s < b)
        if typ == 'R':
            k[m] = -10000 / v
        elif typ == 'A':
            kin = -10000 / el[i - 1][3] if i > 0 and el[i - 1][2] == 'R' else 0.0
            kut = -10000 / el[i + 1][3] if i + 1 < len(el) and el[i + 1][2] == 'R' else 0.0
            if el[i - 1][2] == 'A':  # S-kurva: andra halvan börjar på noll
                kin = 0.0
            if i + 1 < len(el) and el[i + 1][2] == 'A':
                kut = 0.0
            k[m] = kin + (kut - kin) * (s[m] - a) / (b - a)
    return k


# Förskjutning: korrelera simulerade 20 m-medel mot PMS riktning 1
sig = res.signal
ss = np.arange(900, 6700, 0.5); kf = k_facit(ss)
best = None
cum = np.r_[0, np.cumsum(kf) * 0.5]
for skala in np.arange(0.990, 1.0101, 0.001):
    for off in np.arange(17000, 18500, 1.0):
        a = (sig.a.to_numpy() - off) * skala; b = (sig.b.to_numpy() - off) * skala
        ok = (a > 1000) & (b < 6650) & np.isfinite(sig.k_mix) & ~((b > 2700) & (a < 2920))
        if ok.sum() < 100:
            continue
        med = (np.interp(b[ok], ss, cum[1:]) - np.interp(a[ok], ss, cum[1:])) / (b[ok] - a[ok])
        err = np.mean(np.clip(np.abs(med - sig.k_mix.to_numpy()[ok]), 0, 5) ** 2)
        if best is None or err < best[1]:
            best = (off, err, skala)
off = best[0]; skala = best[2]
print(f'Förskjutning ritning -> PMS LL: {off:.0f} m, skala {skala:.3f}, robust RMS kurvatur {np.sqrt(best[1]):.2f}')

# Horisontalradier
rows = []
for c in res.horisontal.itertuples():
    ka, kb = (c.karna_start - off) * skala, (c.karna_slut - off) * skala
    mid = (ka + kb) / 2
    el = [e for e in F.PLAN if e[2] == 'R' and e[0] <= mid < e[1]]
    if not el:
        continue
    a, b, _, R = el[0]
    rows.append(dict(kurva=c.kurva_nr, km_start=round(c.start - off), km_slut=round(c.slut - off),
                     R_facit=abs(R), facit_typ=('korg' if any(e[2]=='A' and abs(e[0]-a)<2 for e in F.PLAN) and any(e[2]=='A' and abs(e[1]-b)<2 for e in F.PLAN) else ''), R_motor=round(c.radie), R_lag=round(c.radie_lag), R_hog=round(c.radie_hog),
                     inom_intervall=c.radie_lag <= abs(R) <= c.radie_hog,
                     fel_pct=round(100 * (c.radie / abs(R) - 1), 1), tecken_ok=(np.sign(c.tecken) == -np.sign(R)),
                     karna_facit=f'{a}-{b}', karna_motor=f'{ka:.0f}-{kb:.0f}', flaggor=c.flaggor))
h = pd.DataFrame(rows)
print(h.to_string())

# Gamla metoden (ChatGPT-versionen): viktad median över kandidatens intervall, största av riktningarna
old = []
for r in h.itertuples():
    a0, b0 = r.km_start + off, r.km_slut + off
    vals = []
    for d, sign in ((res.signal[['a', 'b', 'k']], 1),):
        q = d[(d.a >= a0) & (d.b <= b0) & (d.k.abs() >= 2)]
        if len(q): vals.append(np.median(10000 / q.k.abs()))
    q2 = res.signal[(res.signal.a >= a0) & (res.signal.b <= b0) & (res.signal.k2.abs() >= 2)]
    if len(q2): vals.append(np.median(10000 / q2.k2.abs()))
    old.append(max(vals) if vals else np.nan)
h['R_gammal_metod'] = np.round(old)
h['fel_gammal_pct'] = np.round(100 * (h.R_gammal_metod / h.R_facit - 1), 1)

# Tvärfall
tv = res.tvarfall.merge(res.horisontal[['kurva_nr', 'karna_start', 'karna_slut']], on='kurva_nr')
trows = []
for t in tv.itertuples():
    mid = ((t.karna_start + t.karna_slut) / 2 - off) * skala
    for (a, b), e in F.TVARFALL.items():
        if a <= mid < b:
            trows.append(dict(kurva=t.kurva_nr, km=f'{a}-{b}', tvarfall_facit=e,
                              r1=round(t.tvarfall_in_r1, 2), r2=round(t.tvarfall_in_r2, 2)))
tvd = pd.DataFrame(trows)
print(tvd.to_string())

# Vertikal
vrows = []
for v in res.vertikal.itertuples():
    mid = ((v.start + v.slut) / 2 - off) * skala
    el = [e for e in F.PROFIL if e[2] == 'Rv' and e[0] <= mid < e[1]]
    if not el: continue
    a, b, _, Rv = el[0]
    vrows.append(dict(km=f'{v.start-off:.0f}-{v.slut-off:.0f}', typ=v.typ, Rv_facit=abs(Rv),
                      typ_facit='konvex' if Rv > 0 else 'konkav', Rv_motor=round(v.radie),
                      lag=round(v.radie_lag), hog=round(v.radie_hog),
                      inom=v.radie_lag <= abs(Rv) <= v.radie_hog, fel_pct=round(100 * (v.radie / abs(Rv) - 1), 1),
                      facit_km=f'{a}-{b}'))
vd = pd.DataFrame(vrows)
print(vd.to_string())

# Höjder
p = res.profil
hz = pd.DataFrame([dict(km=k, z_facit=z, z_pms=round(float(np.interp(k / skala + off, p.s, p.z)), 2)) for k, z in F.HOJDER.items()])
hz['diff'] = (hz.z_pms - hz.z_facit).round(2)
print(hz.to_string())
with pd.ExcelWriter('/home/claude/work/validering_e12.xlsx') as w:
    h.to_excel(w, 'horisontal', index=False); tvd.to_excel(w, 'tvarfall', index=False)
    vd.to_excel(w, 'vertikal', index=False); hz.to_excel(w, 'hojder', index=False)
pickle.dump(dict(off=off, h=h, tv=tvd, v=vd, hz=hz), open('/home/claude/work/validering_e12.pkl', 'wb'))
