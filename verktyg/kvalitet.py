"""Kvalitetssäkring av radier och tvärsektioner mot oberoende källor.

1. Vertikalradier: jämförs med Lantmäteriets markhöjdmodell längs vägens mitt (från tvärsektionerna).
2. Längdprofil: PMS-höjder jämförs med höjdmodellen (formen, efter att en konstant nivåskillnad tagits bort).
3. Horisontalradier: motorns radie mot cirkelanpassning till mätbilens GNSS-spår och NVDB-geometri,
   samt överensstämmelse mellan riktningarna.
4. Tvärsektioner: tolkad vägbredd mot PMS vägbredd, saknade kanter och höjdhopp.

Skriver kvalitet.json och kvalitet.csv i /home/claude/work/kvalitet.
"""
import json, pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

UT = Path('/home/claude/work/kvalitet'); UT.mkdir(exist_ok=True)
VAGAR = [4, 14, 83, 86, 87, 90, 331, 335, 348, 352]
rapport = {}

# ---------------------------------------------------------- 1-2 vertikalt och profil
vert_rader = []
prof_rader = []
for vn in VAGAR:
    tf = Path(f'/home/claude/work/tvar/vag{vn}.json'); df = Path(f'/home/claude/work/dtm/vag{vn}.npz')
    r = pickle.load(open(f'/home/claude/work/res/vag_{vn}.pkl', 'rb'))
    if df.exists():      # hela vägen: höjdmodellen längs mätbilens spår (verktyg/dtm_langs.py)
        D = np.load(df); s = D['s']; z = D['z']
    elif tf.exists():    # bara provsträckan: tvärsektionernas mitt
        T = json.loads(tf.read_text())
        s = np.array([k['s'] for k in T['sektioner']]); z = np.array([k['z_mitt'] for k in T['sektioner']])
    else:
        continue
    # broar: höjdmodellen visar marken under bron -> uteslut sektioner som avviker kraftigt från grannarna
    med = pd.Series(z).rolling(9, center=True, min_periods=3).median().to_numpy()
    ok = np.abs(z - med) < 0.4
    # profil: PMS z (relativ) mot höjdmodell
    pr = r.profil; pz = np.interp(s, pr.s, pr.z, left=np.nan, right=np.nan)
    m = ok & np.isfinite(pz)
    diff = pz[m] - z[m]; niv = np.median(diff)
    # jämför formen över 200 m-fönster (lutningsskillnad tas bort lokalt)
    res_ = []
    for a in np.arange(s.min(), s.max() - 200, 100):
        w = m & (s >= a) & (s < a + 200)
        if w.sum() < 15: continue
        d = pz[w] - z[w]
        if not np.isfinite(d).all(): continue
        k = np.polyfit(s[w], d, 1); res_.append(np.std(d - np.polyval(k, s[w])))
    prof_rader.append(dict(vag=vn, niva_pms_minus_dtm=round(float(niv), 2), spridning_m=round(float(np.std(diff - niv)), 3),
                           form_rms_200m=round(float(np.median(res_)), 3) if res_ else None))
    for c in r.vertikal.itertuples():
        if c.start < s.min() + 20 or c.slut > s.max() - 20 or c.langd < 40:
            continue
        w = ok & (s >= c.start) & (s <= c.slut)
        if w.sum() < 5:
            continue
        if not np.isfinite(z[w]).all():
            w = w & np.isfinite(z)
            if w.sum() < 5: continue
        try:
            k, cov = np.polyfit(s[w], z[w], 2, cov=True)
        except (np.linalg.LinAlgError, ValueError):
            continue
        if not np.all(np.isfinite(cov)): continue
        a2 = k[0]; u = np.sqrt(cov[0, 0])
        rv = 1 / (2 * abs(a2)) if a2 != 0 else np.inf
        typ_dtm = 'konvex' if a2 < 0 else 'konkav'
        rv_lag = 1 / (2 * (abs(a2) + 1.645 * u)); rv_hog = 1 / (2 * max(abs(a2) - 1.645 * u, 1e-9))
        if not (np.isfinite(rv) and np.isfinite(rv_lag) and np.isfinite(rv_hog)):
            continue
        overlapp = not (rv_hog < c.radie_lag or rv_lag > c.radie_hog)
        vert_rader.append(dict(vag=vn, nr=c.vertikal_nr, start=round(c.start), slut=round(c.slut), typ=c.typ, typ_dtm=typ_dtm,
                               Rv_motor=round(c.radie), Rv_lag=round(c.radie_lag), Rv_hog=round(c.radie_hog),
                               Rv_dtm=round(rv), Rv_dtm_lag=round(rv_lag), Rv_dtm_hog=round(min(rv_hog, 1e6)),
                               kvot=round(c.radie / rv, 3), intervall_overlappar=overlapp, samma_typ=typ_dtm == c.typ))
V = pd.DataFrame(vert_rader); P = pd.DataFrame(prof_rader)
V.to_csv(UT / 'vertikal_mot_hojdmodell.csv', index=False)
if len(V):
    vv = V[V.Rv_motor < 20000]
    rapport['vertikal'] = dict(antal=int(len(vv)), median_kvot=round(float(vv.kvot.median()), 3),
                               inom_20pct=round(float((np.abs(np.log(vv.kvot)) < np.log(1.2)).mean()), 2),
                               intervall_overlappar=round(float(vv.intervall_overlappar.mean()), 2),
                               samma_typ=round(float(vv.samma_typ.mean()), 2))
rapport['profil'] = prof_rader

# ---------------------------------------------------------- 3 horisontalt
H = pd.read_csv('/home/claude/work/leverans/kontroll_radier_gnss_nvdb.csv')
H = H[(H.R_motor >= 50) & (H.R_motor <= 5000)]
H['kvot_gnss'] = H.R_motor / H.R_gnss; H['kvot_nvdb'] = H.R_motor / H.R_nvdb; H['kvot_rikt'] = H.R_r1 / H.R_r2
def andel(x, g=0.10):
    x = x.dropna(); return round(float((np.abs(np.log(x)) < np.log(1 + g)).mean()), 3) if len(x) else None
rapport['horisontal'] = dict(antal=int(len(H)), median_gnss=round(float(H.kvot_gnss.median()), 3), inom10_gnss=andel(H.kvot_gnss),
                             median_nvdb=round(float(H.kvot_nvdb.median()), 3), inom10_nvdb=andel(H.kvot_nvdb),
                             inom10_riktningar=andel(H.kvot_rikt))
# avvikare: motorn och GNSS skiljer mer än 25 % och NVDB stöder GNSS
H['avvikare'] = (np.abs(np.log(H.kvot_gnss)) > np.log(1.25)) & (np.abs(np.log(H.R_gnss / H.R_nvdb)) < np.log(1.15))
H[H.avvikare].to_csv(UT / 'horisontal_avvikare.csv', index=False)
rapport['horisontal']['avvikare'] = int(H.avvikare.sum())
rapport['horisontal']['per_vag'] = {int(v): dict(n=int(len(g)), inom10_gnss=andel(g.kvot_gnss), avvikare=int(g.avvikare.sum())) for v, g in H.groupby('vag')}

# ---------------------------------------------------------- 4 tvärsektioner
sek_rader = []
for vn in VAGAR:
    tf = Path(f'/home/claude/work/tvar/vag{vn}.json')
    if not tf.exists():
        continue
    T = json.loads(tf.read_text())
    bredd = np.array([k['sidor']['vanster']['kant'] + k['sidor']['hoger']['kant'] for k in T['sektioner']])
    pmsb = np.array([k.get('bredd', T['vagbredd']) for k in T['sektioner']])
    hv = T['vagbredd'] / 2
    fall = np.array([k['sidor'][sd].get('kant_metod') == 'pms' for k in T['sektioner'] for sd in ('vanster', 'hoger')])
    z = np.array([k['z_mitt'] for k in T['sektioner']]); med = pd.Series(z).rolling(9, center=True, min_periods=3).median().to_numpy()
    hopp = np.abs(z - med) >= 0.4
    sek_rader.append(dict(vag=vn, sektioner=len(bredd), vagbredd_pms=round(float(np.median(pmsb)), 2), tolkad_bredd_median=round(float(np.median(bredd)), 2),
                          tolkad_bredd_p10_p90=[round(float(x), 2) for x in np.percentile(bredd, [10, 90])],
                          andel_inom_1m=round(float((np.abs(bredd - pmsb) <= 2.0).mean()), 2),
                          andel_kant_ej_funnen=round(float(fall.mean()), 3), andel_hojdhopp=round(float(hopp.mean()), 3),
                          stodremsa_per_sida=round(float(np.median(bredd - pmsb) / 2), 2), vagbredd_pms_median=round(float(np.median(pmsb)), 2), motorvag=bool(T.get('korbana'))))
rapport['tvarsektioner'] = sek_rader

# ---------------------------------------------------------- 5 sidoområden: laser mot vägbild
sido = []
for vn in VAGAR:
    f = Path(f'/home/claude/work/tvar/vag{vn}_sidokontroll.json')
    if not f.exists():
        continue
    U = json.loads(f.read_text())
    kl = [(r[sd]['skog_bild'] >= 0.3, r[sd]['laser'] is not None and r[sd]['laser'] <= 15) for r in U for sd in ('vanster', 'hoger') if 'skog_bild' in r[sd]]
    if not kl:
        continue
    ka = np.array(kl)
    sido.append(dict(vag=vn, n=int(len(ka)), lika=round(float((ka[:, 0] == ka[:, 1]).mean()), 2),
                     laser_skog_bild_oppet=int((~ka[:, 0] & ka[:, 1]).sum()), bild_skog_laser_oppet=int((ka[:, 0] & ~ka[:, 1]).sum())))
rapport['sidoomraden'] = sido

# ---------------------------------------------------------- 6 tvärprofil 6,4 m mot exportens tvärfall
VL = json.load(open('/home/claude/data/lan_22/vagar.json'))[0]['roads']
tp = []
for vn in VAGAR:
    for suf, rk in (('', 1), ('_r2', 2)):
        f = Path(f'/home/claude/work/tvar/vag{vn}_tvarprofil{suf}.json'); ef = Path(f'/home/claude/src/arkiv/raw/vag_{vn}/export_r{rk}.xlsx')
        if not f.exists() or not ef.exists():
            continue
        D = pd.DataFrame(json.loads(f.read_text())['sektioner']).drop_duplicates('s')
        bok = pd.read_excel(ef, sheet_name=None)
        m = [v for k, v in bok.items() if k.startswith('Mätdata') and 'K1' in k][0]
        m = m[pd.to_numeric(m.StartLL, errors='coerce').notna()].copy()
        mid = (m.StartLL.astype(float) + m.SlutLL.astype(float)) / 2
        if rk == 2:
            mid = [x['l'] for x in VL if x['vn'] == vn and x['r'] == 2 and x['vun'] == 0][0] - mid
        m['mid'] = mid; m['tf'] = pd.to_numeric(m['Tvärfall'], errors='coerce'); m['d'] = m['Mätdatum'].astype(str).str[:10]
        m = m.sort_values('mid')
        J = pd.merge_asof(D.assign(mid=D.s + 10.0).sort_values('mid'), m[['mid', 'tf', 'd']], on='mid', tolerance=10.0, direction='nearest')
        J = J[(J.d == J.datum) & J.tf.notna() & J.tf_eget.notna()]
        rad = dict(vag=vn, riktning=rk, n=int(len(D)), takfall=round(float(D.takfall.mean()), 2), jamforda=int(len(J)))
        if len(J) >= 20:
            dd = J.tf_eget - J.tf
            rad.update(korr=round(float(np.corrcoef(J.tf_eget, J.tf)[0, 1]), 3), diff_median=round(float(dd.median()), 2),
                       diff_abs_median=round(float(dd.abs().median()), 2))
        tp.append(rad)
rapport['tvarprofil'] = tp

# ---------------------------------------------------------- 7 flera mätår (se utvardera_flerar.py)
f = UT / 'flerar_utvardering.json'
if f.exists():
    rapport['flerar'] = json.loads(f.read_text())
(UT / 'kvalitet.json').write_text(json.dumps(rapport, ensure_ascii=False, indent=1))
print(json.dumps(rapport, ensure_ascii=False, indent=1))
