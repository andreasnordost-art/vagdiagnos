"""Beräkna tvärsektioner för en delsträcka. Exempel:
LM_USERNAME=... LM_PASSWORD=... python kor_tvarsektion.py 83 3000 8000 ut.json
"""
import json, sys, time, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import LineString
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vagdiagnos.pms import las_export, valj_senaste
from vagdiagnos.analys import _lagg_pa_ref
from vagdiagnos.referens import Referenslinje
from vagdiagnos import tvarsektion as T

vn, s0, s1, ut = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), Path(sys.argv[4])
STEG = float(sys.argv[5]) if len(sys.argv) > 5 else 10.0   # avstånd mellan sektioner (m)
RAW = Path('/home/claude/src/arkiv/raw') / f'vag_{vn}'
e1 = las_export(RAW / 'export_r1.xlsx'); e2 = las_export(RAW / 'export_r2.xlsx')
m1, _ = valj_senaste(e1.matning); m2, _ = valj_senaste(e2.matning)
ref = Referenslinje(m1.start.to_numpy(float), m1.e.to_numpy(), m1.n.to_numpy())
r2 = _lagg_pa_ref(ref, m2); r2 = r2[np.isfinite(r2.b)].sort_values('b')

def p2(s):
    return np.interp(s, r2.b, r2.e), np.interp(s, r2.b, r2.n)

# Motorväg (skilda körbanor): sektionerna följer riktning 1:s körbana i stället för vägens mitt
_vd = las_export(RAW / 'export_r1.xlsx').vagdata
_q = _vd[(_vd.start < s1) & (_vd.slut > s0)].vagtyp.astype(str)
KORBANA = bool(len(_q) and _q.str.startswith('Motor').mean() > 0.5)

def mitt_punkt(s):
    x1, y1 = ref.punkt(s); x2, y2 = p2(s)
    if not KORBANA and np.hypot(x2 - x1, y2 - y1) < 12:
        return (x1 + x2) / 2, (y1 + y2) / 2
    # Bara riktning 1: flytta 1,75 m åt vänster från körfältets mitt
    xa, ya = ref.punkt(s - 5); xb, yb = ref.punkt(s + 5); L = np.hypot(xb - xa, yb - ya)
    return x1 - 1.75 * (yb - ya) / L, y1 + 1.75 * (xb - xa) / L

def mittlinje(s):
    xa, ya = mitt_punkt(s - 10); xb, yb = mitt_punkt(s + 10); cx, cy = mitt_punkt(s)
    L = np.hypot(xb - xa, yb - ya)
    return cx, cy, (xb - xa) / L, (yb - ya) / L

# Vägbredd ur ny PMS-export om den finns
vagbredd = 7.0
try:
    from vagdiagnos import pms as _p0
    _p0.EGENSKAPER = _p0.EGENSKAPER + ["Vagracke", "VagrackeTyp", "VagrackeMitt"]
    _v1 = [x for x in _p0.vaglista(Path('/home/claude/data'), 22) if x['vn'] == vn and x['r'] == 1 and x['vun'] == 0][0]
    ny = _p0.hamta_export(Path('/home/claude/data/racke'), 22, _v1, start=max(0, int(s0) - 100), slut=min(int(_v1['l']), int(s1) + 100))
except Exception as e:
    print('vägdata saknas:', e); ny = Path(f'/home/claude/data/lan_22/vag_{vn}/export_r1.xlsx')
if ny.exists():
    vd = las_export(ny).vagdata
    q = vd[(vd.start < s1) & (vd.slut > s0)]
    if 'vagbredd' in q and q.vagbredd.notna().any():
        vagbredd = float(q.vagbredd.median())

linje = LineString([mitt_punkt(s) for s in np.arange(s0 - 20, s1 + 21, 10)])
korridor = linje.buffer(T.HALVBREDD + 8)
x0, y0, x1, y1 = korridor.bounds
t = Transformer.from_crs(3006, 4326, always_xy=True)
lo0, la0 = t.transform(x0, y0); lo1, la1 = t.transform(x1, y1)
bbox_w = (min(lo0, lo1), min(la0, la1), max(lo0, lo1), max(la0, la1))
cache = ut.with_suffix('.cache.pkl')
if cache.exists():
    dtm, tr, kallor_dtm, P, kallor_pm = pickle.loads(cache.read_bytes())
else:
    t0 = time.time()
    dtm, tr, kallor_dtm = T.las_dtm((x0, y0, x1, y1), bbox_w)
    print('markhöjdmodell', dtm.shape, f'{time.time()-t0:.0f}s', flush=True)
    t0 = time.time()
    try:
        P, kallor_pm = T.las_punktmoln((x0, y0, x1, y1), bbox_w, korridor)
    except Exception as e:
        print('punktmoln ej tillgängligt:', str(e)[:200]); P, kallor_pm = np.zeros((0, 4)), [dict(fel=str(e)[:200])]
    print('punkter', len(P), f'{time.time()-t0:.0f}s', flush=True)
    if len(P):
        cache.write_bytes(pickle.dumps((dtm, tr, kallor_dtm, P, kallor_pm)))
sek = T.sektioner(ref, mittlinje, dtm, tr, P, s0, s1, vagbredd=vagbredd, steg=STEG)
# Belagd bredd per sektion ur PMS vägdata. På mötesfri väg anges bredden per riktning: summera båda.
try:
    from vagdiagnos import pms as _pms
    _pms.EGENSKAPER = list(dict.fromkeys(_pms.EGENSKAPER + ["Vagracke", "VagrackeTyp", "VagrackeMitt"]))
    _vl = _pms.vaglista(Path('/home/claude/data'), 22)
    _vd = {}
    for _r in (1, 2):
        _v = [x for x in _vl if x['vn'] == vn and x['r'] == _r and x['vun'] == 0][0]; _L = _v['l']
        _a, _b = (s0, s1) if _r == 1 else (_L - s1, _L - s0)
        _f = _pms.hamta_export(Path('/home/claude/data/racke'), 22, _v, start=max(0, int(_a) - 100), slut=min(int(_L), int(_b) + 100))
        _d = las_export(_f).vagdata.copy()
        if _r == 2:
            _d['start'], _d['slut'] = _L - _d.slut, _L - _d.start
        _vd[_r] = _d
    def _rad(r, s):
        d = _vd[r]; q = d[(d.start <= s) & (d.slut > s)]
        return q.iloc[0] if len(q) else None
    for k in sek:
        a1 = _rad(1, k['s']); a2 = _rad(2, k['s'])
        if a1 is None or pd.isna(a1.vagbredd):
            continue
        b = float(a1.vagbredd)
        if 'mötesfri' in str(a1.vagtyp) and a2 is not None and pd.notna(a2.vagbredd) and not KORBANA:
            b += float(a2.vagbredd)
        k['bredd'] = round(b, 2)
except Exception as e:
    print('bredd per sektion saknas:', e)
T3857 = Transformer.from_crs(3006, 3857, always_xy=True)
for k in sek:
    a = T3857.transform(k['x0'], k['y0']); b = T3857.transform(k['x1'], k['y1'])
    k['m0'] = [round(a[0]), round(a[1])]; k['m1'] = [round(b[0]), round(b[1])]
ut.write_text(json.dumps(dict(vag=vn, s0=s0, s1=s1, steg=STEG, vagbredd=vagbredd, korbana=KORBANA, sektioner=sek,
                              kallor=dict(markhojdmodell=kallor_dtm, laserdata=kallor_pm)), ensure_ascii=False))
print('sektioner', len(sek))
