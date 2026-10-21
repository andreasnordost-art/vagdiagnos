"""Tvärprofil 6,4 m (TP_65) ur PMSv4 för provsträckorna: tvärfall per körfältshalva och takprofil.

PMS-exportens Tvärfall avser bara mätbilens eget körfält (medel per 20 m). TP_65 ger 65 laserhöjder
över 6,4 m centrerat på körfältet (x = -3,2..3,2 m, negativt mot vägmitt), också per 20 m. Ur den
beräknas tvärfallet i eget körfält, tvärfallet strax över mittlinjen och om vägen har takfall
(krön mellan körfälten) eller ensidigt fall. Ensidigt fall krävs i kurvor med skevning.

Exempel: python tvarprofil.py 83 3000 8000 [2]  -> /home/claude/work/tvar/vag83_tvarprofil[_r2].json
(riktning 2: x och tvärfall avser riktning 2:s eget körfält; s räknas om till riktning 1)
"""
import json, sys, time
from pathlib import Path
import numpy as np
import requests

URL = "https://pmsv4.trafikverket.se/internal/api/data/tvarprofil"
X = np.arange(-3200, 3201, 100) / 1000.0
vn, s0, s1 = int(sys.argv[1]), int(float(sys.argv[2])), int(float(sys.argv[3]))
RIKT = int(sys.argv[4]) if len(sys.argv) > 4 else 1
vagar = json.load(open('/home/claude/data/lan_22/vagar.json'))[0]['roads']
v = [x for x in vagar if x['vn'] == vn and x['r'] == RIKT and x['vun'] == 0][0]
if RIKT == 2:   # riktning 2 har egen längdmätning från andra änden: s = L - LL
    s0, s1 = int(v['l'] - s1), int(v['l'] - s0)


def hamta(a, b):
    p = dict(vagNatId=v['id'], category=1, globalDateSetting=1, propertyName='tvarprofil_65', korfalt=10, riktning=RIKT,
             startLL=a, slutLL=b, giltighetsdatum=v['gd'], rotated='false', offsetToZero='true', type=2)
    for f in range(5):
        try:
            r = requests.get(URL, params=p, timeout=120)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(2 + 3 * f)
    return []


def lutning(x, z, a, b):
    w = (x >= a) & (x <= b) & np.isfinite(z)
    if w.sum() < 6:
        return None
    return float(np.polyfit(x[w], z[w], 1)[0] / 10.0)  # mm/m -> %


def knack(z):
    """Linje eller knäckt linje (krön) över x = -3,0..2,0 m. Krön godtas om det minskar RMS tydligt
    och ger fall åt båda håll."""
    w = (X >= -3.0) & (X <= 2.0) & np.isfinite(z)
    x, y = X[w], z[w]
    k1 = np.polyfit(x, y, 1); r1 = np.sqrt(np.mean((y - np.polyval(k1, x)) ** 2))
    hel = float(k1[0] / 10)
    bast = None
    for xb in np.arange(-2.4, 0.81, 0.1):
        A = np.c_[np.ones_like(x), np.minimum(x - xb, 0), np.maximum(x - xb, 0)]
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        r = np.sqrt(np.mean((y - A @ c) ** 2))
        if bast is None or r < bast[0]:
            bast = (r, xb, c[1] / 10, c[2] / 10)
    r2, xb, vl, hl = bast
    tak = bool(vl > 0.5 and hl < -0.5 and r2 < 0.6 * r1 and (vl - hl) > 1.5)
    if tak:
        return True, round(float(xb), 1), float(hl), float(vl), hel
    return False, None, hel, None, hel


ut = []
for a in range(s0, s1, 1000):
    for rec in hamta(a, min(a + 1000, s1)):
        z = np.array(rec['tvarprofil'], float)
        if len(z) != 65:
            continue
        tak, xk, eget, motsatt, hel = knack(z)
        full = lutning(X, z, -3.2, 3.2)   # som exportens Tvärfall: linje över hela profilen
        a_, b_ = rec['startLL'], rec['slutLL']
        if RIKT == 2:
            a_, b_ = v['l'] - b_, v['l'] - a_
        ut.append(dict(s=a_, t=b_, datum=rec['periodDatum'][:10], tf_eget=None if eget is None else round(eget, 2),
                       tf_motsatt=None if motsatt is None else round(motsatt, 2), tf_hel=None if hel is None else round(hel, 2), tf_full=None if full is None else round(full, 2),
                       kron_x=xk, takfall=tak))
    print(vn, a, len(ut), flush=True)
ut.sort(key=lambda r: r['s'])
suff = '' if RIKT == 1 else '_r2'
Path(f'/home/claude/work/tvar/vag{vn}_tvarprofil{suff}.json').write_text(json.dumps(dict(
    vag=vn, riktning=RIKT, kalla=f'PMSv4 tvärprofil TP_65 (6,4 m, 65 lasrar), riktning {RIKT} K1. s i riktning 1:s längdmätning.',
    sektioner=ut), ensure_ascii=False))
