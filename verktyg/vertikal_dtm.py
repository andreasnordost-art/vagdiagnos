"""Vertikalradier mot höjdmodellen längs hela vägen (samma metod som kvalitet.py, avsnitt 1), för valfritt län.

Exempel: VD_WORK=/home/claude/work/lan23 python vertikal_dtm.py 45,14,84,87
-> {VD_WORK}/kvalitet/vertikal_mot_hojdmodell.csv
"""
import os, pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
WORK = Path(os.environ.get('VD_WORK', '/home/claude/work'))
rader = []
for vn in [int(v) for v in sys.argv[1].split(',')]:
    r = pickle.load(open(WORK / f'res/vag_{vn}.pkl', 'rb'))
    D = np.load(WORK / f'dtm/vag{vn}.npz'); s = D['s']; z = D['z']
    med = pd.Series(z).rolling(9, center=True, min_periods=3).median().to_numpy()
    ok = np.abs(z - med) < 0.4                     # broar och hopp i höjdmodellen utesluts
    for c in r.vertikal.itertuples():
        if c.start < s.min() + 20 or c.slut > s.max() - 20 or c.langd < 40:
            continue
        w = ok & (s >= c.start) & (s <= c.slut) & np.isfinite(z)
        if w.sum() < 5:
            continue
        try:
            k, cov = np.polyfit(s[w], z[w], 2, cov=True)
        except (np.linalg.LinAlgError, ValueError):
            continue
        if not np.all(np.isfinite(cov)) or cov[0, 0] < 0:
            continue
        a2 = k[0]; u = np.sqrt(cov[0, 0])
        rv = 1 / (2 * abs(a2)) if a2 != 0 else np.inf
        typ = 'konvex' if a2 < 0 else 'konkav'
        lag = 1 / (2 * (abs(a2) + 1.645 * u)); hog = 1 / (2 * max(abs(a2) - 1.645 * u, 1e-9))
        if not (np.isfinite(rv) and np.isfinite(lag)):
            continue
        rader.append(dict(vag=vn, nr=c.vertikal_nr, start=round(c.start), slut=round(c.slut), typ=c.typ, typ_dtm=typ,
                          Rv_motor=round(c.radie), Rv_lag=round(c.radie_lag), Rv_hog=round(c.radie_hog),
                          Rv_dtm=round(rv), Rv_dtm_lag=round(lag), Rv_dtm_hog=round(min(hog, 1e6)), kvot=round(c.radie / rv, 3),
                          intervall_overlappar=not (hog < c.radie_lag or lag > c.radie_hog), samma_typ=typ == c.typ))
V = pd.DataFrame(rader); V.to_csv(WORK / 'kvalitet/vertikal_mot_hojdmodell.csv', index=False)
vv = V[V.Rv_motor < 20000]
print(len(V), 'kurvor; median kvot', round(vv.kvot.median(), 3), 'överlapp', round(vv.intervall_overlappar.mean(), 3), 'samma typ', round(vv.samma_typ.mean(), 3))
