"""Hela kedjan för ett nytt län (utan provsträckor): export, motor, GNSS-kontroll.

Exempel: VD_WORK=/home/claude/work/lan23 VD_LAN=23 python kor_lan.py 45,14,84,87
Därefter: hamta_riskdata.py, solblandning.py, riskceller.py, kor_bana.py och flerar.py med samma miljövariabler.
"""
import os, sys, pickle, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vagdiagnos import pms
from vagdiagnos.pms import las_export
from vagdiagnos.analys import analysera
from validera_radier import cirkel

WORK = Path(os.environ['VD_WORK']); LAN = int(os.environ['VD_LAN'])
vagar = [int(x) for x in sys.argv[1].split(',')]
vl = pms.vaglista(Path('/home/claude/data'), LAN)
rows = []
for vn in vagar:
    t = time.time(); ex = []
    for r in (1, 2):
        v = [x for x in vl if x['vn'] == vn and x['r'] == r and x['vun'] == 0][0]
        ex.append(las_export(pms.hamta_export(Path('/home/claude/data'), LAN, v)))
    res = analysera(ex[0], ex[1]); res.ref.tree = None
    pickle.dump(res, open(WORK / f'res/vag_{vn}.pkl', 'wb'))
    print(vn, f'{time.time()-t:.0f}s', len(res.horisontal), 'h-kurvor', len(res.vertikal), 'v-kurvor', flush=True)
    # GNSS-kontroll: cirkel genom mätbilens koordinater i varje kurvas cirkelbåge
    rs, rx, ry = np.asarray(res.ref.s), np.asarray(res.ref.x), np.asarray(res.ref.y)
    for c in res.horisontal.itertuples():
        ks, ke = c.karna_start, c.karna_slut
        if ke - ks < 60:
            continue
        sel = (rs >= ks) & (rs <= ke)
        Rg, rmsg = cirkel(np.c_[rx[sel], ry[sel]]) if sel.sum() >= 4 else (np.nan, np.nan)
        rows.append(dict(vag=vn, kurva=c.kurva_nr, karna_m=ke - ks, R_motor=c.radie, osak=c.osakerhet_rel, R_r1=c.radie_r1, R_r2=c.radie_r2,
                         R_gnss=Rg, gnss_rms=rmsg, n_gnss=int(sel.sum()), R_nvdb=np.nan, nvdb_rms=np.nan, n_nvdb=0, flaggor=c.flaggor))
pd.DataFrame(rows).to_csv(WORK / 'kvalitet/kontroll_radier_gnss_nvdb.csv', index=False)
print('KLART')
