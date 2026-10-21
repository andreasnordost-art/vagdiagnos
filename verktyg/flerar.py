"""Flera mätår per kurva: anpassa varje kurvas radie separat för varje års PMS-mätning.

Varje år (2016-2025) med mätning i riktning 1 ger en oberoende radie för de kurvor som motorn
hittat i senaste mätningen. Medelvärdet (i log-radie) och spridningen mellan åren skrivs till
/home/claude/work/kvalitet/flerar.csv. Används i diagnos.vidga_osakerhet som en extra källa.

Exempel: python flerar.py 83,87 2016-2025
"""
import os as _os
WORK = _os.environ.get('VD_WORK', '/home/claude/work'); LAN = int(_os.environ.get('VD_LAN', '22'))
import sys, pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vagdiagnos import pms
from vagdiagnos.pms import las_export, valj_senaste
from vagdiagnos.modell import anpassa_flerbage
from vagdiagnos.analys import KORSNING_K

ROT = Path('/home/claude/data')
UT = Path(f'{WORK}/kvalitet/flerar.csv')
vagar = [int(x) for x in sys.argv[1].split(',')]
a0, a1 = (int(x) for x in sys.argv[2].split('-'))
vl = pms.vaglista(ROT, LAN)


def ladda(vn, ar):
    v = [x for x in vl if x['vn'] == vn and x['r'] == 1 and x['vun'] == 0][0]
    try:
        e = las_export(pms.hamta_export(ROT, LAN, v, ar=ar))
        m, _ = valj_senaste(e.matning)
        m = m[m.k.notna()]
        if len(m) < 50:
            return vn, ar, None
        return vn, ar, (m.start.to_numpy(float), m.slut.to_numpy(float), m.k.to_numpy(float), str(m.datum.iloc[0])[:10])
    except Exception as ex:
        print(vn, ar, 'fel', str(ex)[:120], flush=True)
        return vn, ar, None


jobb = [(vn, ar) for vn in vagar for ar in range(a0, a1 + 1)]
with ThreadPoolExecutor(4) as tp:
    data = {(vn, ar): d for vn, ar, d in tp.map(lambda j: ladda(*j), jobb)}

rader = []
for vn in vagar:
    res = pickle.load(open(f'{WORK}/res/vag_{vn}.pkl', 'rb'))
    H = res.horisontal
    for c in H.itertuples():
        if 'sammansatt' in (c.flaggor or ''):
            continue
        for ar in range(a0, a1 + 1):
            d = data.get((vn, ar))
            if d is None:
                continue
            a, b, k, dat = d
            w = (a >= c.start - 1) & (b <= c.slut + 1) & (np.abs(k) < KORSNING_K)
            if w.sum() < 4:
                continue
            f = anpassa_flerbage(a[w], b[w], k[w], nmax=1)
            if f is None:
                continue
            R = f['bagar'][0]['radie']
            if np.isfinite(R) and 30 < R < 20000 and np.sign(f['bagar'][0]['kc']) == c.tecken:
                rader.append(dict(vag=vn, kurva_nr=c.kurva_nr, start=round(c.start), ar=ar, datum=dat,
                                  R_ar=round(R, 1), R_motor=round(c.radie, 1)))
    print(vn, 'klar', sum(r['vag'] == vn for r in rader), flush=True)

D = pd.DataFrame(rader)
if UT.exists():
    gam = pd.read_csv(UT)
    D = pd.concat([gam[~gam.vag.isin(vagar)], D])
D.to_csv(UT, index=False)
print('KLART', len(D))
