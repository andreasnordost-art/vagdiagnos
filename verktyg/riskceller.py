"""Riskunderlag per 20-metersavsnitt: vattensamling, avvattning och solbländning.

Ur PMS (båda riktningarna, senaste mätningen):
  va  - vattenarea i spåren (dm², 20 m), största av riktningarna
  res - resulterande lutning (%), sqrt(tvärfall² + längslutning²), minsta av riktningarna
  sd  - spårdjup max17 (mm, 100 m-värden), största av riktningarna
Ur solblandning.py: sol = timmar per år med bländrisk per riktning, månader och klockslag.

-> /home/claude/work/risk/vag{vn}_risk.json
"""
import os as _os
WORK = _os.environ.get('VD_WORK', '/home/claude/work'); LAN = int(_os.environ.get('VD_LAN', '22'))
import json, pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RISK = Path(f'{WORK}/risk')
VL = json.load(open(f'/home/claude/data/lan_{LAN}/vagar.json'))[0]['roads']


def blad(p, namn):
    b = pd.read_excel(p, sheet_name=None)
    return next((v for k, v in b.items() if k.startswith(namn) and 'K1' in k), None)


for vn in [int(v) for v in sys.argv[1].split(',')]:
    r = pickle.load(open(f'{WORK}/res/vag_{vn}.pkl', 'rb'))
    n = int(np.ceil((r.langd + 1) / 20.0)); mid = np.arange(n) * 20.0 + 10
    va = np.full(n, np.nan); res = np.full(n, np.nan); sd = np.full(n, np.nan)
    for rk in (1, 2):
        p = Path(f'/home/claude/data/risk/lan_{LAN}/vag_{vn}/export_r{rk}.xlsx')
        L = [x['l'] for x in VL if x['vn'] == vn and x['r'] == rk and x['vun'] == 0][0]
        m20 = blad(p, 'Mätdata 20m'); m100 = blad(p, 'Mätdata 100m')
        for m, kol in ((m20, None), (m100, None)):
            if m is None:
                continue
            m = m[pd.to_numeric(m.StartLL, errors='coerce').notna()].copy()
            c = (m.StartLL.astype(float) + m.SlutLL.astype(float)) / 2
            m['s'] = c if rk == 1 else L - c
            m = m.sort_values('s')
            def till(v, maxavst):
                v = pd.to_numeric(v, errors='coerce').to_numpy(float); ss = m.s.to_numpy(float)
                ok = np.isfinite(v)
                if ok.sum() < 2:
                    return np.full(n, np.nan)
                ut = np.interp(mid, ss[ok], v[ok])
                j = np.clip(np.searchsorted(ss[ok], mid), 1, ok.sum() - 1)
                avst = np.minimum(np.abs(ss[ok][j] - mid), np.abs(ss[ok][j - 1] - mid))
                ut[avst > maxavst] = np.nan
                return ut
            if 'Vattenarea' in m:
                va = np.fmax(va, till(m['Vattenarea'], 20))
                tf = till(m['Tvärfall'], 20); g = till(m['Backighet'], 20)
                res = np.fmin(res, np.sqrt(tf ** 2 + g ** 2))
            if 'Spårdjup max 17' in m:
                sd = np.fmax(sd, till(m['Spårdjup max 17'], 60))
    sol = json.loads((RISK / f'vag{vn}_sol.json').read_text())
    def r2(a, k=2):
        return [None if not np.isfinite(v) else round(float(v), k) for v in a]
    ut = dict(vag=vn, cell=20, va=r2(va), res=r2(res), sd=r2(sd, 1),
              sol=[[int(round(a)), int(round(b))] for a, b in sol['timmar'][:n]],
              solm=[list(x) for x in sol['manader'][:n]], solk=[list(x) for x in sol['klockslag'][:n]])
    (RISK / f'vag{vn}_risk.json').write_text(json.dumps(ut, separators=(',', ':')))
    f = lambda a: np.nanpercentile(a, [50, 90, 99]).round(2).tolist()
    print(vn, 'va', f(va), 'andel>0.2', round(float(np.nanmean(va > 0.2)), 3), '| res', f(res), 'andel<0.5', round(float(np.nanmean(res < 0.5)), 3),
          '| sd', f(sd), 'andel>=15', round(float(np.nanmean(sd >= 15)), 3), flush=True)
