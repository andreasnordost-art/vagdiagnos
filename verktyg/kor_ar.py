"""Analysera varje års PMS-mätning för sig (2021-2025), för att väga samman kurvorna.

Resultat: /home/claude/work/res_ar/vag_{vn}_{ar}.pkl. Hoppar över år som redan är klara.
Exempel: python kor_ar.py 83,348 2021-2025
"""
import sys, pickle, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vagdiagnos import pms
from vagdiagnos.pms import las_export
from vagdiagnos.analys import analysera
UT = Path('/home/claude/work/res_ar'); UT.mkdir(exist_ok=True)
vagar = [int(x) for x in sys.argv[1].split(',')]
a0, a1 = (int(x) for x in sys.argv[2].split('-'))
vl = pms.vaglista(Path('/home/claude/data'), 22)
for vn in vagar:
    for ar in range(a0, a1 + 1):
        ut = UT / f'vag_{vn}_{ar}.pkl'
        if ut.exists():
            continue
        t = time.time()
        try:
            ex = []
            for r in (1, 2):
                v = [x for x in vl if x['vn'] == vn and x['r'] == r and x['vun'] == 0][0]
                ex.append(las_export(pms.hamta_export(Path('/home/claude/data'), 22, v, ar=ar)))
            n1 = ex[0].matning.k.notna().sum(); n2 = ex[1].matning.k.notna().sum()
            if min(n1, n2) < 0.5 * ex[0].matning.shape[0]:
                print(vn, ar, 'för lite data', n1, n2, flush=True); pickle.dump(None, open(ut, 'wb')); continue
            res = analysera(ex[0], ex[1]); res.ref.tree = None
            pickle.dump(res, open(ut, 'wb'))
            print(vn, ar, f'{time.time()-t:.0f}s', len(res.horisontal), 'kurvor', flush=True)
        except Exception as e:
            print(vn, ar, 'fel', str(e)[:200], flush=True)
print('KLART')
