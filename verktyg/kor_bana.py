"""Radier ur GNSS-banan för alla kurvor (vagdiagnos.bana). -> /home/claude/work/kvalitet/bana.csv"""
import os as _os
WORK = _os.environ.get('VD_WORK', '/home/claude/work'); LAN = int(_os.environ.get('VD_LAN', '22'))
import pickle, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vagdiagnos.bana import radier_ur_bana
ut = []
for vn in ([int(v) for v in _os.environ['VD_VAGAR'].split(',')] if _os.environ.get('VD_VAGAR') else [4, 14, 83, 86, 87, 90, 331, 335, 348, 352]):
    r = pickle.load(open(f'{WORK}/res/vag_{vn}.pkl', 'rb'))
    B = radier_ur_bana(r); B.insert(0, 'vag', vn)
    B = B.merge(r.horisontal[['kurva_nr', 'radie', 'karna_start', 'karna_slut', 'flaggor']], on='kurva_nr')
    ut.append(B); print(vn, len(B), flush=True)
pd.concat(ut).to_csv(f'{WORK}/kvalitet/bana.csv', index=False)
