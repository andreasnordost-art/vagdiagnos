"""Hämta NVDB:s vägräcken (via PMSv4) för en delsträcka, per sida i riktning 1.

PMS redovisar räcken per betraktelseriktning. Riktning 2 (Mot) har egen längdmätning
från vägens andra ände; den räknas om till riktning 1 med s = L - LL.
Antagande: räcke i en riktning avser vägens högra sida i den riktningen, alltså
Med -> höger och Mot -> vänster i tvärsektionerna.

Exempel: python hamta_racke.py 83 3000 8000 ut.json
"""
import json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vagdiagnos import pms

vn, s0, s1, ut = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), Path(sys.argv[4])
pms.EGENSKAPER = pms.EGENSKAPER + ["Vagracke", "VagrackeTyp", "VagrackeMitt"]
ROT = Path('/home/claude/data/racke'); lan = 22
vl = pms.vaglista(Path('/home/claude/data'), lan)
res = {}
for r, sida in ((1, 'hoger'), (2, 'vanster')):
    v = [x for x in vl if x['vn'] == vn and x['r'] == r and x['vun'] == 0][0]; L = v['l']
    a, b = (s0, s1) if r == 1 else (L - s1, L - s0)
    p = pms.hamta_export(ROT, lan, v, start=max(0, int(a) - 100), slut=min(int(L), int(b) + 100))
    bok = pd.read_excel(p, sheet_name=None)
    vd = next(df for n, df in bok.items() if n.startswith('Vägdata'))
    if r == 2:
        vd['a'], vd['b'] = L - vd.SlutLL, L - vd.StartLL
    else:
        vd['a'], vd['b'] = vd.StartLL, vd.SlutLL
    q = vd[(vd['Har vägräcke'] == 'Ja') & (vd.b > s0) & (vd.a < s1)].sort_values('a')
    lst = []
    for x in q.itertuples():
        typ = vd.loc[x.Index, 'Vägräckestyp'] if 'Vägräckestyp' in vd else None
        typ = None if pd.isna(typ) else str(typ)
        if lst and abs(lst[-1][1] - x.a) < 1 and lst[-1][2] == typ:
            lst[-1][1] = float(x.b)
        else:
            lst.append([float(x.a), float(x.b), typ])
    res[sida] = lst
    mitt = vd[(vd['Har mitträcke'] == 'Ja') & (vd.b > s0) & (vd.a < s1)].sort_values('a')
    if len(mitt):
        res.setdefault(f'mitt_r{r}', []).extend([[float(x.a), float(x.b)] for x in mitt.itertuples()])
res['kalla'] = 'NVDB via PMSv4 (Har vägräcke, Vägräckestyp)'
ut.write_text(json.dumps(res, ensure_ascii=False, indent=1))
print(json.dumps(res, ensure_ascii=False))
