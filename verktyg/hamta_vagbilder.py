"""Hämta PMS-vägbilder (framåtkamera) var 20:e meter för en delsträcka, båda riktningarna.

Exempel: python hamta_vagbilder.py 83 3000 8000 /home/claude/work/bilder/vag83
Riktning 2 har egen längdmätning (s = L - LL); filnamnet anger s i riktning 1.
"""
import sys, json, time
from pathlib import Path
import requests, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vagdiagnos import pms
vn, s0, s1, ut = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), Path(sys.argv[4])
ut.mkdir(parents=True, exist_ok=True)
vl = pms.vaglista(Path('/home/claude/data'), 22)
pms.EGENSKAPER = pms.EGENSKAPER + ["Vagracke", "VagrackeTyp", "VagrackeMitt"]
BAS = pms.BAS + 'image'
index = []
for r in (1, 2):
    v = [x for x in vl if x['vn'] == vn and x['r'] == r and x['vun'] == 0][0]; L = v['l']
    a_, b_ = (s0, s1) if r == 1 else (L - s1, L - s0)
    exp = pms.hamta_export(Path('/home/claude/data/racke'), 22, v, start=max(0, int(a_) - 100), slut=min(int(L), int(b_) + 100))
    m = next(df for n, df in pd.read_excel(exp, sheet_name=None).items() if n.startswith('Mätdata'))
    def en(s):
        ll = s if r == 1 else L - s
        rad = m.iloc[(m.StartLL - ll).abs().argmin()]
        datum = str(rad['Mätdatum'])[:10]  # periodens datum styr bilduttaget
        f = ut / f'r{r}_{s:05d}.jpg'
        if not f.exists():
            q = None
            for forsok in range(4):
                q = requests.get(BAS, params=dict(lankod=22, vagnummer=vn, vagundernummer=0, riktning=r, korfalt=10,
                                                 lopandeLangd=int(ll), periodDatum=datum, imageSize=2), timeout=90)
                if q.status_code in (200, 204): break
                time.sleep(2)
            if q.status_code != 200 or not q.content:
                return dict(r=r, s=s, ll=int(ll), datum=datum, fil=None)
            f.write_bytes(q.content)
        return dict(r=r, s=s, ll=int(ll), datum=datum, fil=f.name)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(4) as ex:
        index += list(ex.map(en, range(int(s0), int(s1) + 1, 20)))
    print('riktning', r, 'klar', flush=True)
(ut / 'index.json').write_text(json.dumps(index, indent=0))
