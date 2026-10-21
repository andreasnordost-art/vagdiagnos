"""Hämta PMS-export med spårdjup, vattenarea, textur och kantdjup för hela vägarna (båda riktningar).

-> /home/claude/data/risk/lan_22/vag_{vn}/export_r{r}.xlsx
"""
import os as _os
WORK = _os.environ.get('VD_WORK', '/home/claude/work'); LAN = int(_os.environ.get('VD_LAN', '22'))
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vagdiagnos import pms
pms.EGENSKAPER = pms.EGENSKAPER + ["Spardjup_Max17_MV", "Spardjup_Max15_MV", "Vattenarea", "Textur_MV", "Kantdjup_MV"]
ROT = Path('/home/claude/data/risk')
vl = pms.vaglista(Path('/home/claude/data'), LAN)
jobb = [x for x in vl if x['vn'] in [int(v) for v in sys.argv[1].split(',')] and x['vun'] == 0]
def en(v):
    try:
        p = pms.hamta_export(ROT, LAN, v); print(v['vn'], v['r'], 'ok', p.stat().st_size, flush=True)
    except Exception as e:
        print(v['vn'], v['r'], 'fel', str(e)[:150], flush=True)
with ThreadPoolExecutor(3) as ex:
    list(ex.map(en, jobb))
print('KLART')
