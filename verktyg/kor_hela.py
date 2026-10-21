"""Tvärsektioner (var 20:e meter), laser och sikt för hela vägarna, i delsträckor om 5 km.

Hoppar över delsträckor som redan är klara (…_sikt.json finns) och provsträckorna, så körningen
kan avbrytas och startas om. Två delsträckor körs parallellt. Laserdatat (cache) raderas när
sikten för delsträckan är beräknad. Inloggning via LM_USERNAME/LM_PASSWORD i miljön.
"""
import subprocess, sys, time, pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
W = Path('/home/claude/work/tvar'); PY = '/home/claude/venv/bin/python'; M = Path('/home/claude/motor')
PROV = {83: (3000, 8000), 348: (84900, 89900), 352: (62800, 67800), 4: (35200, 40200), 90: (154500, 159500),
        335: (46000, 51000), 86: (52600, 57600), 331: (15300, 20300), 14: (26500, 31500), 87: (22700, 27700)}
LOGG = open('/home/claude/work/kor_hela.log', 'a')
def logg(*a):
    print(time.strftime('%H:%M:%S'), *a, file=LOGG, flush=True)

jobb = []
for vn, (p0, p1) in PROV.items():
    L = int(pickle.load(open(f'/home/claude/work/res/vag_{vn}.pkl', 'rb')).langd) // 10 * 10
    for a, b in ((0, p0 - 20), (p1 + 20, L)):
        x = a
        while x < b - 50:
            y = min(x + 5000, b); jobb.append((vn, x, y)); x = y + 20

def kor(j):
    vn, a, b = j; stam = f'vag{vn}_{a}_{b}'
    if (W / f'{stam}_sikt.json').exists():
        return
    t0 = time.time()
    if not (W / f'{stam}.json').exists():
        r = subprocess.run([PY, 'verktyg/kor_tvarsektion.py', str(vn), str(a), str(b), str(W / f'{stam}.json'), '20'], cwd=M, capture_output=True, text=True)
        if r.returncode != 0:
            logg('FEL tvärsektion', stam, r.stderr[-300:]); return
    r = subprocess.run([PY, 'verktyg/sikt_plan.py', str(vn), stam], cwd=M, capture_output=True, text=True)
    if r.returncode != 0:
        logg('FEL sikt', stam, r.stderr[-300:]); return
    (W / f'{stam}.cache.pkl').unlink(missing_ok=True)
    logg('klar', stam, f'{time.time() - t0:.0f}s')

# räcken ur NVDB för hela vägarna (en gång per väg)
for vn in PROV:
    f = W / f'vag{vn}_racke_hela.json'
    if not f.exists():
        L = int(pickle.load(open(f'/home/claude/work/res/vag_{vn}.pkl', 'rb')).langd)
        r = subprocess.run([PY, 'verktyg/hamta_racke.py', str(vn), '0', str(L), str(f)], cwd=M, capture_output=True, text=True)
        logg('räcken', vn, 'ok' if r.returncode == 0 else r.stderr[-200:])

logg('start', len(jobb), 'delsträckor')
with ThreadPoolExecutor(2) as ex:
    list(ex.map(kor, jobb))
logg('ALLT_KLART')
