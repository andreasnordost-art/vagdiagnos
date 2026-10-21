"""Typ av närmaste fasta föremål per sida i tvärsektionerna: träd, buskage, stolpe, byggnad eller ledning.

Laserdatat (Lantmäteriet, skog) är oklassat ovan mark, så typen bestäms av punktklustrets form runt
föremålet (±4 m längs vägen, ±2,5 m i sidled): höjd över marken, hur brett klustret är, hur platt
toppen är och om det finns punkter nära marken. Därefter jämförs med skyltregistret ur vägbilderna:
en stolpe eller ett märke som syns i bilderna på samma plats avgör typen.

Regler (h = högsta höjd över mark, b = klustrets bredd som standardavvikelse i plan):
  - ledning: alla punkter högre än 4 m och inga under 3 m (luftledning eller trädkrona över vägen
    räknas inte som påkörbar)
  - stolpe: b < 0,45 m och h 3-16 m
  - byggnad: b > 1,6 m, minst 12 punkter som ligger i ett plan (tak; avvikelse < 0,2 m) och h 2,5-12 m
  - buskage: h < 4 m; annars träd
Typen är en skattning; laserdatat har 1-3 punkter per m².

Exempel: python foremalstyp.py 83   (skriver in typen i /home/claude/work/tvar/vag83.json)
"""
import json, pickle, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vagdiagnos.tvarsektion import bilinjar

W = Path('/home/claude/work/tvar')


def typ_ur_kluster(q, hg):
    """q: punkter (x, y, z), hg: höjd över mark per punkt."""
    m = hg > 1.5
    if m.sum() < 3:
        return None, {}
    q, hg = q[m], hg[m]
    h = float(hg.max())
    b = float(np.sqrt(np.var(q[:, 0]) + np.var(q[:, 1])))
    # tak: punkterna ligger i ett plan (lutande eller platt); trädkronor är ojämna
    if len(q) >= 12:
        A = np.c_[q[:, 0] - q[:, 0].mean(), q[:, 1] - q[:, 1].mean(), np.ones(len(q))]
        c, *_ = np.linalg.lstsq(A, q[:, 2], rcond=None)
        platt = float(np.sqrt(np.mean((q[:, 2] - A @ c) ** 2)))
    else:
        platt = 9.0
    lag = int((hg < 3.0).sum())
    f = dict(h=round(h, 1), b=round(b, 2), platt=round(platt, 2), n=int(len(q)), lag=lag)
    if hg.min() > 4.0 and lag == 0 and len(q) <= 12:
        return 'ledning', f
    if b < 0.45 and 3.0 <= h <= 16:
        return 'stolpe', f
    if b > 1.6 and platt < 0.2 and 2.5 <= h <= 12:
        return 'byggnad', f
    if h < 4.0:
        return 'buskage', f
    return 'träd', f


for vn in [int(v) for v in sys.argv[1].split(',')]:
    tf = W / f'vag{vn}.json'
    T = json.loads(tf.read_text())
    dtm, tr, _, P, _ = pickle.loads((W / f'vag{vn}.cache.pkl').read_bytes())
    P = P[(P[:, 3] == 1)]                       # oklassat ovan mark (ej mark, brus eller bro)
    tree = cKDTree(P[:, :2])
    reg = []
    rf = W / f'vag{vn}_skyltregister.json'
    if rf.exists():
        reg = [x for x in json.loads(rf.read_text())['skyltar'] if x['kat'] not in ('kantstolpe', 'baksida')]
    raknare = {}
    for k in T['sektioner']:
        ex, ey = (k['x1'] - k['x0']) / 60, (k['y1'] - k['y0']) / 60       # enhetsvektor i sidled (60 m sektion)
        for sd in ('vanster', 'hoger'):
            fo = k['sidor'][sd].get('foremal')
            if not fo or fo.get('t') is None:
                continue
            t = fo['t']; X = k['x0'] + (t + 30) * ex; Y = k['y0'] + (t + 30) * ey
            idx = tree.query_ball_point([X, Y], 4.0)
            if not idx:
                continue
            q = P[idx, :3]
            # inom ±2,5 m i sidled från föremålet
            d_sid = (q[:, 0] - X) * ex + (q[:, 1] - Y) * ey
            q = q[np.abs(d_sid) <= 2.5]
            if not len(q):
                continue
            hg = q[:, 2] - bilinjar(dtm, tr, q[:, 0], q[:, 1])
            typ, f = typ_ur_kluster(q, hg)
            kalla = 'laser'
            # vägbilderna: stolpe, belysning eller märke på samma plats
            for x in reg:
                if abs(x['s'] - k['s']) <= 10 and np.sign(x['t']) == np.sign(t) and abs(abs(x['t']) - abs(t)) <= 4:
                    if x['kat'] == 'stolpe':
                        typ = x['namn'].lower(); kalla = 'vägbild'
                    elif x['kat'] not in ('ovrigt',):
                        typ = 'vägmärke'; kalla = 'vägbild'
                    break
            if typ:
                fo['typ'] = typ; fo['typ_kalla'] = kalla
                raknare[typ] = raknare.get(typ, 0) + 1
    tf.write_text(json.dumps(T, ensure_ascii=False))
    print(vn, dict(sorted(raknare.items(), key=lambda z: -z[1])))
