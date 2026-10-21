"""Uppmätta hastigheter från TomTom (Traffic Stats, Route Analysis) kopplade till vägens längdmätning.

Läser ett eller flera TomTom-uttag (JSON-resultatet av ett jobb). Varje segment läggs på vägens
referenslinje (mätbilens GNSS-spår i riktning 1): segmentets punkter projiceras på linjen och ger
start och slut i vägens km. Stiger km längs segmentet gäller det riktning 1 (ökande km), annars riktning 2.

Utdata: {VD_WORK}/tomtom/vag{vn}.json med segmenten (s0, s1, riktning, percentiler 5-95 % i steg om 5,
median, medel, standardavvikelse, antal mätningar, hastighetsgräns) och uttagets period och tidsintervall.
Flera uttag (t.ex. en per riktning) slås ihop; segment för samma riktning och sträcka ersätts av det senaste.

Exempel: VD_WORK=/home/claude/work python tomtom.py 14 /home/claude/data/tomtom/raw/jobs_10115536_results_E14.json
"""
import json, os, pickle, sys
from pathlib import Path
import numpy as np
from pyproj import Transformer
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
WORK = os.environ.get('VD_WORK', '/home/claude/work')
T = Transformer.from_crs(4326, 3006, always_xy=True)
MAX_AVST = 30.0   # m; punkter längre från vägen än så räknas inte (annan väg, ramp)


def referens(vn):
    r = pickle.load(open(f'{WORK}/res/vag_{vn}.pkl', 'rb'))
    s, x, y = (np.asarray(a, float) for a in (r.ref.s, r.ref.x, r.ref.y))
    ok = np.isfinite(s) & np.isfinite(x) & np.isfinite(y); s, x, y = s[ok], x[ok], y[ok]
    t = np.arange(s.min(), s.max(), 1.0)     # tät linje, 1 m
    return t, np.interp(t, s, x), np.interp(t, s, y)


def las(fil, t, X, Y, tree):
    d = json.loads(Path(fil).read_text())
    dr = {x['@id']: x for x in d.get('dateRanges', [])}; ts = {x['@id']: x for x in d.get('timeSets', [])}
    ut, meta = [], []
    for rt in d['routes']:
        for sg in rt['segmentResults']:
            sh = sg.get('shape') or []
            if len(sh) < 2:
                continue
            lo = np.array([p['longitude'] for p in sh]); la = np.array([p['latitude'] for p in sh])
            px, py = T.transform(lo, la)
            avst, j = tree.query(np.c_[px, py])
            ok = avst <= MAX_AVST
            if ok.sum() < 2:
                continue
            ss = t[j[ok]]
            r = 1 if ss[-1] >= ss[0] else 2
            for tr in sg['segmentTimeResults']:
                p = tr.get('speedPercentiles') or []
                ut.append(dict(s0=float(min(ss)), s1=float(max(ss)), r=r, p=[round(float(v), 1) for v in p],
                               med=tr.get('medianSpeed'), mv=tr.get('averageSpeed'), sd=tr.get('standardDeviationSpeed'),
                               n=tr.get('sampleSize'), lim=sg.get('speedLimit'), ts=tr.get('timeSet'), dr=tr.get('dateRange'),
                               avst=round(float(np.median(avst[ok])), 1)))
        meta.append(dict(rutt=rt.get('routeName'), kalla=rt.get('probeSource'), karta=(rt.get('mapsVersions') or [''])[0],
                         period=[f"{x['from']}–{x['to']}" for x in dr.values()], tider=[x['name'] for x in ts.values()],
                         jobb=d.get('jobName'), skapad=d.get('creationTime', '')[:10]))
    return ut, meta


def main():
    vn = int(sys.argv[1]); filer = sys.argv[2:]
    t, X, Y = referens(vn); tree = cKDTree(np.c_[X, Y])
    utf = Path(f'{WORK}/tomtom'); utf.mkdir(parents=True, exist_ok=True); ut = utf / f'vag{vn}.json'
    seg, meta = [], []
    if ut.exists():
        g = json.loads(ut.read_text()); seg, meta = g['segment'], g['uttag']
    for f in filer:
        ny, m = las(f, t, X, Y, tree)
        for r in {x['r'] for x in ny}:   # nyare uttag ersätter äldre för samma riktning och sträcka
            a = min(x['s0'] for x in ny if x['r'] == r); b = max(x['s1'] for x in ny if x['r'] == r)
            seg = [x for x in seg if not (x['r'] == r and x['s0'] >= a - 5 and x['s1'] <= b + 5)]
        seg += ny; meta += m
        for r in (1, 2):
            q = [x for x in ny if x['r'] == r]
            if q:
                print(f, f'riktning {r}: {len(q)} segment, km {min(x["s0"] for x in q)/1000:.2f}–{max(x["s1"] for x in q)/1000:.2f}, '
                      f'median avstånd till vägen {np.median([x["avst"] for x in q]):.1f} m')
    seg.sort(key=lambda x: (x['r'], x['s0']))
    ut.write_text(json.dumps(dict(segment=seg, uttag=meta), ensure_ascii=False))
    print('skrev', ut, len(seg))


if __name__ == '__main__':
    main()
