"""Sikt i plan och profil ur Lantmäteriets höjdmodell och laserdata (skog).

För varje tvärsektion (var 10:e m) och körriktning placeras ett öga 1,10 m över vägbanan i
körfältets mitt. Siktlinjen dras rakt (i plan och höjd) till ett hinder 0,35 m högt i samma
körfält längre fram. Linjen räknas som skymd om
  - marken (höjdmodellen) når upp till linjen, t.ex. en skärning eller ett krön, eller
  - laserpunkter ovan mark (vegetation, byggnader) ligger inom 0,5 m från linjen i plan och
    når upp till linjens höjd.
Punkter på själva vägbanan räknas inte (där kan det ha stått fordon vid skanningen).
Tillgänglig sikt = längsta avstånd (10 m steg upp till 300 m) innan första skymda linje.

Exempel: python sikt_plan.py 83
Läser /home/claude/work/tvar/vag83.json och vag83.cache.pkl, skriver vag83_sikt.json.
"""
import json, pickle, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vagdiagnos.tvarsektion import bilinjar

OGA, HINDER, MAXD, STEG = 1.10, 0.35, 300, 10
vn = int(sys.argv[1]); W = Path('/home/claude/work/tvar')
STAM = sys.argv[2] if len(sys.argv) > 2 else f'vag{vn}'     # t.ex. vag4_0_5000 för en delsträcka
T = json.loads((W / f'{STAM}.json').read_text())
dtm, tr, _, P, _ = pickle.loads((W / f'{STAM}.cache.pkl').read_bytes())
if len(P):
    P = P[P[:, 3] != 2]                          # bara punkter ovan mark
S = T['sektioner']
DS = (S[1]['s'] - S[0]['s']) if len(S) > 1 else 10.0       # sektionsavstånd (10 eller 20 m)
s = np.array([k['s'] for k in S])
cx = np.array([(k['x0'] + k['x1']) / 2 for k in S]); cy = np.array([(k['y0'] + k['y1']) / 2 for k in S])
nx = np.array([(k['x1'] - k['x0']) / 60 for k in S]); ny = np.array([(k['y1'] - k['y0']) / 60 for k in S])
# vägens mitt mellan släntkrönen och belagd halvbredd per sektion
tc = np.array([(k['sidor']['hoger']['kant'] - k['sidor']['vanster']['kant']) / 2 for k in S])
hb = np.array([k.get('bredd', T['vagbredd']) / 2 for k in S])
mx, my = cx + tc * nx, cy + tc * ny            # vägbanans mitt
korbana = T.get('korbana', False)
# Vägbanans höjd: PMS-profilen (följer brobanor), nivåjusterad mot höjdmodellen längs mitten
res = pickle.load(open(f'/home/claude/work/res/vag_{vn}.pkl', 'rb')); pr = res.profil
zm_dtm = bilinjar(dtm, tr, mx, my); zp = np.interp(s, pr.s, pr.z, left=np.nan, right=np.nan)
NIVA = float(np.nanmedian(zp - zm_dtm))
def zvag(i):
    v = np.interp(s[i], pr.s, pr.z, left=np.nan, right=np.nan) - NIVA
    return v if np.isfinite(v) else zm_dtm[i]

# markhöjd för punkterna och sökträd
if len(P):
    hP = P[:, 2] - bilinjar(dtm, tr, P[:, 0], P[:, 1])
    hog = np.isfinite(hP) & (hP > 0.3)
    P = P[hog]
    # ta bort punkter på vägbanan (närmaste sektion)
    tsek = cKDTree(np.c_[mx, my]); _, j = tsek.query(P[:, :2])
    tv = (P[:, 0] - mx[j]) * nx[j] + (P[:, 1] - my[j]) * ny[j]
    P = P[np.abs(tv) > hb[j] + 0.3]
    HG = P[:, 2] - bilinjar(dtm, tr, P[:, 0], P[:, 1])   # höjd över mark
    trad = cKDTree(P[:, :2]) if len(P) else None
else:
    trad = None


def korfalt(i, rikt):
    """Ögats/hindrets läge: körfältets mitt (höger körfält i färdriktningen)."""
    off = hb[i] / 2 if not korbana else min(hb[i] / 2, 1.75)
    sg = 1 if rikt == 1 else -1
    return mx[i] + sg * off * nx[i], my[i] + sg * off * ny[i]


def zv(x, y):
    return bilinjar(dtm, tr, np.atleast_1d(x), np.atleast_1d(y))


ut = []
for i in range(len(S)):
    rad = dict(s=float(s[i]))
    for rikt in (1, 2):
        ex, ey = korfalt(i, rikt); ez = zvag(i)
        if not np.isfinite(ez):
            rad[f'r{rikt}'] = None; continue
        ez += OGA
        tillg, hinder, hp, slut = None, None, None, False
        for D in range(int(max(20, DS)), MAXD + 1, int(max(STEG, DS))):
            j = i + (int(round(D / DS)) if rikt == 1 else -int(round(D / DS)))
            if j < 0 or j >= len(S):
                slut = True; break
            tx, ty = korfalt(j, rikt); tz = zvag(j)
            if not np.isfinite(tz):
                slut = True; break
            tz += HINDER
            L = np.hypot(tx - ex, ty - ey); n = max(int(L), 2)
            f = np.linspace(0.04, 0.97, n)
            px, py = ex + (tx - ex) * f, ey + (ty - ey) * f; pz = ez + (tz - ez) * f
            mark = zv(px, py)
            # på vägbanan används vägens höjd (brobanor finns inte i höjdmodellen)
            ii = np.clip(np.round(i + (f * D / DS if rikt == 1 else -f * D / DS)).astype(int), 0, len(S) - 1)
            pa = (px - mx[ii]) * nx[ii] + (py - my[ii]) * ny[ii]
            pa_vag = np.abs(pa) <= hb[ii]
            vz = np.array([zvag(k) for k in ii])
            mark = np.where(pa_vag, vz, mark)
            skym_m = np.isfinite(mark) & (mark > pz)
            skym_v = np.zeros(n, bool); hojd_v = np.zeros(n)
            if trad is not None:
                nara = trad.query_ball_point(np.c_[px, py], r=0.5)
                for q, idx in enumerate(nara):
                    if idx:
                        m_ = P[idx, 2] >= pz[q]
                        if m_.any():
                            skym_v[q] = True; hojd_v[q] = float(HG[np.array(idx)[m_]].max())
            if skym_m.any() or skym_v.any():
                q = int(np.argmax(skym_m | skym_v))
                hinder = 'terräng' if skym_m[q] and not skym_v[q] else ('låg vegetation' if hojd_v[q] < 1.5 else 'vegetation')
                hp = [round(float(px[q]), 1), round(float(py[q]), 1)]
                break
            tillg = D
        lut = None
        j2 = i + (int(round(100 / DS)) if rikt == 1 else -int(round(100 / DS)))
        if 0 <= j2 < len(S):
            z2 = zvag(j2)
            if np.isfinite(z2):
                lut = round(float((z2 - (ez - OGA)) / 100 * 100), 2)   # % över 100 m framåt
        rad[f'r{rikt}'] = dict(sikt=tillg if tillg is not None else 10, max=(tillg == MAXD and hinder is None),
                               hinder=hinder, hp=hp, lut=lut, slut=bool(slut and hinder is None))
    ut.append(rad)
(W / f'{STAM}_sikt.json').write_text(json.dumps(ut))
d1 = [r['r1']['sikt'] for r in ut if r.get('r1') and not r['r1']['slut']]
print(vn, 'sikt r1 median', np.median(d1), 'andel <110 m', round(np.mean(np.array(d1) < 110), 3))
