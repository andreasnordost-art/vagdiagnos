"""Markhöjd ur Lantmäteriets markhöjdmodell (1 m) längs hela vägen, var 10:e meter.

Punkterna följer mätbilens GNSS-spår i riktning 1 (körfältets mitt). Höjdmodellen läses punktvis
direkt ur molnoptimerade GeoTIFF-filer, så bara de block som vägen passerar hämtas.
Inloggning: LM_USERNAME och LM_PASSWORD i miljön (sparas aldrig).

Exempel: python dtm_langs.py 83,87  -> /home/claude/work/dtm/vag83.npz (s, x, y, z)
"""
import os, pickle, sys, time
from pathlib import Path
import numpy as np
import requests
import rasterio
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
STAC = "https://api.lantmateriet.se/stac-hojd/v1/search"
WORK = os.environ.get('VD_WORK', '/home/claude/work')
UT = Path(f'{WORK}/dtm'); UT.mkdir(exist_ok=True)
T = Transformer.from_crs(3006, 4326, always_xy=True)
u, p = os.environ["LM_USERNAME"], os.environ["LM_PASSWORD"]
ENV = dict(GDAL_HTTP_USERPWD=f"{u}:{p}", GDAL_HTTP_AUTH="BASIC", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
           CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif", GDAL_HTTP_MAX_RETRY="10", GDAL_HTTP_RETRY_DELAY="1",
           GDAL_HTTP_RETRY_CODES="403,429,500,502,503,504",
           CURL_CA_BUNDLE=os.environ.get("CURL_CA_BUNDLE", "/root/.ccr/ca-bundle.crt"))


def sok(bbox):
    for f in range(6):
        try:
            r = requests.post(STAC, json=dict(collections=["dtm-cog"], bbox=list(bbox), limit=500), timeout=60)
            if r.status_code == 200:
                return r.json()["features"]
        except requests.RequestException:
            pass
        time.sleep(2 + 3 * f)
    raise RuntimeError("STAC-sökningen misslyckades")


for vn in [int(v) for v in sys.argv[1].split(',')]:
    ut = UT / f'vag{vn}.npz'
    if ut.exists():
        continue
    r = pickle.load(open(f'{WORK}/res/vag_{vn}.pkl', 'rb'))
    rs, rx, ry = np.asarray(r.ref.s, float), np.asarray(r.ref.x, float), np.asarray(r.ref.y, float)
    s = np.arange(rs.min(), rs.max(), 10.0)
    x, y = np.interp(s, rs, rx), np.interp(s, rs, ry)
    # inga punkter över luckor > 60 m i GNSS-spåret
    idx = np.searchsorted(rs, s); gap = np.r_[np.inf, np.diff(rs), np.inf][idx]
    z = np.full(len(s), np.nan)
    t0 = time.time()
    with rasterio.Env(**ENV):
        for a in range(0, len(s), 300):          # 3 km åt gången
            b = min(a + 300, len(s)); m = slice(a, b)
            lo, la = T.transform([x[m].min() - 50, x[m].max() + 50], [y[m].min() - 50, y[m].max() + 50])
            items = sok((min(lo), min(la), max(lo), max(la)))
            for it in items:
                href = it["assets"]["data"]["href"]
                for f in range(4):
                    try:
                        with rasterio.open("/vsicurl/" + href) as d:
                            bb = d.bounds
                            w = np.where((x[m] >= bb.left) & (x[m] < bb.right) & (y[m] > bb.bottom) & (y[m] <= bb.top) & np.isnan(z[m]))[0] + a
                            if len(w):
                                v = np.array([q[0] for q in d.sample(list(zip(x[w], y[w])))], float)
                                v[(v < -100) | (v > 3000)] = np.nan
                                z[w] = v
                        break
                    except Exception as e:
                        if f == 3:
                            print(vn, 'fel', href[-40:], str(e)[:80], flush=True)
                        time.sleep(2 + 3 * f)
            print(vn, f'{s[b-1]/1000:.1f} km', f'{np.isfinite(z[:b]).mean():.2f}', f'{time.time()-t0:.0f}s', flush=True)
    z[gap > 60] = np.nan
    np.savez(ut, s=s, x=x, y=y, z=z)
    print(vn, 'KLAR', len(s), flush=True)
print('ALLA KLARA')
