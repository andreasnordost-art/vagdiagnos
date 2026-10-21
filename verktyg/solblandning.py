"""Solbländning längs vägen: timmar per år då en lågt stående sol ligger nära förarens blickriktning.

Solens läge räknas med NOAA:s solpositionsformler (noggrannhet cirka 0,1°) var 10:e minut under ett
år, för vägens mittpunkt. För varje 20-metersavsnitt och körriktning räknas blickriktningen ur
planlinjen (verklig nordriktning, meridiankonvergensen i SWEREF 99 TM är borträknad) och längslutningen
ur längdprofilen. Bländning räknas när solen står över horisonten, högst 15° över blickriktningen
(högre upp skyddar solskyddet) och högst 20° åt sidan från färdriktningen. Skuggning från terräng, skog och byggnader ingår inte, så värdet
är det största möjliga (molnfritt och utan skugga).

-> /home/claude/work/risk/vag{vn}_sol.json: per avsnitt [timmar r1, timmar r2], månader (bitmask)
   och vanligaste klockslag (lokal tid) per riktning.
Exempel: python solblandning.py 83,348
"""
import os as _os
WORK = _os.environ.get('VD_WORK', '/home/claude/work'); LAN = int(_os.environ.get('VD_LAN', '22'))
import json, pickle, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
from pyproj import Transformer
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

UT = Path(f'{WORK}/risk'); UT.mkdir(exist_ok=True)
VINKEL = 20.0          # grader åt sidan från färdriktningen
HOJD = 15.0            # grader över blickriktningen (solskyddets gräns)
STEG_MIN = 10
T = Transformer.from_crs(3006, 4326, always_xy=True)


def solposition(tider, lat, lon):
    """NOAA: höjd och azimut (grader, azimut från norr medurs) för UTC-tider."""
    jd = np.array([t.timestamp() / 86400.0 + 2440587.5 for t in tider])
    jc = (jd - 2451545.0) / 36525.0
    L0 = np.mod(280.46646 + jc * (36000.76983 + jc * 0.0003032), 360)
    M = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
    e = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)
    Mr = np.radians(M)
    C = np.sin(Mr) * (1.914602 - jc * (0.004817 + 0.000014 * jc)) + np.sin(2 * Mr) * (0.019993 - 0.000101 * jc) + np.sin(3 * Mr) * 0.000289
    sann = L0 + C
    omega = 125.04 - 1934.136 * jc
    lam = sann - 0.00569 - 0.00478 * np.sin(np.radians(omega))
    eps0 = 23 + (26 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60) / 60
    eps = eps0 + 0.00256 * np.cos(np.radians(omega))
    dekl = np.degrees(np.arcsin(np.sin(np.radians(eps)) * np.sin(np.radians(lam))))
    y = np.tan(np.radians(eps / 2)) ** 2
    L0r = np.radians(L0)
    eqt = 4 * np.degrees(y * np.sin(2 * L0r) - 2 * e * np.sin(Mr) + 4 * e * y * np.sin(Mr) * np.cos(2 * L0r)
                         - 0.5 * y * y * np.sin(4 * L0r) - 1.25 * e * e * np.sin(2 * Mr))
    minuter = np.array([t.hour * 60 + t.minute + t.second / 60 for t in tider])
    tst = np.mod(minuter + eqt + 4 * lon, 1440)
    ha = np.where(tst / 4 < 0, tst / 4 + 180, tst / 4 - 180)
    latr, dr, har = np.radians(lat), np.radians(dekl), np.radians(ha)
    cosz = np.sin(latr) * np.sin(dr) + np.cos(latr) * np.cos(dr) * np.cos(har)
    zen = np.degrees(np.arccos(np.clip(cosz, -1, 1)))
    el = 90 - zen
    # ljusbrytning nära horisonten
    el = el + np.where(el > -0.575, 1.02 / np.tan(np.radians(el + 10.3 / (el + 5.11))) / 60, 0)
    az = np.degrees(np.arccos(np.clip((np.sin(latr) * np.cos(np.radians(zen)) - np.sin(dr)) / (np.cos(latr) * np.sin(np.radians(zen))), -1, 1)))
    az = np.where(ha > 0, np.mod(az + 180, 360), np.mod(540 - az, 360))
    return el, az


for vn in [int(v) for v in sys.argv[1].split(',')]:
    r = pickle.load(open(f'{WORK}/res/vag_{vn}.pkl', 'rb'))
    rs, rx, ry = np.asarray(r.ref.s, float), np.asarray(r.ref.x, float), np.asarray(r.ref.y, float)
    n = int(np.ceil((r.langd + 1) / 20.0))
    s = np.arange(n) * 20.0 + 10
    x, y = np.interp(s, rs, rx), np.interp(s, rs, ry)
    x1, y1 = np.interp(s + 20, rs, rx), np.interp(s + 20, rs, ry)
    x0, y0 = np.interp(s - 20, rs, rx), np.interp(s - 20, rs, ry)
    bar_grid = np.degrees(np.arctan2(x1 - x0, y1 - y0))                     # grid-bäring, från norr medurs
    lon, lat = T.transform(x, y)
    lon2, lat2 = T.transform(x, y + 1000.0)                                   # grid-norr i geografiska koordinater
    gamma = np.degrees(np.arctan2((lon2 - lon) * np.cos(np.radians(lat)), lat2 - lat))
    bar = np.mod(bar_grid + gamma, 360)                                       # verklig bäring, riktning 1
    pr = r.profil
    z = np.interp(s, pr.s, pr.z); dz = np.gradient(z, s)
    theta = np.degrees(np.arctan(np.clip(dz, -0.15, 0.15)))
    clat, clon = float(np.mean(lat)), float(np.mean(lon))
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    tider = [t0 + timedelta(minutes=STEG_MIN * k) for k in range(int(365 * 24 * 60 / STEG_MIN))]
    el, az = solposition(tider, clat, clon)
    m = (el > 0) & (el < HOJD + 9)
    el, az = el[m], az[m]; tider = [t for t, q in zip(tider, m) if q]
    manad = np.array([t.month for t in tider]); timme = np.array([(t.hour + 1 + (1 if 3 < t.month < 11 else 0)) % 24 for t in tider])  # ungefärlig svensk tid
    azr = np.radians(az)
    ut = dict(h=[], m=[], kl=[])
    for rikt in (1, 2):
        hb = np.mod(bar + (0 if rikt == 1 else 180), 360); th = theta if rikt == 1 else -theta
        H = np.zeros(n); M = np.zeros(n, int); KL = np.zeros(n, int)
        for a in range(0, n, 800):
            b = min(a + 800, n)
            daz = np.abs(np.mod(np.degrees(azr)[None, :] - hb[a:b][:, None] + 180, 360) - 180)
            rel = el[None, :] - th[a:b][:, None]          # solens höjd över blickriktningen
            g = (daz <= VINKEL) & (rel <= HOJD) & (rel >= -2)
            H[a:b] = g.sum(1) * STEG_MIN / 60.0
            for k in range(b - a):
                if g[k].any():
                    M[a + k] = int(np.bitwise_or.reduce(1 << (manad[g[k]] - 1)))
                    KL[a + k] = int(np.bincount(timme[g[k]], minlength=24).argmax())
        ut['h'].append([round(float(v), 1) for v in H]); ut['m'].append(M.tolist()); ut['kl'].append(KL.tolist())
    (UT / f'vag{vn}_sol.json').write_text(json.dumps(dict(vag=vn, cell=20, vinkel=VINKEL, hojd=HOJD, timmar=list(zip(*ut['h'])),
                                                          manader=list(zip(*ut['m'])), klockslag=list(zip(*ut['kl']))), separators=(',', ':')))
    hh = np.array(ut['h'])
    print(vn, n, 'median h/år', np.median(hh), 'p90', np.percentile(hh, 90), 'max', hh.max(), flush=True)
