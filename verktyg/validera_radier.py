"""Oberoende kontroll av horisontalradier utan ritningar.

Jämför motorns radie (från mätbilens gyro, PMS Kurvatur) med två källor som
mäter något helt annat:
  1. GNSS: mätbilens egna koordinater (Start E/N) i båda körfälten.
  2. NVDB: vägnätets planlinje som PMS använder (geometry_r1.json).
I båda fallen anpassas en cirkel till punkterna i kurvans cirkelbåge.
"""
import json, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def cirkel(xy):
    x = xy[:, 0] - xy[:, 0].mean(); y = xy[:, 1] - xy[:, 1].mean()
    A = np.c_[2 * x, 2 * y, np.ones(len(x))]
    (cx, cy, c), *_ = np.linalg.lstsq(A, x * x + y * y, rcond=None)
    r = np.sqrt(c + cx * cx + cy * cy)
    return float(r), float(np.sqrt(np.mean((np.hypot(x - cx, y - cy) - r) ** 2)))


def kor(res_dir, raw_dir, vagar):
    rows = []
    for vn in vagar:
        r = pickle.load(open(Path(res_dir) / f'vag_{vn}.pkl', 'rb'))
        nv = np.array(json.loads((Path(raw_dir) / f'vag_{vn}/geometry_r1.json').read_text()))[:, :2]
        tree = cKDTree(nv)
        # GNSS-punkter: riktning 1 (start a) – koordinater finns i referenslinjen
        for c in r.horisontal.itertuples():
            ks, ke = c.karna_start, c.karna_slut
            if ke - ks < 60:
                continue
            ss = np.arange(ks, ke + 0.1, 10.0)
            x, y = r.ref.punkt(ss)
            # GNSS: använd originalpunkterna inom kärnan
            sel = (r.ref.s >= ks) & (r.ref.s <= ke)
            g = np.c_[r.ref.x[sel], r.ref.y[sel]]
            Rg, rmsg = cirkel(g) if len(g) >= 4 else (np.nan, np.nan)
            # NVDB: hörnpunkter mellan närmaste punkter till kärnans ändar
            _, i0 = tree.query([x[0], y[0]]); _, i1 = tree.query([x[-1], y[-1]])
            i0, i1 = sorted((i0, i1))
            p = nv[i0:i1 + 1]
            Rn, rmsn = cirkel(p) if len(p) >= 4 else (np.nan, np.nan)
            rows.append(dict(vag=vn, kurva=c.kurva_nr, karna_m=ke - ks, R_motor=c.radie,
                             osak=c.osakerhet_rel, R_r1=c.radie_r1, R_r2=c.radie_r2,
                             R_gnss=Rg, gnss_rms=rmsg, n_gnss=len(g), R_nvdb=Rn, nvdb_rms=rmsn,
                             n_nvdb=len(p), flaggor=c.flaggor))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    d = kor(sys.argv[1], sys.argv[2], [int(v) for v in sys.argv[3].split(',')])
    d.to_csv(sys.argv[4], index=False)
    ok = d[(d.R_motor < 2000) & (d.flaggor.fillna('') == '')]
    for ref in ['R_gnss', 'R_nvdb', 'R_r2']:
        q = ok[np.isfinite(ok[ref])]
        kv = q.R_motor / q[ref]
        print(f'{ref}: n={len(q)} median kvot {kv.median():.3f}  IQR {kv.quantile(.25):.2f}-{kv.quantile(.75):.2f}  inom ±10%: {(abs(kv-1)<.1).mean():.0%}  inom ±20%: {(abs(kv-1)<.2).mean():.0%}')
    q = ok[np.isfinite(ok.R_gnss) & np.isfinite(ok.R_nvdb)]
    kv = q.R_gnss / q.R_nvdb
    print(f'GNSS mot NVDB: median {kv.median():.3f} IQR {kv.quantile(.25):.2f}-{kv.quantile(.75):.2f}')
