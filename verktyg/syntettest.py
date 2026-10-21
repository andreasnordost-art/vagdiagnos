"""Kör syntetiskt facit för horisontal- och vertikalskattning."""
import sys, itertools, numpy as np, pandas as pd
sys.path.insert(0, '.')
from vagdiagnos.syntet import simulera_horisontal
from vagdiagnos.modell import anpassa_trapets

def horisontal(brus=0.8, seed=1):
    rng = np.random.default_rng(seed); rows = []
    for R, Lk, Lc in itertools.product([150, 300, 550, 900, 1500, 2500], [0, 30, 80, 250], [0, 20, 60, 200]):
        if Lk == 0 and Lc == 0: continue
        A = np.sqrt(Lk * R)
        for fas in [0, 7, 13]:
            a, b, y, p = simulera_horisontal(R, A, Lc, brus=brus, fas=fas, rng=rng)
            f = anpassa_trapets(a, b, y)
            m = np.abs(y); sel = m >= 2
            rows.append(dict(R=R, Lk=Lk, Lc=Lc, fas=fas, R_skattad=f['radie'], sigma_kc=f['sigma_kc'],
                             kc=f['kc'], langd_fit=f['slut'] - f['start'], Lc_fit=f['Lc'],
                             R_median_gammal=np.median(10000 / m[sel]) if sel.any() else np.nan,
                             R_ovre=10000 / m.max()))
    return pd.DataFrame(rows)

if __name__ == '__main__':
    d = horisontal()
    d['fel'] = d.R_skattad / d.R - 1
    d['fel_gammal'] = d.R_median_gammal / d.R - 1
    d['z'] = (abs(d.kc) - 10000 / d.R) / d.sigma_kc
    print(d.groupby(['Lk','Lc']).apply(lambda g: pd.Series(dict(p50=100*g.fel.median(), p90=100*g.fel.abs().quantile(.9), gammal=100*g.fel_gammal.median(), z90=g.z.abs().quantile(.9)))).round(1).to_string())
    print('R_ovre >= R alltid?', (d.R_ovre >= d.R * 0.97).mean())
