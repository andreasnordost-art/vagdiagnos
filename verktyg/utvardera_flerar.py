"""Utvärdera flera mätår: ger medelvärdet över åren bättre överensstämmelse med GNSS-spåret än en mätning?

Skriver /home/claude/work/kvalitet/flerar_utvardering.json.
"""
import json
import numpy as np
import pandas as pd

F = pd.read_csv('/home/claude/work/kvalitet/flerar.csv')
H = pd.read_csv('/home/claude/work/leverans/kontroll_radier_gnss_nvdb.csv')
H = H[(H.R_motor >= 50) & (H.R_motor <= 5000) & (H.gnss_rms < 0.5)]
g = F.groupby(['vag', 'kurva_nr'])
S = g.agg(n=('R_ar', 'size'), R_motor=('R_motor', 'first'),
          lmed=('R_ar', lambda x: float(np.mean(np.log(x)))),
          lsd=('R_ar', lambda x: float(np.std(np.log(x), ddof=1)) if len(x) > 1 else np.nan)).reset_index()
S['R_flerar'] = np.exp(S.lmed)
J = S.merge(H[['vag', 'kurva', 'R_gnss']], left_on=['vag', 'kurva_nr'], right_on=['vag', 'kurva'])
J = J[J.n >= 3]


def mad(x):
    return float(np.median(np.abs(x)))


ut = dict(kurvor=int(len(J)), median_antal_ar=float(J.n.median()), median_sd_mellan_ar=round(float(J.lsd.median()), 3))
e_motor = np.log(J.R_motor / J.R_gnss); e_fler = np.log(J.R_flerar / J.R_gnss)
ut['median_abs_logfel_motor'] = round(mad(e_motor), 4)
ut['median_abs_logfel_flerar'] = round(mad(e_fler), 4)
ut['inom10_motor'] = round(float((np.abs(e_motor) < np.log(1.1)).mean()), 3)
ut['inom10_flerar'] = round(float((np.abs(e_fler) < np.log(1.1)).mean()), 3)
best = None
for w in np.arange(0, 1.01, 0.1):
    e = np.log(np.exp((1 - w) * np.log(J.R_motor) + w * J.lmed) / J.R_gnss)
    m = mad(e)
    if best is None or m < best[1]:
        best = (round(float(w), 1), round(m, 4))
ut['basta_vikt_flerar'] = best
# per radieklass
J['klass'] = pd.cut(J.R_motor, [0, 1000, 2500, 1e9], labels=['<1000', '1000-2500', '>2500'])
ut['per_klass'] = {str(k): dict(n=int(len(q)), motor=round(mad(np.log(q.R_motor / q.R_gnss)), 4),
                                flerar=round(mad(np.log(q.R_flerar / q.R_gnss)), 4),
                                sd_ar=round(float(q.lsd.median()), 3)) for k, q in J.groupby('klass', observed=True)}
open('/home/claude/work/kvalitet/flerar_utvardering.json', 'w').write(json.dumps(ut, indent=1, ensure_ascii=False))
print(json.dumps(ut, indent=1, ensure_ascii=False))
