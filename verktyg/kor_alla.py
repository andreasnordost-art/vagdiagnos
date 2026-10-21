"""Kör analysen för alla vägar i arkivet och spara resultat som pickle."""
import sys, pickle, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vagdiagnos.pms import las_export
from vagdiagnos.analys import analysera
RAW = Path(sys.argv[1]); UT = Path(sys.argv[2]); UT.mkdir(parents=True, exist_ok=True)
vagar = [int(x) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 else [4,14,83,86,87,90,331,335,348,352]
for vn in vagar:
    t = time.time()
    e1 = las_export(RAW / f'vag_{vn}/export_r1.xlsx'); e2 = las_export(RAW / f'vag_{vn}/export_r2.xlsx')
    r = analysera(e1, e2)
    r.ref.tree = None
    pickle.dump(r, open(UT / f'vag_{vn}.pkl', 'wb'))
    print(vn, f'{time.time()-t:.0f}s', len(r.horisontal), 'h-kurvor', len(r.vertikal), 'v-kurvor', {k: round(v, 3) if isinstance(v, float) else v for k, v in r.kvalitet.items() if k != 'matdatum'}, flush=True)
