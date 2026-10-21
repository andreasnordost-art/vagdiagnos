"""Kontroll av sidoområdena: laserdata mot PMS-vägbilder.

Var 100:e meter (riktning 1, framåtbild) segmenteras bilden med Mask2Former (Mapillary Vistas).
På rader som motsvarar 8-22 m framför bilen mäts på varje sida
  - var vägbanan slutar (klasserna väg/vägmarkering), och
  - var vegetationen börjar utanför vägbanan.
Kameramodellen (platt mark, horisont rad 160, 640x480) kalibreras i varje bild mot PMS vägbredd:
högra vägkanten ligger en kvarts vägbredd till höger om kameran, vänstra tre kvarts vägbredd till vänster.
Avståndet vägkant -> vegetation jämförs med laserns avstånd körbanekant -> närmaste föremål över 2,5 m
i sektionerna som bilden ser (8-22 m framför).

Exempel: python sidokontroll.py 83  -> /home/claude/work/tvar/vag83_sidokontroll.json
"""
import json, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

MODELL = "facebook/mask2former-swin-large-mapillary-vistas-semantic"
torch.set_num_threads(2)
proc = AutoImageProcessor.from_pretrained(MODELL)
model = Mask2FormerForUniversalSegmentation.from_pretrained(MODELL).eval()
lab = {int(k): v.lower() for k, v in model.config.id2label.items()}
VAG = [i for i, n in lab.items() if n in ('road', 'lane marking - general', 'service lane', 'crosswalk - plain', 'pothole', 'manhole', 'catch basin')]
VEG = [i for i, n in lab.items() if n == 'vegetation']
Y0, H_KAM, F = 160.0, 2.2, 320.0

for vn in [int(x) for x in sys.argv[1:]]:
    W = Path('/home/claude/work/tvar'); BM = Path('/home/claude/work/bilder') / f'vag{vn}'
    T = json.loads((W / f'vag{vn}.json').read_text()); S = T['sektioner']
    s = np.array([k['s'] for k in S])
    idx = [b for b in json.loads((BM / 'index.json').read_text()) if b['fil'] and b['r'] == 1 and b['s'] % 100 == 0]
    ut = []
    for b in idx:
        im = Image.open(BM / b['fil']).convert('RGB')
        with torch.no_grad():
            seg = proc.post_process_semantic_segmentation(model(**proc(images=im, return_tensors='pt')), target_sizes=[(480, 640)])[0].numpy()
        vag = np.isin(seg, VAG); veg = np.isin(seg, VEG)
        i = int(np.clip(np.searchsorted(s, b['s'] + 15), 0, len(s) - 1))
        bredd = S[i].get('bredd', T['vagbredd'])
        rad = dict(s=b['s'], fil=b['fil'])
        # Träd: vegetation som når över horisonten. Trädets fot = nedersta vegetationspixeln i samma kolumn.
        hog = veg[int(Y0) - 25]          # kolumner med vegetation klart över horisonten
        for sida, sg in (('vanster', -1), ('hoger', 1)):
            matt = []
            kol = range(330, 640, 3) if sg > 0 else range(310, -1, -3)
            for x in kol:
                if not hog[x]:
                    continue
                y = int(Y0) - 25
                while y < 479 and veg[y + 1, x]: y += 1
                if y < Y0 + 6:
                    continue
                d = H_KAM * F / (y - Y0)
                if not (8 <= d <= 30):
                    continue
                rv = vag[y]
                # vägkant på samma rad
                xe = 320
                if sg > 0:
                    while xe < 639 and (rv[xe] or rv[min(xe + 2, 639)]): xe += 1
                else:
                    while xe > 0 and (rv[xe] or rv[max(xe - 2, 0)]): xe -= 1
                if (sg > 0 and x <= xe) or (sg < 0 and x >= xe) or xe in (0, 639):
                    continue
                lat_kant = (xe - 320) * d / F
                kant_vantad = bredd / 4 if sg > 0 else -3 * bredd / 4
                k = kant_vantad / lat_kant if abs(lat_kant) > 0.3 else 1.0
                k = min(max(k, 0.6), 1.6)
                matt.append(abs((x - xe) * d / F * k))
            # närmaste trädkant = 10:e percentilen (robust mot enstaka felpixlar)
            m = [float(np.percentile(matt, 10))] * 3 if len(matt) >= 3 else []
            matt = matt if matt else [None] * 3
            m = [x for x in matt if x is not None]
            # laser: närmaste föremål >2,5 m från körbanekant i sektionerna 8-22 m framför
            las = []
            for kk in S:
                if b['s'] + 8 <= kk['s'] <= b['s'] + 22:
                    sd = kk['sidor'][sida]; fo = sd.get('foremal')
                    kV, kH = kk['sidor']['vanster']['kant'], kk['sidor']['hoger']['kant']; tc = (kH - kV) / 2; hb = kk.get('bredd', T['vagbredd']) / 2
                    kb = max(0.5, min(hb - tc, kV)) if sida == 'vanster' else max(0.5, min(hb + tc, kH))
                    las.append(max(0.0, abs(fo['t']) - kb) if fo else None)
            lv = [x for x in las if x is not None]
            # skog i bilden: andel av sidans kolumner där vegetation når klart över horisonten
            xs_ = np.arange(330, 640) if sg > 0 else np.arange(0, 310)
            skog_bild = float(hog[xs_].mean())
            rad[sida] = dict(bild=round(float(np.median(m)), 1) if len(m) >= 3 else None,
                             bild_oppet=not m, skog_bild=round(skog_bild, 2),
                             laser=round(float(np.median(lv)), 1) if lv else None,
                             laser_oppet=len(las) > 0 and not lv)
        ut.append(rad)
    (W / f'vag{vn}_sidokontroll.json').write_text(json.dumps(ut))
    # samstämmighet skog/öppet: laser = föremål över 2,5 m inom 15 m från körbanekant
    kl = [(r[sd]['skog_bild'] >= 0.3, r[sd]['laser'] is not None and r[sd]['laser'] <= 15) for r in ut for sd in ('vanster', 'hoger')]
    if kl:
        ka = np.array(kl); print(vn, 'skog/öppet lika', round(float((ka[:, 0] == ka[:, 1]).mean()), 2), 'n', len(ka),
                                  'bara laser', int((~ka[:, 0] & ka[:, 1]).sum()), 'bara bild', int((ka[:, 0] & ~ka[:, 1]).sum()), flush=True)
    par = [(r[sd]['bild'], r[sd]['laser']) for r in ut for sd in ('vanster', 'hoger') if r[sd]['bild'] is not None and r[sd]['laser'] is not None]
    if par:
        a = np.array(par); dif = a[:, 1] - a[:, 0]
        print(vn, 'par', len(a), 'median laser-bild', round(float(np.median(dif)), 1), 'inom 2 m', round(float(np.mean(np.abs(dif) <= 2)), 2), flush=True)
    else:
        print(vn, 'inga par', flush=True)
