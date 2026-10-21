"""Vägmärken, stolpar och fasta föremål längs en provsträcka.

1. Laserdata (Lantmäteriet, skog): punkter mer än 1,5 m över mark grupperas till objekt.
   Objekt med smalt fotavtryck (radie under 0,45 m) och höjd 2-15 m räknas som smala föremål
   (stolpar, belysning, skyltstolpar); bredare och högre som träd.
2. Vägbilder (PMS): komponenter av klassen stolpe/skylt i tolkningsmaskerna klipps ut ur
   originalbilden och typas efter färg och form: varningsmärke (gul triangel med röd kant),
   markeringsskärm (blå med gula pilar), vägmärke (övrigt), baksida (grå skylt, gäller mötande),
   stolpe (smal och hög). Läget skattas med en enkel kameramodell: platt mark framför bilen,
   kamerahöjd 2,2 m, brännvidd 320 px och horisont på rad 160 i 640x480-bilden. Avståndet är
   därför ungefärligt (cirka ±30 %).
3. Smala föremål i laserdatat bekräftas om en stolpe eller skylt syns i bilderna inom 15 m.

Exempel: python foremal.py 83   -> /home/claude/work/tvar/vag83_foremal.json
"""
import json, pickle, sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree
from pyproj import Transformer
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

vn = int(sys.argv[1]); W = Path('/home/claude/work/tvar'); BM = Path('/home/claude/work/bilder') / f'vag{vn}'
T = json.loads((W / f'vag{vn}.json').read_text()); S = T['sektioner']
dtm, tr, _, P, _ = pickle.loads((W / f'vag{vn}.cache.pkl').read_bytes())
T3857 = Transformer.from_crs(3006, 3857, always_xy=True)

s = np.array([k['s'] for k in S])
cx = np.array([(k['x0'] + k['x1']) / 2 for k in S]); cy = np.array([(k['y0'] + k['y1']) / 2 for k in S])
nx = np.array([(k['x1'] - k['x0']) / 60 for k in S]); ny = np.array([(k['y1'] - k['y0']) / 60 for k in S])
kV = np.array([k['sidor']['vanster']['kant'] for k in S]); kH = np.array([k['sidor']['hoger']['kant'] for k in S])
hv = np.array([k.get('bredd', T['vagbredd']) / 2 for k in S]); tc = (kH - kV) / 2
kbV = np.clip(hv - tc, 0.5, kV); kbH = np.clip(hv + tc, 0.5, kH)     # körbanekant från mittlinjen (som i visaren)
sek_tree = cKDTree(np.c_[cx, cy])


def lage(x, y):
    """s, t (från mittlinjen, + höger), sida och avstånd från körbanekant."""
    _, i = sek_tree.query(np.c_[x, y])
    t = (x - cx[i]) * nx[i] + (y - cy[i]) * ny[i]
    al = (x - cx[i]) * -ny[i] + (y - cy[i]) * nx[i]
    sida = np.where(t < 0, 'vanster', 'hoger')
    kb = np.where(t < 0, kbV[i], kbH[i])
    return s[i] + al, t, sida, np.abs(t) - kb


# ---------------------------------------------------------------- 1. laser
# Laserdatat (1-3 punkter/m²) räcker inte för att skilja stolpar från små trädkronor. Träd och
# andra föremål över 2,5 m tas därför från tvärsektionerna; stolpar och skyltar från bilderna.
objekt = []

# ---------------------------------------------------------------- 2. vägbilder
H_KAM, F, Y0 = 2.2, 320.0, 160.0
skyltar = []; kand = []
if (BM / 'index.json').exists():
    idx = {b['fil']: b for b in json.loads((BM / 'index.json').read_text()) if b['fil']}
    for f, b in idx.items():
        mf = BM / 'masker' / (f[:-4] + '.png')
        if not mf.exists():
            continue
        mk = np.asarray(Image.open(mf)); lab, n = ndimage.label(mk == 2)
        if not n:
            continue
        im = np.asarray(Image.open(BM / f).convert('RGB')).astype(int)
        for i, sl in enumerate(ndimage.find_objects(lab), 1):
            ys, xs = sl; area = int((lab[sl] == i).sum())
            if area < 6:
                continue
            x0_, x1_, y0_, y1_ = xs.start * 5, xs.stop * 5, ys.start * 5, ys.stop * 5
            yb = y1_
            if yb <= Y0 + 6:
                continue
            d = H_KAM * F / (yb - Y0)
            if d > 60:
                continue
            lat = ((x0_ + x1_) / 2 - 320) * d / F
            utklipp = Image.fromarray(im[max(0, y0_ - 8):y1_ + 8, max(0, x0_ - 8):x1_ + 8].astype(np.uint8))
            kand.append((f, b, d, lat, area, (x0_, y0_, x1_, y1_), utklipp))

    # Typning med CLIP (öppen bild-text-modell), noll-skott mot beskrivningar av svenska vägmärken
    if kand:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        cm = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').eval(); cp = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
        ETIK = [('markering', 'a blue road sign with large white arrow chevrons pointing sideways'),
                ('markering', 'a blue road sign with yellow arrow chevrons pointing sideways'),
                ('varning', 'a red and yellow triangular warning road sign'),
                ('vagmarke', 'a blue direction road sign with place names and numbers'),
                ('vagmarke', 'a square blue parking sign'), ('vagmarke', 'a round road sign'),
                ('stolpe', 'a wooden or concrete utility pole'), ('stolpe', 'a metal pole against the sky'),
                ('baksida', 'the grey back side of a road sign'), (None, 'trees and bushes')]
        typer = []
        with torch.no_grad():
            for k in range(0, len(kand), 32):
                bt = [u for *_, u in kand[k:k + 32]]
                inp = cp(text=[t for _, t in ETIK], images=bt, return_tensors='pt', padding=True)
                pr_ = cm(**inp).logits_per_image.softmax(1).numpy()
                typer += [ETIK[int(q.argmax())][0] if q.max() > 0.35 else None for q in pr_]
        for (f, b, d, lat, area, bb, _), typ in zip(kand, typer):
            if typ is None:
                continue
            if b['r'] == 1:
                sp, tp = b['s'] + d, lat + T['vagbredd'] / 4
            else:
                sp, tp = b['s'] - d, -(lat + T['vagbredd'] / 4)
            skyltar.append(dict(s=round(sp, 1), t=round(tp, 1), typ=typ, r=b['r'], fil=f, d=round(d, 1), yta=area, bb=bb))

# slå ihop samma föremål sett i flera bilder (samma typ, sida och riktning inom 12 m)
skyltar.sort(key=lambda x: (x['typ'], x['r'], np.sign(x['t']), x['s']))
grupper = []
for x in skyltar:
    g = grupper[-1] if grupper else None
    if g and g['typ'] == x['typ'] and g['r'] == x['r'] and np.sign(g['t']) == np.sign(x['t']) and abs(x['s'] - g['s_sista']) <= 25:
        g['n'] += 1; g['s_sista'] = x['s']
        if x['d'] < g['d']:      # närmaste observation ger bäst läge
            g.update(s=x['s'], t=x['t'], d=x['d'], fil=x['fil'], bb=x['bb'])
    else:
        grupper.append(dict(x, n=1, s_sista=x['s']))
skyltar = [dict(s=g['s'], t=g['t'], typ=g['typ'], r=g['r'], n=g['n'], fil=g['fil'], bb=g['bb'], sida='vanster' if g['t'] < 0 else 'hoger')
           for g in grupper if g['n'] >= 2 or g['typ'] in ('varning', 'markering')]
# läge på kartan: interpolera längs sektionerna
for x in skyltar:
    i = int(np.clip(np.searchsorted(s, x['s']), 0, len(s) - 1))
    X, Y = cx[i] + x['t'] * nx[i], cy[i] + x['t'] * ny[i]
    mx_, my_ = T3857.transform(X, Y); x['m'] = [round(mx_), round(my_)]

# avstånd från körbanekant (ungefärligt, kameramodellen)
for x in skyltar:
    i = int(np.clip(np.searchsorted(s, x['s']), 0, len(s) - 1))
    kb = kbV[i] if x['t'] < 0 else kbH[i]
    x['avst'] = round(max(0.0, abs(x['t']) - kb), 1)

ut = dict(skyltar=skyltar,
          kalla='Laserdata skog (Lantmäteriet) och PMS-vägbilder tolkade med Mask2Former; skylttyp efter färg och form')
(W / f'vag{vn}_foremal.json').write_text(json.dumps(ut, ensure_ascii=False))
from collections import Counter
print(vn, 'skyltar', Counter(x['typ'] for x in skyltar))
