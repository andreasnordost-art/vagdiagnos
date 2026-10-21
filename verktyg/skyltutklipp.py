"""Hämta vägbilden i hög upplösning (1280x960) för varje hittad skylt och klipp ut den.

-> /home/claude/work/skyltar/vag{vn}_{i}.png samt kontaktark för granskning.
"""
import json, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests
from PIL import Image, ImageDraw

BAS = "https://pmsv4.trafikverket.se/internal/api/image"
UT = Path('/home/claude/work/skyltar'); HR = UT / 'hog'; HR.mkdir(parents=True, exist_ok=True)
VAGAR = [int(v) for v in sys.argv[1].split(',')]


def hog(vn, b):
    f = HR / f"vag{vn}_{b['fil']}"
    if not f.exists():
        for k in range(4):
            q = requests.get(BAS, params=dict(lankod=22, vagnummer=vn, vagundernummer=0, riktning=b['r'], korfalt=10,
                                              lopandeLangd=b['ll'], periodDatum=b['datum'], imageSize=3), timeout=90)
            if q.status_code == 200 and q.content:
                f.write_bytes(q.content); break
            time.sleep(2)
    return f


rader = []
for vn in VAGAR:
    idx = {b['fil']: b for b in json.loads(Path(f'/home/claude/work/bilder/vag{vn}/index.json').read_text()) if b['fil']}
    sk = json.loads(Path(f'/home/claude/work/tvar/vag{vn}_foremal.json').read_text())['skyltar']
    with ThreadPoolExecutor(4) as ex:
        filer = list(ex.map(lambda x: hog(vn, idx[x['fil']]), sk))
    for i, (x, f) in enumerate(zip(sk, filer)):
        if not f.exists():
            continue
        im = Image.open(f).convert('RGB'); x0, y0, x1, y1 = [2 * v for v in x['bb']]
        w, h = x1 - x0, y1 - y0; m = max(24, int(0.6 * max(w, h)))
        u = im.crop((max(0, x0 - m), max(0, y0 - m), min(1280, x1 + m), min(960, y1 + m)))
        u.thumbnail((220, 220)); fn = UT / f'vag{vn}_{i}.png'; u.save(fn)
        rader.append(dict(id=f'{vn}_{i}', vag=vn, i=i, typ=x['typ'], r=x['r'], s=x['s'], fil=fn.name))
    print(vn, len(sk), flush=True)
(UT / 'index.json').write_text(json.dumps(rader, ensure_ascii=False))
# kontaktark, 24 per ark
for k in range(0, len(rader), 24):
    ark = Image.new('RGB', (6 * 230, 4 * 250), 'white'); d = ImageDraw.Draw(ark)
    for j, rr in enumerate(rader[k:k + 24]):
        u = Image.open(UT / rr['fil']); cx, cy = (j % 6) * 230, (j // 6) * 250
        ark.paste(u, (cx + 5, cy + 22)); d.text((cx + 5, cy + 4), f"{rr['id']} {rr['typ']}", fill='black')
    ark.save(UT / f'ark_{k // 24:02d}.png')
print('ark', (len(rader) + 23) // 24)
