"""Skyltregister för provsträckorna: hittade föremål med tolkad märkestyp, läge och utklipp.

Indata: vag{vn}_foremal.json (automatisk hittning i vägbilderna), skyltar/index.json och
skyltar/tolkning.py (manuell tolkning av varje utklipp i 1280x960-bilderna, med kod enligt
vägmärkesförordningen). Feltolkningar (kategori 'ej') tas bort, och samma märke sett i flera
bilder slås ihop (samma kod, riktning och sida inom 40 m).

-> /home/claude/work/tvar/vag{vn}_skyltregister.json
"""
import base64, io, json, sys
from pathlib import Path
from PIL import Image

SK = Path('/home/claude/work/skyltar'); W = Path('/home/claude/work/tvar')
K = {}
exec((SK / 'tolkning.py').read_text())          # definierar K: id -> (kategori, kod, namn)
INDEX = {r['id']: r for r in json.loads((SK / 'index.json').read_text())}
MED_BILD = {'varning', 'forbud', 'pabud', 'vajning', 'anvisning', 'vagvisning', 'markering', 'tillagg', 'baksida', 'ovrigt'}


def bild(fn):
    im = Image.open(SK / fn).convert('RGB'); im.thumbnail((140, 140))
    b = io.BytesIO(); im.save(b, 'JPEG', quality=72)
    return 'data:image/jpeg;base64,' + base64.b64encode(b.getvalue()).decode()


for vn in [int(v) for v in sys.argv[1].split(',')]:
    sk = json.loads((W / f'vag{vn}_foremal.json').read_text())['skyltar']
    ut = []
    for i, x in enumerate(sk):
        kat, kod, namn = K[f'{vn}_{i}']
        if kat == 'ej':
            continue
        y = dict(s=x['s'], t=x['t'], kat=kat, kod=kod, namn=namn, r=x['r'], sida=x['sida'], m=x['m'], avst=x['avst'],
                 n=x.get('n', 1), fil=x['fil'], id=f'{vn}_{i}')
        if kat in MED_BILD:
            y['bild'] = bild(INDEX[y['id']]['fil'])
        ut.append(y)
    # samma märke i flera bilder
    ut.sort(key=lambda y: (y['kat'], y['kod'], y['namn'], y['r'], y['sida'], y['s']))
    slut = []
    for y in ut:
        g = slut[-1] if slut else None
        if (g and g['kat'] not in ('kantstolpe', 'stolpe') and (g['kat'], g['kod'], g['namn'], g['r'], g['sida']) == (y['kat'], y['kod'], y['namn'], y['r'], y['sida'])
                and abs(y['s'] - g['s']) <= 40):
            g['n'] += y['n']
            continue
        slut.append(y)
    slut.sort(key=lambda y: y['s'])
    (W / f'vag{vn}_skyltregister.json').write_text(json.dumps(dict(
        skyltar=slut, kalla='PMS-vägbilder (Trafikverket). Föremålen hittades automatiskt (Mask2Former) och tolkades '
                            'manuellt i bilder med 1280x960 pixlar. Läget är skattat ur bilden (cirka ±30 % i avstånd).'),
        ensure_ascii=False))
    from collections import Counter
    print(vn, len(sk), '->', len(slut), dict(Counter(y['kat'] for y in slut)))
