"""Tolka PMS-vägbilder med en öppen segmenteringsmodell (Mask2Former, Mapillary Vistas).

För varje bild räknas andelen pixlar per klassgrupp (räcke, stolpe, skylt, vegetation)
i närområdet till vänster och höger om vägen. Dessutom sparas en liten bild med
räcken och stolpar markerade, för visning i tvärsektionsvisaren.

Exempel: python tolka_vagbilder.py /home/claude/work/bilder/vag83
"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

MAPP = Path(sys.argv[1])
MODELL = "facebook/mask2former-swin-large-mapillary-vistas-semantic"
torch.set_num_threads(2)
proc = AutoImageProcessor.from_pretrained(MODELL)
model = Mask2FormerForUniversalSegmentation.from_pretrained(MODELL).eval()
lab = {int(k): v.lower() for k, v in model.config.id2label.items()}
GRUPPER = {
    "racke": [i for i, n in lab.items() if "guard rail" in n or "guard-rail" in n],
    "stolpe": [i for i, n in lab.items() if n in ("pole", "utility pole", "street light") or "support--pole" in n or "utility-pole" in n or "street-light" in n],
    "skylt": [i for i, n in lab.items() if "traffic sign" in n or "traffic-sign" in n or "sign frame" in n],
    "vegetation": [i for i, n in lab.items() if n in ("vegetation",) or "nature--vegetation" in n],
    "barriar": [i for i, n in lab.items() if n in ("barrier", "fence", "wall") or "construction--barrier--fence" in n or "construction--barrier--wall" in n],
}
print({g: [lab[i] for i in ids] for g, ids in GRUPPER.items()}, flush=True)
FARG = {"racke": (255, 140, 0), "stolpe": (230, 0, 200), "skylt": (230, 0, 200), "barriar": (255, 210, 0)}

ut_mapp = MAPP / "tolkade"; ut_mapp.mkdir(exist_ok=True)
mask_mapp = MAPP / "masker"; mask_mapp.mkdir(exist_ok=True)
index = json.loads((MAPP / "index.json").read_text())
res_fil = MAPP / "tolkning.json"
res = json.loads(res_fil.read_text()) if res_fil.exists() else {}
t0 = time.time()
for k, b in enumerate(index):
    if not b["fil"] or (b["fil"] in res and (mask_mapp / (b["fil"][:-4] + ".png")).exists()):
        continue
    im = Image.open(MAPP / b["fil"]).convert("RGB")
    W, H = im.size
    with torch.no_grad():
        inp = proc(images=im, return_tensors="pt")
        out = model(**inp)
    seg = proc.post_process_semantic_segmentation(out, target_sizes=[(H, W)])[0].numpy()
    # Närområde: under horisonten (nedre 58 %), vänster resp. höger tredjedel
    y0 = int(H * 0.42)
    zon = {"vanster": (slice(y0, H), slice(0, int(W * 0.38))), "hoger": (slice(y0, H), slice(int(W * 0.62), W))}
    r = {}
    for sida, (ys, xs) in zon.items():
        z = seg[ys, xs]
        r[sida] = {g: round(float(np.isin(z, ids).mean()), 4) for g, ids in GRUPPER.items()}
        # stolpar/skyltar: antal kolumner med vertikala stolpsegment (grov räkning av objekt)
        st = np.isin(z, GRUPPER["stolpe"] + GRUPPER["skylt"])
        kol = st.sum(0) > 6
        r[sida]["stolpobjekt"] = int(np.sum(np.diff(np.r_[0, kol.astype(int)]) == 1))
    res[b["fil"]] = r
    # liten etikettbild (128x96): 1 räcke, 2 stolpe/skylt, 3 staket/barriär, för överlägg på originalbilden
    lb = np.zeros_like(seg, dtype=np.uint8)
    lb[np.isin(seg, GRUPPER["barriar"])] = 3; lb[np.isin(seg, GRUPPER["stolpe"] + GRUPPER["skylt"])] = 2; lb[np.isin(seg, GRUPPER["racke"])] = 1
    Image.fromarray(lb).resize((128, 96), Image.NEAREST).save(mask_mapp / (b["fil"][:-4] + ".png"))
    # liten bild med markeringar
    a = np.asarray(im).astype(np.float32)
    for g, c in FARG.items():
        m = np.isin(seg, GRUPPER[g])
        a[m] = a[m] * 0.35 + np.array(c) * 0.65
    Image.fromarray(a.clip(0, 255).astype(np.uint8)).resize((384, 288), Image.LANCZOS).save(ut_mapp / b["fil"], quality=62)
    if k % 10 == 0:
        res_fil.write_text(json.dumps(res))
        print(k, len(index), f"{(time.time()-t0)/max(1,len(res)):.1f} s/bild", flush=True)
res_fil.write_text(json.dumps(res))
print("klar", len(res))
