"""Tvärsektioner längs vägen ur Lantmäteriets markhöjdmodell (1 m) och Laserdata Skog.

Inloggning till Lantmäteriet läses från miljövariablerna LM_USERNAME och LM_PASSWORD
och sparas aldrig i filer eller resultat.

För varje sektion (var 10:e meter) beräknas:
  - markprofil vinkelrätt mot vägens mittlinje, ±BREDD m
  - vägbanekanter, innerslänt (lutning 1:n), dikesdjup, ytterslänt eller bankslänt
  - närmaste föremål över 2,5 m (träd, byggnader, bergskärning ur punktmolnet) från vägkant
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import requests

STAC = "https://api.lantmateriet.se/stac-hojd/v1/search"
HALVBREDD = 30.0     # m från mittlinjen
STEG_TVAR = 0.25     # m mellan punkter i tvärled
OBJ_HOJD = 2.5       # m över mark räknas som föremål (träd, buskage över mannahöjd, byggnad)


def _auth():
    u, p = os.environ.get("LM_USERNAME"), os.environ.get("LM_PASSWORD")
    if not u or not p:
        raise RuntimeError("Sätt LM_USERNAME och LM_PASSWORD för Lantmäteriets nedladdning")
    return u, p


def stac_sok(samling, bbox_wgs84):
    r = requests.post(STAC, json=dict(collections=[samling], bbox=list(bbox_wgs84), limit=500), timeout=60)
    r.raise_for_status()
    return r.json()["features"]


# ---------------------------------------------------------------- markhöjdmodell
def las_dtm(bbox_sweref, bbox_wgs84):
    """Läs markhöjdmodellens 1 m-grid för en bbox (SWEREF 99 TM). Returnerar (z, transform)."""
    import rasterio
    from rasterio.merge import merge
    u, p = _auth()
    env = dict(GDAL_HTTP_USERPWD=f"{u}:{p}", GDAL_HTTP_AUTH="BASIC", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
               CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif", GDAL_HTTP_MAX_RETRY="10", GDAL_HTTP_RETRY_DELAY="1",
               GDAL_HTTP_RETRY_CODES="403,429,500,502,503,504",
               CURL_CA_BUNDLE=os.environ.get("CURL_CA_BUNDLE", "/root/.ccr/ca-bundle.crt"))
    items = stac_sok("dtm-cog", bbox_wgs84)
    kallor = []
    with rasterio.Env(**env):
        dsets = [rasterio.open("/vsicurl/" + it["assets"]["data"]["href"]) for it in items]
        z, tr = merge(dsets, bounds=bbox_sweref, res=1.0, nodata=np.nan, dtype="float32")
        for it, d in zip(items, dsets):
            kallor.append(dict(id=it["id"], href=it["assets"]["data"]["href"], datum=it["properties"].get("datetime")))
            d.close()
    return z[0], tr, kallor


def bilinjar(z, tr, x, y):
    col = (x - tr.c) / tr.a - 0.5
    row = (y - tr.f) / tr.e - 0.5
    c0 = np.floor(col).astype(int); r0 = np.floor(row).astype(int)
    fc = col - c0; fr = row - r0
    H, W = z.shape
    ok = (c0 >= 0) & (r0 >= 0) & (c0 + 1 < W) & (r0 + 1 < H)
    out = np.full(x.shape, np.nan)
    c0o, r0o, fco, fro = c0[ok], r0[ok], fc[ok], fr[ok]
    out[ok] = (z[r0o, c0o] * (1 - fco) * (1 - fro) + z[r0o, c0o + 1] * fco * (1 - fro)
               + z[r0o + 1, c0o] * (1 - fco) * fro + z[r0o + 1, c0o + 1] * fco * fro)
    return out


# ---------------------------------------------------------------- punktmoln
class _AuthStream:
    def __init__(self, url):
        from laspy.copc import HttpRangeStream
        self.s = HttpRangeStream(url)
        self.s.session.auth = _auth()
        sess = self.s.session
        orig = sess.get
        def get(*a, **k):
            import time
            for i in range(15):
                r = orig(*a, **k)
                if r.status_code != 403:
                    return r
                sess.close()  # ny uppkoppling -> annan server bakom lastbalanseraren
                time.sleep(0.5)
            return r
        self.s.session.get = get

    def __getattr__(self, k):
        return getattr(self.s, k)


def las_punktmoln(bbox_sweref, bbox_wgs84, korridor):
    """Hämta punkter ovan mark inom en korridor (shapely-polygon i SWEREF) ur Laserdata Skog (COPC).

    Korridoren läses i rutor om 400 m. Finns flera skanningar (olika år) används i varje ruta den
    nyaste som faktiskt har data där (minst 0,3 punkter/m² inklusive mark), så att avverkad skog
    i en äldre skanning inte ger falska hinder och glapp i en nyare skanning fylls av den äldre.
    """
    from laspy.copc import CopcReader, Bounds
    from shapely.geometry import box
    from shapely import contains_xy
    items = stac_sok("dsm-skoglig-copc", bbox_wgs84)
    items = sorted(items, key=lambda it: str(it["properties"].get("datetime", "")), reverse=True)
    delar, kallor = [], []
    x0, y0, x1, y1 = bbox_sweref
    lasare = {}
    def las(it):
        if it["id"] not in lasare:
            href = next(a["href"] for a in it["assets"].values() if a["href"].endswith(".copc.laz"))
            try:
                lasare[it["id"]] = CopcReader(_AuthStream(href))
            except Exception as e:
                kallor.append(dict(id=it["id"], fel=str(e)[:200]))
                if "403" in str(e):
                    raise RuntimeError("Laserdata Skog: åtkomst nekad (beställning ej godkänd)")
                lasare[it["id"]] = None
        return lasare[it["id"]]
    anvanda = {}
    for bx0 in np.arange(x0, x1, 400.0):
        for by0 in np.arange(y0, y1, 400.0):
            ruta = box(bx0, by0, bx0 + 400, by0 + 400).intersection(korridor)
            if ruta.is_empty:
                continue
            for it in items:
                rdr = las(it)
                if rdr is None:
                    continue
                h = rdr.header
                if h.maxs[0] < bx0 or h.mins[0] > bx0 + 400 or h.maxs[1] < by0 or h.mins[1] > by0 + 400:
                    continue
                pts = rdr.query(Bounds(mins=np.array([bx0, by0]), maxs=np.array([bx0 + 400, by0 + 400])))
                if len(pts) == 0:
                    continue
                del_ruta = ruta.intersection(box(h.mins[0], h.mins[1], h.maxs[0], h.maxs[1]))
                if del_ruta.is_empty:
                    continue
                xs = np.asarray(pts.x); ys = np.asarray(pts.y); kl = np.asarray(pts.classification)
                inne = contains_xy(del_ruta, xs, ys)
                if inne.sum() < 0.3 * del_ruta.area:
                    continue                      # skanningen saknar data här: prova en äldre
                m = inne & (kl != 2)
                delar.append(np.c_[xs[m], ys[m], np.asarray(pts.z)[m], kl[m]].astype(np.float64))
                anvanda.setdefault(it["id"], [it, 0]); anvanda[it["id"]][1] += int(m.sum())
                ruta = ruta.difference(del_ruta)  # resten av rutan kan ligga i en annan tile
                if ruta.is_empty or ruta.area < 1:
                    break
    for iid, (it, n) in anvanda.items():
        href = next(a["href"] for a in it["assets"].values() if a["href"].endswith(".copc.laz"))
        kallor.append(dict(id=iid, href=href, datum=it["properties"].get("datetime"), punkter=n))
    if not delar:
        return np.zeros((0, 4)), kallor
    P = np.vstack(delar)
    return P, kallor


# ---------------------------------------------------------------- analys av en sektion
def _lutning_text(dz, dx):
    if abs(dz) < 0.05:
        return None
    n = abs(dx / dz)
    return n


def analysera_sektion(t, z, halvbredd_vag):
    """t: avstånd från mittlinjen (negativt = vänster i färdriktning 1), z: markhöjd.

    Returnerar dict med vägkanter, slänter per sida och klassning (dike/bank/skärning).
    """
    ok = np.isfinite(z)
    if ok.sum() < len(z) * 0.6:
        return None
    zc = np.interp(0, t[ok], z[ok])
    res = dict(z_mitt=float(zc), sidor={})
    for sida, sgn in (("vanster", -1), ("hoger", 1)):
        m = (sgn * t >= 0) & ok
        u = sgn * t[m]; zz = z[m]
        o = np.argsort(u); u, zz = u[o], zz[o]
        # Vägkant: där lutningen först överstiger 12 % (brantare än skevningen) nära förväntad halvbredd
        zs = pd.Series(zz).rolling(5, center=True, min_periods=1).mean().to_numpy()
        g = np.gradient(zs, u)
        hall = pd.Series(np.abs(g)).rolling(4, min_periods=1).min().shift(-3).fillna(0).to_numpy()  # brant minst 1 m framåt
        sok = (u >= halvbredd_vag - 1.5) & (u <= halvbredd_vag + 4.5)
        # 1) släntens brytpunkt: största krökningen nära förväntad kant. Höjdmodellen rundar av
        #    vägrenen, så en ren lutningsgräns missar ofta kanten.
        gs = pd.Series(g).rolling(5, center=True, min_periods=1).mean().to_numpy()
        g2 = np.gradient(gs, u)
        ref = int(np.argmin(np.abs(u - halvbredd_vag)))
        fram = int(np.argmin(np.abs(u - (halvbredd_vag + 3.0))))
        dz = zs[fram] - zs[ref]
        metod = "pms"; ik = ref
        if sok.any() and abs(dz) > 0.12:
            idx = np.where(sok)[0]
            j = idx[np.argmin(g2[idx])] if dz < 0 else idx[np.argmax(g2[idx])]
            # det ska börja en verklig slänt efter brytpunkten (minst 6 % lutning 1 m ut)
            j1 = min(len(u) - 1, j + int(1.0 / STEG_TVAR))
            if abs(zs[j1] - zs[j]) / max(u[j1] - u[j], 1e-6) > 0.06 and abs(g2[j]) > 0.02:
                ik = int(j); metod = "brytpunkt"
        if metod == "pms":
            kand = np.where(sok & (np.abs(g) > 0.12) & (hall > 0.08))[0]
            if len(kand):
                ik = int(kand[0]); metod = "lutning"
        kant_u, kant_z = float(u[ik]), float(zz[ik])
        ut = u >= kant_u
        uu, zu = u[ut], zz[ut]
        if len(uu) < 8:
            res["sidor"][sida] = dict(kant=kant_u, typ="okänd"); continue
        # Dikesbotten: lägsta punkt inom 8 m från kanten
        inom = uu <= kant_u + 8
        ib = int(np.argmin(np.where(inom, zu, np.inf)))
        botten_u, botten_z = float(uu[ib]), float(zu[ib])
        djup = kant_z - botten_z
        # Är det bank (marken fortsätter neråt) eller dike (stiger igen)?
        efter = zu[ib:]; efter_u = uu[ib:]
        stig = (efter.max() - botten_z) if len(efter) else 0
        sida_res = dict(kant=kant_u, kant_z=kant_z, kant_metod=metod)
        if djup < 0.15 and np.median(np.gradient(zs[ut], uu)[:20]) > 0.1:
            typ = "skärning"
        elif stig < 0.2 and djup > 0.3:
            typ = "bank"
        elif djup >= 0.15:
            typ = "dike"
        else:
            typ = "flackt"
        sida_res["typ"] = typ
        if typ == "bank":
            # Bankslänt: följ marken nedåt tills den planar ut (flackare än 1:8 över 2 m), högst 25 m
            gu = np.gradient(zs[ut], uu)
            w = int(2 / STEG_TVAR)
            flat = pd.Series(np.abs(gu)).rolling(w, min_periods=1).max().shift(-w + 1).fillna(1).to_numpy()
            kand = np.where((uu > kant_u + 1.0) & (flat < 1 / 8) & (uu <= kant_u + 25))[0]
            jb = kand[0] if len(kand) else int(np.argmin(np.where(uu <= kant_u + 25, zu, np.inf)))
            botten_u, botten_z = float(uu[jb]), float(zu[jb])
            djup = kant_z - botten_z
        if typ in ("dike", "bank"):
            # innerslänt: från kant till botten (eller till 2/3 av bankhöjden)
            sida_res["inner"] = dict(u0=kant_u, z0=kant_z, u1=botten_u, z1=botten_z,
                                     n=_lutning_text(kant_z - botten_z, botten_u - kant_u))
            sida_res["dikesdjup" if typ == "dike" else "bankhojd"] = float(djup)
        if typ == "dike" and len(efter) > 4:
            # ytterslänt: från botten till där marken planar ut (lutning < 1:6) eller 10 m
            ge = np.gradient(efter, efter_u)
            slut = np.where((efter_u > botten_u + 1.0) & (np.abs(ge) < 1 / 6))[0]
            je = slut[0] if len(slut) else min(len(efter) - 1, int(10 / STEG_TVAR))
            sida_res["ytter"] = dict(u0=botten_u, z0=botten_z, u1=float(efter_u[je]), z1=float(efter[je]),
                                     n=_lutning_text(efter[je] - botten_z, efter_u[je] - botten_u))
        if typ == "skärning":
            j = min(len(uu) - 1, int(6 / STEG_TVAR))
            sida_res["ytter"] = dict(u0=kant_u, z0=kant_z, u1=float(uu[j]), z1=float(zu[j]),
                                     n=_lutning_text(zu[j] - kant_z, uu[j] - kant_u))
        res["sidor"][sida] = sida_res
    return res


def sektioner(ref, mittlinje, dtm, tr, punkter, s0, s1, vagbredd=7.0, steg=10.0):
    """Beräkna tvärsektioner mellan s0 och s1 (riktning 1:s längdmätning)."""
    t = np.arange(-HALVBREDD, HALVBREDD + 1e-9, STEG_TVAR)
    from scipy.spatial import cKDTree
    tree = cKDTree(punkter[:, :2]) if len(punkter) else None
    ut = []
    for s in np.arange(s0, s1 + 1e-9, steg):
        cx, cy, tx, ty = mittlinje(s)
        nx, ny = ty, -tx                     # normal åt höger i färdriktning 1
        X = cx + t * nx; Y = cy + t * ny
        z = bilinjar(dtm, tr, X, Y)
        a = analysera_sektion(t, z, vagbredd / 2)
        if a is None:
            continue
        sek = dict(s=float(s), x0=float(X[0]), y0=float(Y[0]), x1=float(X[-1]), y1=float(Y[-1]),
                   t=t[::2].round(2).tolist(), z=[None if not np.isfinite(v) else round(float(v), 2) for v in z[::2]],
                   **a)
        # Föremål över OBJ_HOJD m inom ±5 m längs vägen
        if tree is not None:
            idx = tree.query_ball_point([cx, cy], r=HALVBREDD + 6)
            if idx:
                P = punkter[idx]
                dx, dy = P[:, 0] - cx, P[:, 1] - cy
                along = dx * tx + dy * ty; tv = dx * nx + dy * ny
                m = (np.abs(along) <= 5) & (np.abs(tv) <= HALVBREDD)
                P, tv = P[m], tv[m]
                mark = bilinjar(dtm, tr, P[:, 0], P[:, 1])
                hojd = P[:, 2] - mark
                # vegetationssilhuett: högsta punkt per 0,5 m i tvärled
                fack = np.round(tv / 0.5) * 0.5
                sil = pd.DataFrame(dict(t=fack, h=hojd, z=P[:, 2])).groupby("t").max().reset_index()
                sil = sil[sil.h > 0.5]
                sek["veg"] = [[float(a), round(float(b), 1), round(float(c), 2)] for a, b, c in sil[["t", "h", "z"]].to_numpy()]
                for sida, sgn in (("vanster", -1), ("hoger", 1)):
                    sd = a["sidor"].get(sida)
                    if not sd:
                        continue
                    kant = sd["kant"]
                    mm = (sgn * tv > kant) & (hojd >= OBJ_HOJD)
                    if mm.any():
                        avst = (sgn * tv[mm] - kant)
                        k = int(np.argmin(avst))
                        sd["foremal"] = dict(avstand=round(float(avst[k]), 1), hojd=round(float(hojd[mm][k]), 1),
                                             t=round(float(tv[mm][k]), 2))
        ut.append(sek)
    return ut
