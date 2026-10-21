"""Standardnivå per 20-metersavsnitt.

För varje avsnitt och hastighet bedöms fyra kontroller: horisontalradie,
skevning (tvärfall i kurva), vertikalradie och stoppsikt över krön.
Varje kontroll ger OK, BRIST eller OSÄKER utifrån mätvärdets
osäkerhetsintervall (90 %):
    OK      hela intervallet klarar kravet
    BRIST   hela intervallet underskrider kravet
    OSÄKER  intervallet omsluter kravet

Två standarder redovisas:
    ny   nybyggnad/förbättring: riktvärden för radier, gränsvärde stoppsikt (fig. 8.1.5.1.5-1)
    om   ombyggnad med befintlig linjeföring: gränsvärden för radier, stoppsikt fig. 8.1.5.1.5-3
Säkerställd nivå = högsta hastighet där alla kontroller är OK (även alla lägre).
Möjlig nivå      = högsta hastighet utan någon BRIST (även alla lägre).
"""
from __future__ import annotations


import numpy as np
import pandas as pd

from . import vgu

HASTIGHETER = [40, 50, 60, 70, 80, 90, 100, 110, 120]
OK, BRIST, OSAKER, EJ = "OK", "BRIST", "OSÄKER", "–"
DOMANER = ["horisontal", "skevning", "vertikal", "sikt"]
NAMN = {"horisontal": "Horisontalradie", "skevning": "Skevning", "vertikal": "Vertikalradie",
        "sikt": "Stoppsikt (krön)"}
RANK = {BRIST: 0, OSAKER: 1, OK: 2, EJ: 3}


def _status(lag, hog, krav):
    if krav is None or not np.isfinite(lag):
        return EJ
    if lag >= krav:
        return OK
    if hog < krav:
        return BRIST
    return OSAKER


def _varst(a, b):
    return a if RANK[a] <= RANK[b] else b


GNSS_MAX_RMS = 0.5   # m; cirkelanpassningen till GNSS-spåret räknas som pålitlig under detta
# Kalibrering av motorns 90 %-intervall för horisontalradier (enkla kurvor) mot GNSS-spåret:
# intervallen täckte 97,5 % i stället för 90 % (444 kurvor), 90 %-kvantilen av |fel|/halvbredd var 0,66.
# Faktorn 0,7 ger 90 % täckning med liten marginal. Sammansatta kurvor kalibreras inte (sämre facit).
K_HORISONTAL = 0.70   # (ersatt av radieberoende faktorer nedan)
def k_horisontal(R):
    """Kalibreringsfaktor per radieklass (90 %-kvantilen av |fel|/halvbredd mot GNSS: 0,58-0,62 under
    1 000 m, 0,79 för 1 000-2 500 m). Över 2 500 m lämnas intervallen orörda."""
    return 0.62 if R < 1000 else 0.80 if R < 2500 else 1.0
# Empiriska standardavvikelser (log-radie) för GNSS-cirkelanpassningen per radieklass, med golv 0,02
def sd_gnss(R):
    return 0.02 if R < 1000 else 0.047 if R < 2500 else 0.133


BANA_MAX_RMS = 1.0    # m, GNSS-banans avvikelse; större betyder att kurvformen inte stämmer
BANA_MAX_REL = 0.10
SD_BANA = 0.03        # lägsta osäkerhet (log-radie) för banans radie; jämförelse mot cirkeln gav 1 %


def vidga_osakerhet(hor, gnss=None, flerar=None, bana=None):
    """Kvalitetssäkring: osäkerhetsintervallet ska omfatta alla oberoende skattningar av radien.

    Skattningarna är riktning 1, riktning 2 och (om den finns och är pålitlig) en cirkel anpassad
    till mätbilens GNSS-spår. Om någon av dem ligger utanför intervallet vidgas det dit, med 3 %
    marginal. Det gör att en kurva där källorna är oense redovisas som osäker i stället för klarar.
    gnss: dict kurva_nr -> (R_gnss, rms).
    flerar: dict kurva_nr -> radier från varje mätår 2016-2025 (riktning 1). Kontrollen mot GNSS visade
    att medelvärdet över åren inte är säkrare än motorns skattning (spridningen mellan åren är bara
    ca 2-3 %), så åren vägs inte in i radien. Men intervallet ska omfatta årsmedianen, och kurvor där åren
    är oense (robust spridning över 15 %, R under 2 500 m) flaggas.
    bana: dict kurva_nr -> (R_bana, rel_osäkerhet, rms_m) ur vagdiagnos.bana. Godtas om banan följer
    GNSS-punkterna inom 1 m (rms) och radien är bestämd inom 10 %. Vägs in där cirkeln inte gjort det.
    """
    if hor.attrs.get("vidgad"):
        return hor               # redan kalibrerad och vidgad (funktionen anropas från flera ställen)
    hor = hor.copy()
    for i, c in hor.iterrows():
        enkel = "sammansatt" not in str(c.get("flaggor", ""))
        if enkel and c.radie > 0:
            k = k_horisontal(c.radie)
            hor.at[i, "radie_lag"] = c.radie * (c.radie_lag / c.radie) ** k
            hor.at[i, "radie_hog"] = c.radie * (c.radie_hog / c.radie) ** k
            # Sammanvägning med GNSS-spårets cirkelradie (log-medel, vikt efter osäkerhet, högst 0,5)
            if gnss and c.kurva_nr in gnss:
                rg, rms = gnss[c.kurva_nr]
                if np.isfinite(rg) and rg > 0 and np.isfinite(rms) and rms < GNSS_MAX_RMS:
                    sd_m = np.log(hor.at[i, "radie_hog"] / hor.at[i, "radie_lag"]) / 2 / 1.645
                    w = min(0.5, sd_m ** 2 / (sd_m ** 2 + sd_gnss(c.radie) ** 2))
                    ny = float(np.exp((1 - w) * np.log(c.radie) + w * np.log(rg)))
                    f = ny / c.radie
                    hor.at[i, "radie_motor"] = c.radie
                    hor.at[i, "radie"] = ny; hor.at[i, "radie_lag"] *= f; hor.at[i, "radie_hog"] *= f
            c = hor.loc[i]
        # GNSS-banan (vagdiagnos.bana): motorns kurvform anpassad till koordinaterna. Används där
        # cirkelanpassningen saknas eller är svag: sammansatta kurvor och korta cirkelbågar.
        rb = None
        if bana and c.kurva_nr in bana:
            Rb, relb, rmsb = bana[c.kurva_nr]
            if np.isfinite(Rb) and Rb > 0 and np.isfinite(relb) and relb < BANA_MAX_REL and rmsb < BANA_MAX_RMS:
                rb = Rb
        fusad = "radie_motor" in hor.columns and np.isfinite(hor.at[i, "radie_motor"])
        if rb is not None and not fusad:
            sd_m = np.log(c.radie_hog / c.radie_lag) / 2 / 1.645
            sd_b = max(bana[c.kurva_nr][1], SD_BANA)
            w = min(0.5, sd_m ** 2 / (sd_m ** 2 + sd_b ** 2))
            ny = float(np.exp((1 - w) * np.log(c.radie) + w * np.log(rb)))
            f = ny / c.radie
            hor.at[i, "radie_motor"] = c.radie; hor.at[i, "radie_bana"] = rb
            hor.at[i, "radie"] = ny; hor.at[i, "radie_lag"] = c.radie_lag * f; hor.at[i, "radie_hog"] = c.radie_hog * f
            c = hor.loc[i]
        sk = [x for x in (c.get("radie_r1"), c.get("radie_r2")) if x is not None and np.isfinite(x) and x > 0]
        if rb is not None:
            sk.append(rb)
        if gnss and c.kurva_nr in gnss:
            rg, rms = gnss[c.kurva_nr]
            if np.isfinite(rg) and np.isfinite(rms) and rms < GNSS_MAX_RMS and "sammansatt" not in str(c.get("flaggor", "")):
                sk.append(rg)
        if flerar and c.kurva_nr in flerar and "sammansatt" not in str(c.get("flaggor", "")):
            ra = np.asarray(flerar[c.kurva_nr], float); ra = ra[np.isfinite(ra) & (ra > 0)]
            if len(ra) >= 3:
                lr = np.log(ra); med = float(np.median(lr)); sd = 1.4826 * float(np.median(np.abs(lr - med)))
                hor.at[i, "radie_flerar"] = float(np.exp(med)); hor.at[i, "n_ar"] = len(ra); hor.at[i, "sd_ar"] = sd
                sk.append(float(np.exp(med)))
                if sd > 0.15 and c.radie < 2500:
                    fl = c.get("flaggor", ""); fl = fl if isinstance(fl, str) else ""
                    hor.at[i, "flaggor"] = (fl + "; " if fl else "") + "radien varierar mellan mätåren"
        if not sk:
            continue
        hor.at[i, "radie_lag"] = min(c.radie_lag, 0.97 * min(sk))
        hor.at[i, "radie_hog"] = max(c.radie_hog, 1.03 * max(sk))
    hor.attrs["vidgad"] = True
    return hor


def fusera_vertikal(vert, vdtm=None):
    """Väg samman motorns vertikalradie med höjdmodellens (provsträckor), invers varians i log-radie.

    vdtm: dict vertikal_nr -> (Rv_dtm, Rv_dtm_lag, Rv_dtm_hog). Bara kurvor där höjdmodellen ger samma
    kurvtyp används. Båda intervallen är 90 %; kontrollen visade att de tillsammans täcker 92 %.
    """
    if vert.attrs.get("fuserad") or not vdtm:
        return vert
    vert = vert.copy()
    for i, c in vert.iterrows():
        q = vdtm.get(int(c.vertikal_nr))
        if not q or c.radie <= 0 or c.radie_lag <= 0:
            continue
        rd, rl, rh = q
        sm = np.log(c.radie_hog / c.radie_lag) / 2 / 1.645; sd = np.log(rh / rl) / 2 / 1.645
        if not (np.isfinite(sm) and np.isfinite(sd)) or sm <= 0 or sd <= 0:
            continue
        w = sd ** 2 / (sm ** 2 + sd ** 2)                      # vikt för motorn
        mu = w * np.log(c.radie) + (1 - w) * np.log(rd); sf = 1 / np.sqrt(1 / sm ** 2 + 1 / sd ** 2)
        vert.at[i, "radie_pms"] = c.radie
        vert.at[i, "radie"] = float(np.exp(mu))
        vert.at[i, "radie_lag"] = float(np.exp(mu - 1.645 * sf)); vert.at[i, "radie_hog"] = float(np.exp(mu + 1.645 * sf))
    vert.attrs["fuserad"] = True
    return vert


def markera_takfall(hor, tp=None):
    """Flagga kurvor där PMS 6,4 m-tvärprofil visar takfall (krön vid mittlinjen) i kurvans kärna.

    Kontrollen på provsträckorna visade att exportens Tvärfall redan avser mätbilens eget körfält
    (median skillnad 0,2-0,5 procentenheter mot profilens fall i körfältet), så tvärfallsvärdet
    rättas inte. Men exporten visar inte om motsatt körfält lutar åt andra hållet. Takfall i en kurva
    betyder att vägbanan inte är skevad över hela bredden.
    tp: dict 'r1'/'r2' -> DataFrame med s, t (riktning 1:s längdmätning), takfall.
    """
    if hor.attrs.get("takfall") or not tp or not len(hor):
        return hor
    hor = hor.copy()
    for i, c in hor.iterrows():
        ks, ke = c.karna_start, c.karna_slut
        if ke - ks < 40:
            mid = (ks + ke) / 2; ks, ke = mid - 20, mid + 20
        andelar = []
        for namn in ("r1", "r2"):
            d = tp.get(namn)
            if d is None or not len(d) or d.s.min() > ks or d.t.max() < ke:
                continue          # kurvan ligger inte helt inom provsträckan
            q = d[(d.t > ks) & (d.s < ke)]
            if len(q) >= 2:
                andelar.append(float(q.takfall.mean()))
        if andelar:
            hor.at[i, "takfall"] = max(andelar)
            if max(andelar) >= 0.5:
                fl = c.get("flaggor", ""); fl = fl if isinstance(fl, str) else ""
                hor.at[i, "flaggor"] = (fl + "; " if fl else "") + "takfall i kurvan (ej skevad över hela bredden)"
    hor.attrs["takfall"] = True
    return hor


def kontroller(res):
    """Returnerar lista med (domän, start, slut, V, standard, status, värde, krav, objekt)."""
    ut = []
    hor = vidga_osakerhet(res.horisontal, getattr(res, "gnss", None), getattr(res, "flerar", None), getattr(res, "bana", None))
    tv = res.tvarfall
    h = res.hastighet
    def vagtyp(m):
        q = h[(h.start <= m) & (h.slut > m)].vagtyp.astype(str)
        return q.iloc[0] if len(q) else ""
    # Horisontalradie
    for c in hor.itertuples():
        vt = vagtyp((c.karna_start + c.karna_slut) / 2)
        motesfri = ("mötesfri" in vt) or vt.startswith("Motor")
        for V in HASTIGHETER:
            rikt, grans = vgu.horisontal_krav(V)
            tdok = vgu.tdok_radie(V, motesfri)
            for std, krav in (("ny", rikt), ("om", grans), ("tdok", tdok)):
                ut.append(("horisontal", c.start, c.slut, V, std, _status(c.radie_lag, c.radie_hog, krav),
                           c.radie, krav, f"H{c.kurva_nr}"))
    # Skevning (samma minimikrav i båda standarderna; normalt val redovisas separat)
    if len(tv):
        t = hor.merge(tv, on="kurva_nr")
        h = res.hastighet
        def fyrfalt(m):
            q = h[(h.start <= m) & (h.slut > m)].vagtyp.astype(str)
            return bool(len(q)) and q.str.startswith("Motor").any()
        for c in t.itertuples():
            ff = fyrfalt((c.karna_start + c.karna_slut) / 2)
            e = c.tvarfall_in_min; u = 1.645 * c.u_tvarfall
            ks, ke = c.karna_start, c.karna_slut
            if ke - ks < 40:
                m = (ks + ke) / 2; ks, ke = m - 20, m + 20
            for V in HASTIGHETER:
                if V < 60 or not np.isfinite(e):
                    st = EJ; krav = None
                else:
                    kv_varst, _ = vgu.skevning_krav(c.radie_lag, V, ff)
                    kv_bast, _ = vgu.skevning_krav(c.radie_hog, V, ff)
                    krav = kv_varst
                    if kv_varst is None or kv_bast is None:
                        st = EJ  # radien under figurens band: horisontalkontrollen fångar det
                    elif e - u >= kv_varst:
                        st = OK
                    elif e + u < kv_bast:
                        st = BRIST
                    else:
                        st = OSAKER
                for std in ("ny", "om"):
                    ut.append(("skevning", ks, ke, V, std, st, e, krav, f"H{c.kurva_nr}"))
    # Vertikalradie
    res.vertikal = fusera_vertikal(res.vertikal, getattr(res, "vdtm", None))
    for c in res.vertikal.itertuples():
        for V in HASTIGHETER:
            for std, idx in (("ny", 0), ("om", 1)):
                if c.typ == "konvex":
                    kr = vgu.vertikal_krav("konvex_kort", V)
                    krav = kr[idx] if kr[idx] is not None else kr[0]
                    st = _status(c.radie_lag, c.radie_hog, krav)
                else:
                    ob = vgu.vertikal_krav("konkav_obelyst", V)
                    k_ob = ob[idx] if ob[idx] is not None else ob[0]
                    # Antagande: vägarna saknar belysning, så kravet för obelyst svacka gäller
                    krav = k_ob
                    st = _status(c.radie_lag, c.radie_hog, k_ob) if k_ob is not None else EJ
                ut.append(("vertikal", c.start, c.slut, V, std, st, c.radie, krav, f"V{c.vertikal_nr}"))
    return ut


def sikt_status(res):
    """Status per ögonpunkt (10 m) och hastighet/standard för stoppsikt."""
    s = res.sikt.copy()
    rows = []
    for V in HASTIGHETER:
        for std, variant in (("ny", "grans_nybyggnad"), ("om", "grans_ombyggnad"), ("tdok", None)):
            if std == "tdok":
                k = vgu.tdok_stoppsikt(V)
                if k is None:
                    continue
                krav = np.full(len(s), float(k))
            else:
                var = variant if (variant == "grans_nybyggnad" or V >= 60) else "grans_nybyggnad"
                krav = np.array([vgu.stoppsikt(V, g, var) for g in s.lutning_fram])
            u = 0.05 * krav + 5
            st = np.where(~np.isfinite(s.tillganglig) | s.vid_vagslut, EJ,
                          np.where(s.tillganglig >= krav + u, OK,
                                   np.where(s.tillganglig < krav - u, BRIST, OSAKER)))
            rows.append(pd.DataFrame(dict(s=s.s, riktning=s.riktning, V=V, std=std, status=st,
                                          tillganglig=s.tillganglig, krav=krav)))
    return pd.concat(rows, ignore_index=True)


def celler(res, cell=20.0):
    L = res.langd
    starts = np.arange(0, L, cell)
    n = len(starts)
    # skyltad hastighet
    h = res.hastighet.dropna(subset=["hastighet"])
    mid = starts + cell / 2
    skyltad = np.full(n, np.nan)
    for r in h.itertuples():
        skyltad[(mid >= r.start) & (mid < r.slut)] = r.hastighet
    # data-täckning
    prof = res.profil
    data = np.interp(mid, prof.s, prof.data.astype(float), left=0, right=0) > 0.5

    stat = {(d, std, V): np.full(n, EJ, dtype=object) for d in DOMANER for std in ("ny", "om", "tdok") for V in HASTIGHETER}
    objekt = {(d, std, V): np.full(n, "", dtype=object) for d in DOMANER for std in ("ny", "om", "tdok") for V in HASTIGHETER}
    for dom, a, b, V, std, st, varde, krav, obj in kontroller(res):
        i0 = int(max(np.floor(a / cell), 0)); i1 = int(min(np.ceil(b / cell), n))
        arr = stat[(dom, std, V)]; oa = objekt[(dom, std, V)]
        for i in range(i0, i1):
            if RANK[st] < RANK[arr[i]]:
                arr[i] = st; oa[i] = obj
            elif arr[i] == EJ:
                arr[i] = st; oa[i] = obj
    ss = sikt_status(res)
    ss["cell"] = np.clip((ss.s // cell).astype(int), 0, n - 1)
    for (V, std), g in ss.groupby(["V", "std"]):
        arr = stat[("sikt", std, V)]
        for c, st in zip(g.cell.to_numpy(), g.status.to_numpy()):
            arr[c] = _varst(arr[c], st) if arr[c] != EJ else st

    ut = pd.DataFrame(dict(start=starts, slut=np.minimum(starts + cell, L), skyltad=skyltad, data=data))
    for std in ("ny", "om", "tdok"):
        saker = np.full(n, np.nan); mojlig = np.full(n, np.nan)
        begr = np.full(n, "", dtype=object); begr_obj = np.full(n, "", dtype=object)
        s_ok = np.ones(n, bool); m_ok = np.ones(n, bool)
        for V in HASTIGHETER:
            allok = np.ones(n, bool); ingenbrist = np.ones(n, bool)
            for d in DOMANER:
                a = stat[(d, std, V)]
                allok &= np.isin(a, [OK, EJ]); ingenbrist &= a != BRIST
            s_ok &= allok; m_ok &= ingenbrist
            saker[s_ok] = V; mojlig[m_ok] = V
        # begränsande faktor vid första ej klarade hastighet
        for i in range(n):
            if not data[i]:
                continue
            nasta = (saker[i] + 10) if np.isfinite(saker[i]) else HASTIGHETER[0]
            if nasta > HASTIGHETER[-1]:
                continue
            fakt = [(NAMN[d], stat[(d, std, nasta)][i], objekt[(d, std, nasta)][i]) for d in DOMANER
                    if stat[(d, std, nasta)][i] in (BRIST, OSAKER)]
            fakt.sort(key=lambda x: RANK[x[1]])
            begr[i] = ", ".join(f"{a}{' (osäker)' if b == OSAKER else ''}" for a, b, _ in fakt)
            begr_obj[i] = ", ".join(o for _, _, o in fakt if o)
        saker[~data] = np.nan; mojlig[~data] = np.nan
        ut[f"saker_{std}"] = saker; ut[f"mojlig_{std}"] = mojlig
        ut[f"begransning_{std}"] = begr; ut[f"objekt_{std}"] = begr_obj
        # status vid skyltad hastighet
        sk = np.full(n, "", dtype=object)
        for i in range(n):
            V = skyltad[i]
            if not np.isfinite(V) or not data[i]:
                continue
            V = int(min(max(round(V / 10) * 10, 40), 120))
            st = OK
            for d in DOMANER:
                st = _varst(st, stat[(d, std, V)][i]) if stat[(d, std, V)][i] != EJ else st
            sk[i] = st
        ut[f"vid_skyltad_{std}"] = sk
        for d in DOMANER:
            ut[f"{d}_{std}_skyltad"] = [stat[(d, std, int(min(max(round(V / 10) * 10, 40), 120)))][i]
                                        if np.isfinite(V) else EJ for i, V in enumerate(skyltad)]
    return ut, stat
