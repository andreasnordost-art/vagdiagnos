"""Hämtning och inläsning av PMSv4-data (Trafikverket).

Varje hämtad fil sparas oförändrad med ett kvitto (<fil>.source.json) som
anger URL, metod, anropskropp, tidpunkt, HTTP-status och SHA-256. Inläsningen
kontrollerar alltid kvittot innan filen används.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

BAS = "https://pmsv4.trafikverket.se/internal/api/"

# Egenskaper som begärs i exporten (namn enligt PMS data/dataproperties)
EGENSKAPER = ["Kurvatur", "MatDatum", "PeriodDatum", "Tvarfall", "Tvarfallsvariation",
              "Backighet", "Pausad", "Hastighet", "VagTyp", "Vagroll", "Lankroll",
              "Europavag", "Trafik", "VagBredd", "AntalKorfalt"]


class KvittoFel(RuntimeError):
    pass


def _kvitto(path: Path) -> Path:
    return path.with_name(path.name + ".source.json")


def kontrollera(path: Path) -> dict:
    """Kontrollera fil mot kvitto. Kastar KvittoFel vid avvikelse."""
    k = _kvitto(path)
    if not k.exists():
        raise KvittoFel(f"Kvitto saknas: {k}")
    info = json.loads(k.read_text())
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    if h != info.get("sha256"):
        raise KvittoFel(f"Kontrollsumman stämmer inte: {path}")
    return info


def hamta(path: Path, url: str, params=None, body=None, uppdatera=False) -> Path:
    import requests
    path = Path(path)
    if path.exists() and _kvitto(path).exists() and not uppdatera:
        kontrollera(path)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    for forsok in range(4):
        try:
            r = (requests.get(url, params=params, timeout=120) if body is None
                 else requests.post(url, json=body, timeout=300))
            r.raise_for_status()
            if path.suffix == ".json":
                r.json()
            if path.suffix == ".xlsx" and not r.content.startswith(b"PK"):
                raise ValueError("Svaret är inte en xlsx-fil")
            break
        except Exception:
            if forsok == 3:
                raise
            time.sleep(2 * (forsok + 1))
    path.write_bytes(r.content)
    _kvitto(path).write_text(json.dumps(dict(
        url=r.url, metod="GET" if body is None else "POST", body=body,
        hamtad_utc=dt.datetime.now(dt.timezone.utc).isoformat(), status=r.status_code,
        sha256=hashlib.sha256(r.content).hexdigest(), bytes=len(r.content)),
        ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def vaglista(rot: Path, lan: int, uppdatera=False) -> list[dict]:
    p = hamta(Path(rot) / f"lan_{lan}/vagar.json", BAS + "road/roads",
              params=dict(lanKoder=lan, historic="false"), uppdatera=uppdatera)
    return [r for c in json.loads(p.read_text()) for r in c["roads"]]


def egenskaper(rot: Path, uppdatera=False) -> dict:
    p = hamta(Path(rot) / "egenskaper.json", BAS + "data/dataproperties",
              params=dict(categoryFilter=1), uppdatera=uppdatera)
    return {q["name"]: q for c in json.loads(p.read_text()) for q in c["properties"]}


def hamta_export(rot: Path, lan: int, vag: dict, uppdatera=False, start=0, slut=None, ar=None) -> Path:
    """Hämta PMS-export (xlsx) för en väg och riktning, körfält K1 och K2.

    ar=None: senaste mätdata per läge. ar=2023: senaste mätningen under det året (för att kunna
    väga samman flera års mätningar av samma geometri)."""
    defs = egenskaper(rot)
    assert defs["Kurvatur"]["unit"] == "10000/r", "PMS har ändrat kurvaturens enhet"
    ids = [defs[n]["propertyId"] for n in EGENSKAPER if n in defs]
    body = dict(userName="", fileName=f"vag_{vag['vn']}_r{vag['r']}", imageArray=[],
                lankod=lan, vagnummer=vag["vn"], vagundernummer=vag["vun"],
                betraktelseRiktning=vag["r"], betraktelseKorfalt=10,
                exportRiktningar=[vag["r"]], exportKorfalt=[10, 20], startLL=int(start), slutLL=int(slut if slut is not None else vag["l"]),
                includeCoordinates=True,
                dataRequests=[dict(propertyId=n, orderIndex=i, globalDateSetting=1, isAvg=False) if ar is None else
                              dict(propertyId=n, orderIndex=i, globalDateSetting=4, isAvg=False,
                                   measurementFromDate=f"{ar}-01-01", measurementToDate=f"{ar}-12-31")
                              for i, n in enumerate(ids, 1)],
                includeAvvattningsAtgarder=dict(drainagesData=False, plannedDrainages=False),
                includeBelaggningsAtgarder=dict(pavementsData=False, fictitiousPavements=False,
                                                plannedPavements=False),
                includeAvvikerFranUHStd=[], includeAvvattningsbristkod=False,
                location="https://pmsv4.trafikverket.se/analysera?")
    suffix = "" if (start == 0 and slut is None) else f"_{int(start)}_{int(slut)}"
    if ar is not None:
        suffix += f"_ar{ar}"
    p = Path(rot) / f"lan_{lan}/vag_{vag['vn']}/export_r{vag['r']}{suffix}.xlsx"
    return hamta(p, BAS + "export/analyzedata", body=body, uppdatera=uppdatera)


# ---------------------------------------------------------------- inläsning
MATKOLUMNER = {"StartLL": "start", "SlutLL": "slut", "Kurvatur": "k", "Tvärfall": "tvarfall",
               "Tvärfallsvariation": "tvarfallsvariation", "Backighet": "lutning",
               "Mätdatum faktiskt": "datum", "KF": "korfalt", "Start E": "e", "Start N": "n",
               "Z-koordinat": "z", "Pausad": "pausad"}
VAGKOLUMNER = {"StartLL": "start", "SlutLL": "slut", "Hastighetsgräns": "hastighet",
               "Vägtyp": "vagtyp", "Vägroll": "vagroll", "Länkroll": "lankroll",
               "ÅDT fordon": "adt", "Vägbredd": "vagbredd", "Antal körfält": "antal_korfalt"}


@dataclass
class Export:
    vagnummer: int
    riktning: int
    matning: pd.DataFrame
    vagdata: pd.DataFrame
    info: dict


def las_export(path: Path, krav_kvitto=True) -> Export:
    path = Path(path)
    kv = kontrollera(path) if krav_kvitto else {}
    bok = pd.read_excel(path, sheet_name=None)
    info = dict(zip(bok["Info"].iloc[:, 0].astype(str), bok["Info"].iloc[:, 1]))
    mat, vag = [], []
    for namn, df in bok.items():
        if namn.startswith("Mätdata"):
            d = df.rename(columns=MATKOLUMNER)
            d["blad"] = namn
            d["rad"] = np.arange(2, len(d) + 2)
            mat.append(d)
        elif namn.startswith("Vägdata"):
            vag.append(df.rename(columns=VAGKOLUMNER))
    m = pd.concat(mat, ignore_index=True)
    v = pd.concat(vag, ignore_index=True)
    vn = int(m["Vägnummer"].iloc[0])
    riktning = 1 if str(m["Riktning"].iloc[0]).startswith("Med") else 2
    assert (m["Vägnummer"] == vn).all()
    m["datum"] = pd.to_datetime(m["datum"], errors="coerce")
    m["pausad"] = ~m["pausad"].astype(str).str.lower().isin(["nej", "false", "0"])
    for c in ["k", "tvarfall", "tvarfallsvariation", "lutning", "e", "n", "z"]:
        m[c] = pd.to_numeric(m.get(c), errors="coerce")
    m["kalla"] = f"{path.name}"
    info["kvitto_sha256"] = kv.get("sha256")
    return Export(vn, riktning, m, v, info)


def valj_senaste(m: pd.DataFrame, korfalt="K1", max_overlapp_m=5):
    """Välj ett mätvärde per läge: senaste faktiska mätdatum vinner.

    Rader som överlappar redan valda (nyare) rader med mer än max_overlapp_m
    tas bort. Returnerar (valda, bortvalda).
    """
    q = m[(m.korfalt == korfalt) & ~m.pausad & m.k.notna()].copy()
    q = q.sort_values(["datum", "start"], ascending=[False, True])
    keep = np.zeros(len(q), bool)
    starts = np.array([], float); ends = np.array([], float)
    for i, (a, b) in enumerate(zip(q.start.to_numpy(), q.slut.to_numpy())):
        if len(starts):
            ov = np.clip(np.minimum(ends, b) - np.maximum(starts, a), 0, None).sum()
            if ov > max_overlapp_m:
                continue
        keep[i] = True
        starts = np.append(starts, a); ends = np.append(ends, b)
    return q[keep].sort_values("start").reset_index(drop=True), q[~keep]
