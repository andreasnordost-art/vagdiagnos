"""VGU-referenser, avskrivna och digitaliserade direkt ur Trafikverkets dokument.

Källor (hämtade från Trafikverket, kontrollsummor i raw/common/*.source.json):
  TRVINFRA-00396 "Vägars och gators utformning", version 1.0 (2024-11-01)
  TRV publikation 2024:148 "KRAV Grundvärden", version 1.0

Tabellvärden är avskrivna ur texten. Figurvärden är digitaliserade från
sidorna renderade i 300 dpi med axelkalibrering mot graderingstexterna
(skript: verktyg/digitalisera_vgu.py). Digitaliseringsfelet är ca ±1 %.
"""
from __future__ import annotations
import json
import math
from pathlib import Path

import numpy as np

DOKUMENT = "TRVINFRA-00396 v1.0"
URL = "https://puben.trafikverket.se/dpub/visa-dokument/6f24d1e8-81bf-4e7a-948a-200e71c92f66"

# --- Horisontalradie, Tabell 8.1.6.2.1-1 (s. 234), skevad kurva med fri sikt (m)
HORISONTAL_SKEVAD = {
    #  VR: (riktvärde nybyggnad/förbättring, gränsvärde ombyggnad/lågtrafik)
    30: (60, 30), 40: (100, 60), 50: (150, 100), 60: (250, 150), 70: (300, 250),
    80: (400, 300), 90: (550, 400), 100: (700, 550), 110: (900, 700), 120: (1200, 1200),
}
# Tabell 8.1.6.2.1.1-1 (s. 235), ej skevad kurva, E = -2,5 % (m)
HORISONTAL_OSKEVAD = {50: 1200, 60: 1600, 70: 2100, 80: 2600, 90: 3200,
                      100: 3900, 110: 4600, 120: 5400}

# --- Vertikalradie, Tabell 8.1.6.8.1.1-1 (s. 254–256) (m); None = ej angivet
VERTIKAL = {
    # konvex, lång båglängd där stoppsikt för personbil är dimensionerande
    "konvex_lang": {40: (400, 300), 50: (600, 400), 60: (1100, 700), 70: (1900, 1300),
                    80: (3100, 2300), 90: (4800, 3900), 100: (5600, 4700),
                    110: (8300, None), 120: (12000, None)},
    # konvex, kort båglängd med fri sikt (komfort)
    "konvex_kort": {40: (400, 250), 50: (600, 400), 60: (800, 600), 70: (1000, 800),
                    80: (1300, 1000), 90: (1600, 1300), 100: (1900, 1600),
                    110: (2300, None), 120: (2700, None)},
    "konkav_obelyst": {40: (600, None), 50: (1000, 600), 60: (1500, 1100), 70: (2200, 1700),
                       80: (2900, 2400), 90: (3800, 3300), 100: (4500, 4400),
                       110: (4500, None), 120: (4500, None)},
    "konkav_belyst": {40: (400, 250), 50: (600, 400), 60: (800, 600), 70: (1000, 800),
                      80: (1300, 1000), 90: (1600, 1300), 100: (1900, 1600),
                      110: (2300, None), 120: (2700, None)},
}

# --- Skevning, Figur 8.1.6.6.1-2 (s. 245). Bandgränser i meter radie per VR.
# Varje skevning är tillåten inom ett radieband [min, max]. Banden överlappar;
# vid VR >= 80 väljs normalt den högsta tillåtna skevningen (K236300).
#   5.5 %: [blå streckad, blå heldragen]
#   4.0 %: [röd vänster, röd höger]
#   2.5 %: [grön vänster, grön höger = Tabell 8.1.6.2.1.1-1]
#  -2.5 %: R >= grön höger.  (-2,5 %*, svart streckad: undantag, ej använt)
# Vid VR <= 70 viker röd/blå vänsterkant horisontellt vid 70 km/h: 4,0 och
# 5,5 % saknar då nedre radiegräns i figuren (tabellradien styr).
SKEVNINGSBAND = {
    60:  {5.5: (0, 700),    4.0: (0, 961),    2.5: (205, 1600)},
    70:  {5.5: (0, 913),    4.0: (0, 1255),   2.5: (336, 2100)},
    80:  {5.5: (366, 1158), 4.0: (404, 1589), 2.5: (447, 2600)},
    90:  {5.5: (484, 1429), 4.0: (533, 1964), 2.5: (596, 3200)},
    100: {5.5: (630, 1729), 4.0: (690, 2374), 2.5: (774, 3900)},
    110: {5.5: (787, 2057), 4.0: (876, 2827), 2.5: (993, 4600)},
    120: {5.5: (978, 2409), 4.0: (1099, 3306), 2.5: (1255, 5400)},
}
SKEVNING_MAX_VR50 = 4.0  # K236306


# Tabell 8.1.6.6.1-2 (s. 246): väg med 4 eller fler körfält utan dagvatten i mittremsa.
# VR: (skevning 4,0/5,5 % om R <= första, dubbelsidigt -2,5 % om R > andra)
FYRFALT_SKEVNING = {110: (2000, 2800), 100: (1700, 2400), 80: (1100, 1600)}


def skevning_krav(radie_m: float, vr: int, fyrfalt: bool = False):
    """Returnerar (minsta_tillåtna, normalt_val) inåtriktad skevning i procent.

    -2.5 betyder att ingen skevning krävs (dubbelsidigt tvärfall tillåtet).
    None om radien ligger under alla band (radien klarar då inte heller tabellen).
    """
    vr = int(vr)
    if fyrfalt and vr in FYRFALT_SKEVNING:
        lag, hog = FYRFALT_SKEVNING[vr]
        if radie_m > hog:
            return -2.5, -2.5
        if radie_m <= lag:
            return 4.0, 5.5 if radie_m <= SKEVNINGSBAND[vr][5.5][1] else 4.0
        return 2.5, 2.5
    if vr not in SKEVNINGSBAND:
        return None, None
    if radie_m >= HORISONTAL_OSKEVAD.get(vr, math.inf):
        return -2.5, -2.5
    tillatna = [e for e, (lo, hi) in SKEVNINGSBAND[vr].items() if lo <= radie_m <= hi]
    if not tillatna:
        return None, None
    minst = min(tillatna)
    normal = max(tillatna) if vr >= 80 else minst
    return minst, normal


# --- Stoppsikt, Figur 8.1.5.1.5-1/-2/-3 (s. 227–228), siktsträcka (m) som
# funktion av längslutning (%) -8..+8. Digitaliserade.
_SS = json.loads((Path(__file__).with_name("stoppsikt_vgu.json")).read_text())


def stoppsikt(vr: int, lutning_pct: float, variant: str = "grans_nybyggnad"):
    """Erforderlig stoppsikt (m). variant: grans_nybyggnad | riktvarde | grans_ombyggnad."""
    tab = _SS[variant].get(str(int(vr)))
    if tab is None:
        return None
    g = sorted((float(k), v) for k, v in tab.items() if v is not None)
    x = np.array([a for a, _ in g]); y = np.array([b for _, b in g])
    return float(np.interp(np.clip(lutning_pct, -8, 8), x, y))


OGONHOJD_M = 1.10      # Tabell 8.1.5.1.2-1, personbil
HINDERHOJD_M = 0.35    # Tabell 8.1.5.1.2-1, lågt bromsljus (fotnot 8)


def horisontal_krav(vr: int):
    return HORISONTAL_SKEVAD.get(int(vr), (None, None))


def vertikal_krav(familj: str, vr: int):
    return VERTIKAL[familj].get(int(vr), (None, None))


# --- TDOK 2018:0644 "Beslut om högsta tillåtna hastighetsgräns på befintlig väg" (v1.0, 2019-11-18)
# Tabell 1 (råd): längs huvuddelen av sträckan bör stoppsikt och horisontalradie uppfylla:
#   80 km/h: stoppsikt 85 m, radie 300 m; 90: 110 m, 400 m; 100: 135 m, 500 m (*ett körfält per riktning)
# Mötesfri väg 100 km/h (avsnitt 3.2.1): krav avseende horisontalkurvor utgår.
# 110 km/h mötesfri och 120 km/h motorväg: kraven för 100 km/h ska vara uppfyllda (3.2.3, 3.3.2).
TDOK_URL = "TDOK 2018:0644 v1.0"
TDOK_STOPPSIKT = {80: 85, 90: 110, 100: 135, 110: 135, 120: 135}
TDOK_RADIE = {80: 300, 90: 400, 100: 500, 110: 500, 120: 500}


def tdok_radie(vr: int, motesfri: bool):
    vr = int(vr)
    if vr < 80:
        return None
    if motesfri and vr >= 100:
        return None
    return TDOK_RADIE.get(vr)


def tdok_stoppsikt(vr: int):
    return TDOK_STOPPSIKT.get(int(vr)) if int(vr) >= 80 else None
