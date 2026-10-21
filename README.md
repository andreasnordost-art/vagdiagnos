# Vägdiagnos

Digital diagnos av befintlig geometrisk standard: PMSv4-mätningar jämförda med VGU (nybyggnad och ombyggnad) och TDOK 2018:0644.
Västernorrland (E4, E14, 83, 86, 87, 90, 331, 335, 348, 352, med provsträckor) och Jämtland (E45, E14, 84, 87).

## Innehåll

| Mapp | Vad |
|---|---|
| `leverans/` | Färdig rapport (`Vagdiagnos_Vasternorrland.html`, öppnas direkt i webbläsaren) och CSV-tabeller: 20 m-diagnos, kontroller mot GNSS, höjdmodell, tvärprofil, vägmärken |
| `vagdiagnos/` | Motorn: läsning av PMS-export, kurvanpassning, skevning, vertikalkurvor, sikt, osäkerhet |
| `verktyg/` | Skript för hämtning, kontroller, riskfaktorer, TomTom och bygget av rapporten (`bygg_rapport.py`) |
| `webb/app.html` | Gränssnittet (mall som `bygg_rapport.py` fyller med data) |
| `facit/` | Kontroll mot relationshandling E12 Västra länken |
| `indata/tomtom/` | TomTom-uttaget för E14 Sundsvall–Matfors och det kopplat till vägens km |

## Köra

Python 3.11+ med numpy, pandas, scipy, pyproj, rasterio, requests, openpyxl.

```bash
python verktyg/bygg_rapport.py <resultatkatalog> <utkatalog>
python verktyg/tomtom.py 14 indata/tomtom/jobs_10115536_results_E14.json   # VD_WORK styr arbetskatalog
```

Skripten utgår från arbetskataloger under `/home/claude/work` och `/home/claude/data` (ändra med `VD_WORK`, `VD_LAN`, `VD_VAGAR` eller i skriptet). Rådata och mellanresultat (PMS-export, höjdmodell, laserdata, cirka 3 GB) ingår inte i repot och hämtas om med verktygen.

## Inloggning

Lantmäteriets tjänster läses med `LM_USERNAME` och `LM_PASSWORD` som miljövariabler när skriptet körs. Lägg aldrig in inloggningsuppgifter i filer eller i git.
