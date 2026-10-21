"""Facit: E12 Röbäck–Norra länken (Västra länken), relationshandling 2022-08-26 med PM t.o.m. 2025-09.

Avläst från profilritningarnas band (Plandata, Lutning/Radie, Tvärfall) på ritningarna
4 01 T 03 01–08 och 5 01 T 03 09–15. Gränser mätta i ritningens vektorgrafik
(1:1000, ±1 m), etiketter avlästa visuellt. Längdmätning i ritningens km (m).
Tecken: ritningens R < 0 = vänsterkurva (kontrollerat mot tvärfallsbandet).
Sträckor kring cirkulationsplatserna Teg (~0/874) och Röbäck (~2/850–2/880) är
egna räknade linjer och används inte i jämförelsen.
"""

# (start_km_m, slut_km_m, typ, värde) ; typ: 'R' radie (m, tecken enligt ritning), 'A' klotoid, 'RL' raklinje
PLAN = [
    (979, 1382, "R", -520),
    (1382, 1506, "A", 314),     # klotoid mellan R-520 och R-1500 (korgbåge)
    (1506, 1942, "R", -1500),
    (1942, 2141, "A", 691),     # klotoid R-1500 -> R-4000
    (2141, 2744, "R", -4000),
    (2744, 2799, "R", -400),
    (2799, 2847, "RL", None),
    (2880, 2998, "R", 420),
    (2998, 3038, "R", 800),
    (3038, 3073, "R", 1600),
    (3073, 3201, "R", 800),
    (3201, 3284, "A", 257),
    (3284, 3476, "RL", None),
    (3476, 3565, "A", 250),
    (3565, 3658, "R", -700),
    (3658, 3740, "A", 240),
    (3740, 3822, "A", 240),
    (3822, 4605, "R", 700),
    (4605, 4688, "A", 240),
    (4688, 4737, "RL", None),
    (4737, 5729, "R", -5000),
    (5729, 5804, "A", 280),
    (5804, 6171, "R", -870),
    (6171, 6337, "A", 380),
    (6337, 6423, "A", 262),
    (6423, 6700, "R", 800),
]

# Vertikalgeometri: (start, slut, typ, värde) typ 'L' lutning %, 'Rv' radie (ritningens tecken: + = konvex/krön)
PROFIL = [
    (906, 1115, "Rv", -25000),
    (1115, 1153, "L", 1.53),
    (1153, 1612, "Rv", 16000),
    (1612, 1792, "L", -1.34),
    (1792, 2098, "Rv", -20000),
    (2098, 2198, "L", 0.19),
    (2198, 2434, "Rv", 60000),
    (2434, 2645, "L", -0.21),
    (2645, 3426, "Rv", -40000),
    (3426, 3711, "L", 1.75),
    (3711, 4064, "Rv", 12000),
    (4064, 4115, "L", -1.20),
    (4115, 4535, "Rv", -14000),
    (4535, 4597, "L", 1.80),
    (4597, 5387, "Rv", -50000),
    (5387, 5508, "L", 3.38),
    (5508, 6521, "Rv", 18000),
    (6521, 6700, "L", -2.25),
]

# Projekterat tvärfall i cirkelbågarna (inåt, %), avläst ur tvärfallsbandet
TVARFALL = {
    (979, 1382): 4.0, (1506, 1942): 4.0, (2141, 2744): -2.5, (2880, 2998): 4.0,
    (3073, 3201): 4.0, (3565, 3658): 4.0, (3822, 4605): 4.0, (4737, 5729): -2.5,
    (5804, 6171): 4.0, (6423, 6700): 4.0,
}

# Profilhöjder (RH2000) var 10:e meter, urval för nivåkontroll
HOJDER = {1000: 12.01, 1200: 14.73, 1390: 15.94, 1500: 15.61, 1700: 13.32, 2000: 10.37, 2100: 10.32,
          2500: 10.35, 2700: 9.97, 3000: 10.89, 3200: 12.76, 3500: 17.36, 3600: 19.10, 3900: 22.85,
          4000: 22.61, 4300: 20.40, 4500: 22.06, 5000: 32.62, 5500: 48.01, 6000: 58.18, 6110: 58.55,
          6500: 54.46, 6690: 50.20}
