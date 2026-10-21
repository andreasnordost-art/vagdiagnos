"""Bygg vägrapporten (en fristående HTML-fil) av analysresultaten."""
import base64, gzip, json, pickle, sys
from pathlib import Path
import pandas as pd
from pyproj import Transformer
T3857 = Transformer.from_crs(3006, 3857, always_xy=True)
def till3857(xy):
    out=[]
    for p in xy:
        if p[0] is None: out.append([None,None]); continue
        x,y=T3857.transform(p[0],p[1]); out.append([round(x),round(y)])
    return out
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vagdiagnos.export import bygg, vagnamn

RES = Path(sys.argv[1]); UT = Path(sys.argv[2]); UT.mkdir(parents=True, exist_ok=True)
MALL = Path(__file__).resolve().parents[1] / 'webb/app.html'
ORDNING = [4, 14, 83, 86, 87, 90, 331, 335, 348, 352]

vagar, data, celltabeller = [], {}, []
from vagdiagnos.pms import las_export, valj_senaste
from vagdiagnos.analys import _lagg_pa_ref
from vagdiagnos.referens import Referenslinje
import numpy as np


def bildregister(vn, langd, cell, rawdir='/home/claude/src/arkiv/raw'):
    """PMS-vägbilder hämtas live; här sparas bara vad bild-API:t behöver per 20 m-avsnitt:
    periodens datum för båda riktningarna och riktning 2:s egen löpande längd."""
    raw = Path(rawdir) / f'vag_{vn}'
    try:
        m1, _ = valj_senaste(las_export(raw / 'export_r1.xlsx').matning)
        m2, _ = valj_senaste(las_export(raw / 'export_r2.xlsx').matning)
    except Exception as e:
        print('bildregister saknas', vn, e); return None
    n = int(np.ceil(langd / cell)); mid = np.arange(n) * cell + cell / 2
    datum = sorted(set(m1['Mätdatum'].astype(str).str[:10]) | set(m2['Mätdatum'].astype(str).str[:10]))
    di = {d: i for i, d in enumerate(datum)}
    def per_cell(s_arr, d_arr, max_avst=30):
        o = np.argsort(s_arr); s_arr = s_arr[o]; d_arr = d_arr[o]
        j = np.clip(np.searchsorted(s_arr, mid), 1, len(s_arr) - 1)
        j = np.where(np.abs(s_arr[j - 1] - mid) < np.abs(s_arr[j] - mid), j - 1, j)
        ok = np.abs(s_arr[j] - mid) <= max_avst
        return j, ok, o
    s1 = (m1.start.to_numpy(float) + m1.slut.to_numpy(float)) / 2
    j1, ok1, o1 = per_cell(s1, m1['Mätdatum'].astype(str).str[:10].to_numpy())
    d1 = [di[x] if k else -1 for x, k in zip(m1['Mätdatum'].astype(str).str[:10].to_numpy()[o1][j1], ok1)]
    ref = Referenslinje(m1.start.to_numpy(float), m1.e.to_numpy(), m1.n.to_numpy())
    r2 = _lagg_pa_ref(ref, m2); r2 = r2[np.isfinite(r2.b)]
    j2, ok2, o2 = per_cell(r2.b.to_numpy(float), r2['Mätdatum'].astype(str).str[:10].to_numpy())
    d2 = [di[x] if k else -1 for x, k in zip(r2['Mätdatum'].astype(str).str[:10].to_numpy()[o2][j2], ok2)]
    ll2 = [int(round(x)) if k else -1 for x, k in zip(((r2.start + r2.slut) / 2).to_numpy(float)[o2][j2], ok2)]
    return dict(datum=datum, d1=d1, d2=d2, ll2=ll2)

def precision_html(Q):
    """Metodtext om de åtgärder som skärper värdena (kalibrering, sammanvägning, flera år, tvärprofil)."""
    fa = Q.get('flerar', {}); tp = Q.get('tvarprofil', [])
    t = """<p><b>Skärpta värden.</b> <i>Kalibrerade osäkerhetsintervall.</i> Jämförelsen med GNSS-spåret visade att horisontalradiernas 90 %-intervall var för breda (de täckte 97–98 %). Intervallen har därför smalnats av per radieklass, med faktor 0,62 under 1 000 m, 0,80 för 1 000–2 500 m och 1,0 över 2 500 m, så att täckningen blir ungefär 90 %. Vertikalradiernas intervall täckte 92 % och har behållits.</p>
<p><i>Sammanvägning med GNSS-spåret.</i> Där mätbilens GNSS-spår ger en pålitlig cirkel (spridning under 0,5 m) vägs dess radie in i log-skala, med vikt efter osäkerheten och högst 50 %. Intervallet omfattar alltid båda riktningarnas radie och GNSS-radien.</p>
<p><i>Radier ur GNSS-banan.</i> Motorns kurvform (övergångar och cirkelbågar ur gyrots kurvatur) har också anpassats till mätbilens koordinater genom hela kurvan och 100 m raklinje på var sida. Varje bågas kurvatur, startriktning, läge och längdskala bestäms så att den integrerade banan följer GNSS-punkterna. I en sammansatt kurva bestäms alla bågar samtidigt. Följer banan inte punkterna inom 1 m anpassas varje båge för sig med grannbågarna fasta. Där cirkelanpassningen fungerar bra (enkla kurvor med minst 300 m cirkelbåge) stämmer de två GNSS-metoderna inom 1 % i median. Metoden ger en godkänd radie för 82 % av bågarna i sammansatta kurvor, där cirkelanpassningen bara ger en medelradie. Mot E12-ritningen gav den 1 502 m (ritning 1 500), 4 979 m (5 000) och 847 m (870) för sammansatta bågar, där motorn gav 1 494, 4 879 och 808 m. Banans radie vägs in med högst 50 % för de kurvor som inte redan vägts samman med GNSS-cirkeln, och intervallet omfattar den alltid.</p>
<p><i>Vertikalradier och höjdmodellen.</i> Lantmäteriets markhöjdmodell (1 m) har lästs var 10:e meter längs hela vägarna, i mätbilens spår. Motorns vertikalradie vägs samman med höjdmodellens radie (invers varians i log-radie) när båda ger samma kurvtyp. Svackor bedöms mot kravet för obelyst väg, eftersom det antas att belysning saknas.</p>"""
    if fa:
        t += f"""<p><i>Flera mätår.</i> Varje kurvas radie har också anpassats separat i varje års mätning 2016–2025 (riktning 1, median {fa['median_antal_ar']:.0f} år per kurva, {fa['kurvor']} kurvor med GNSS-kontroll). Radien varierar bara cirka {100*fa['median_sd_mellan_ar']:.0f} % mellan åren. Medelvärdet över åren stämmer inte bättre med GNSS-spåret än motorns skattning (median avvikelse {100*fa['median_abs_logfel_flerar']:.1f} % mot {100*fa['median_abs_logfel_motor']:.1f} %). Felet är alltså systematiskt och inte mätbrus, och fler år gör inte radien säkrare. Åren används därför som kontroll: intervallet omfattar årsmedianen, och kurvor under 2 500 m där åren tydligt skiljer sig (robust spridning över 15 %, 60 av 1 109 kurvor) flaggas som möjlig ombyggnad eller mätstörning.</p>"""
    if tp:
        tak = np.mean([x['takfall'] for x in tp]); kk = [x for x in tp if 'korr' in x]
        t += f"""<p><i>Tvärfall och tvärprofil.</i> På provsträckorna har PMS tvärprofil hämtats för båda riktningarna (65 lasrar över 6,4 m, medel per 20 m). Ur den beräknas fallet i mätbilens eget körfält och om vägen har takfall, alltså krön vid mittlinjen så att körfälten lutar åt var sitt håll. Profilens fall i körfältet stämmer med exportens tvärfall (korrelation {min(x['korr'] for x in kk):.2f}–{max(x['korr'] for x in kk):.2f}, median skillnad {np.median([x['diff_abs_median'] for x in kk]):.1f} procentenheter i {len(kk)} jämförelser). Exportens värde avser alltså körfältet och har inte rättats. Takfall finns i {100*tak:.0f} % av provsträckornas 20-metersavsnitt. Exporten kan inte visa takfall, eftersom den bara mäter ett körfält. Kurvor där kärnan har takfall i minst halva längden flaggas, eftersom vägbanan då inte är skevad över hela bredden.</p>"""
    t += """<p><i>Sikt och körfältsläge.</i> Sikten räknas i båda körfälten med vägbanans höjd ur PMS-profilen. Radien gäller mätbilens spår i körfältets mitt och inte vägens mittlinje. Skillnaden är under 1 % för radier över 200 m och har inte korrigerats.</p>"""
    return t


def tabell(p):
    p = Path(p); return pd.read_csv(p) if p.exists() else None


def lagg_till(vn, lan_namn, lk, prefix, resdir, W, kontroll, rawdir):
    """Bygg en vägs data. W = arbetskatalog med kvalitet/, risk/ och tvar/ för länet."""
    p = Path(resdir) / f'vag_{vn}.pkl'
    if not p.exists():
        return
    r = pickle.load(open(p, 'rb'))
    VDTM = tabell(f'{W}/kvalitet/vertikal_mot_hojdmodell.csv'); FLERAR = tabell(f'{W}/kvalitet/flerar.csv'); BANA = tabell(f'{W}/kvalitet/bana.csv')
    # Kvalitetssäkring: GNSS-radier per kurva (se verktyg/validera_radier.py) vidgar osäkerheten där källorna är oense
    g = kontroll[kontroll.vag == vn]
    r.gnss = {int(k): (float(a), float(b)) for k, a, b in zip(g.kurva, g.R_gnss, g.gnss_rms)}
    # Vertikalradier: höjdmodellens skattning (samma kurvtyp) vägs in
    if VDTM is not None:
        q = VDTM[(VDTM.vag == vn) & VDTM.samma_typ]
        r.vdtm = {int(x.nr): (float(x.Rv_dtm), float(x.Rv_dtm_lag), float(x.Rv_dtm_hog)) for x in q.itertuples()}
    # Tvärfall i kurvor: eget körfält ur PMS 6,4 m-tvärprofil (provsträckor), se verktyg/tvarprofil.py
    r.tp = {}
    for namn, suff in (('r1', ''), ('r2', '_r2')):
        f = Path(f'{W}/tvar/vag{vn}_tvarprofil{suff}.json')
        if f.exists():
            r.tp[namn] = pd.DataFrame(json.loads(f.read_text())['sektioner']).drop_duplicates('s')
    # Flera mätår: radie per år och kurva (verktyg/flerar.py)
    if FLERAR is not None:
        q = FLERAR[FLERAR.vag == vn]
        r.flerar = {int(k): g.R_ar.to_numpy(float) for k, g in q.groupby('kurva_nr')}
    # GNSS-banan: kurvformen anpassad till mätbilens koordinater (verktyg/kor_bana.py)
    if BANA is not None:
        q = BANA[BANA.vag == vn]
        r.bana = {int(x.kurva_nr): (float(x.R_bana), float(x.rel_bana), float(x.rms_bana)) for x in q.itertuples()}
    d, cel = bygg(r, lan_namn)
    d['xy'] = till3857(d['xy']); d['lk'] = lk
    key = f'{prefix}{vn}'
    d['vb'] = bildregister(vn, d['langd'], d['cell'], rawdir)
    _rf = Path(f'{W}/risk') / f'vag{vn}_risk.json'
    if _rf.exists():  # riskfaktorer: vattensamling, avvattning, spårdjup och solbländning per 20 m (verktyg/riskceller.py)
        _R = json.loads(_rf.read_text()); d['risk'] = {k: _R[k] for k in ('va', 'res', 'sd', 'sol', 'solm', 'solk')}
    _sr = Path(f'{W}/tvar') / f'vag{vn}_skyltregister.json'
    if _sr.exists():  # vägmärken på kartan, med utklipp ur vägbilden
        d['skyltar'] = json.loads(_sr.read_text())['skyltar']
    _tt = Path(f'{W}/tomtom') / f'vag{vn}.json'
    if _tt.exists():  # uppmätta hastigheter från TomTom per segment och riktning (verktyg/tomtom.py)
        d['tt'] = json.loads(_tt.read_text())
        mid = (cel['start'].to_numpy(float) + cel['slut'].to_numpy(float)) / 2
        for r in (1, 2):
            q = [x for x in d['tt']['segment'] if x['r'] == r]
            if not q:
                continue
            v85, med = np.full(len(mid), np.nan), np.full(len(mid), np.nan)
            for x in q:
                w = (mid >= x['s0']) & (mid < x['s1'])
                v85[w] = x['p'][16] if len(x['p']) == 19 else np.nan; med[w] = x['med'] if x['med'] is not None else np.nan
            cel[f'tomtom_v85_r{r}'] = v85; cel[f'tomtom_median_r{r}'] = med
    data[key] = base64.b64encode(gzip.compress(json.dumps(d, ensure_ascii=False, separators=(',', ':')).encode(), 9)).decode()
    vagar.append(dict(nyckel=key, namn=d['namn'], kort=d['namn'].replace('Väg ', ''), lan=lan_namn, lk=lk, langd=d['langd']))
    cel.insert(0, 'vag', d['namn']); cel.insert(0, 'lan', lan_namn); celltabeller.append(cel)
    print(lan_namn, vn, len(data[key]) // 1024, 'kB')


# Län som ingår: (namn, länskod, nyckelprefix, resultat, arbetskatalog, GNSS-kontroll, rådata för vägbilder, vägar)
LAN = [('Västernorrland', 22, 'y', RES, '/home/claude/work', '/home/claude/work/leverans/kontroll_radier_gnss_nvdb.csv', '/home/claude/src/arkiv/raw', ORDNING),
       ('Jämtland', 23, 'z', Path('/home/claude/work/lan23/res'), '/home/claude/work/lan23', '/home/claude/work/lan23/kvalitet/kontroll_radier_gnss_nvdb.csv', '/home/claude/data/lan_23', [45, 14, 84, 87])]
for namn, lk, pre, rd, W, kf, raw, vl in LAN:
    if not Path(kf).exists():
        print('hoppar över', namn); continue
    kontroll = pd.read_csv(kf)
    for vn in vl:
        lagg_till(vn, namn, lk, pre, rd, W, kontroll, raw)

val = pickle.load(open('/home/claude/work/validering_e12.pkl', 'rb'))
h = val['h']; v = val['v']; tv = val['tv']
def rad_h(r):
    return f"<tr><td class='n'>{r.R_facit}</td><td class='n'>{r.R_motor}</td><td class='n'>{r.R_lag}–{r.R_hog}</td><td class='n'>{r.fel_pct:+.1f} %</td></tr>"
def rad_v(r):
    return f"<tr><td>{r.typ_facit}</td><td class='n'>{r.Rv_facit}</td><td class='n'>{r.Rv_motor}</td><td class='n'>{r.fel_pct:+.1f} %</td></tr>"
def rad_t(r):
    return f"<tr><td class='n'>{r.km}</td><td class='n'>{r.tvarfall_facit:+.1f}</td><td class='n'>{r.r1:+.2f}</td><td class='n'>{r.r2:+.2f}</td></tr>"
hh = h[h.R_facit.isin([520, 1500, 700, 5000, 870])]
vv = v[v.Rv_facit <= 20000]
metod = f"""
<p><b>Underlag.</b> Trafikverkets vägytemätningar i PMSv4 för båda körriktningarna (körfält K1): kurvatur, tvärfall och backighet som 20-metersmedelvärden samt mätbilens koordinater och höjd. Varje fil har kontrollerats mot sin hämtningskvittens (SHA-256). Kraven kommer från VGU, TRVINFRA-00396 version 1.0: tabell 8.1.6.2.1-1 (horisontalradie), figur 8.1.6.6.1-2 (skevning), tabell 8.1.6.8.1.1-1 (vertikalradie) och figur 8.1.5.1.5-1 och -3 (stoppsikt). <i>Vägar:</i> i Västernorrland E4, E14 och vägarna 83, 86, 87, 90, 331, 335, 348 och 352, med provsträckor på 5 km per väg för sidoområde, sikt i plan och vägmärken. I Jämtland E45, E14 och vägarna 84 och 87, hela vägarna men utan provsträckor. E14 och väg 87 går genom båda länen och redovisas per län med länets egen längdmätning.</p>
<p><b>Horisontalradie.</b> PMS anger kurvaturen som medelvärde över 20 m. Motorn anpassar övergångskurva – cirkelbåge – övergångskurva (även sammansatta kurvor med flera bågar) så att modellens 20-metersmedel stämmer med båda riktningarnas mätvärden. Då återfås cirkelbågens radie även där bågen omges av långa klotoider, vilket ett enkelt medelvärde eller en median inte klarar.</p>
<p><b>Skevning.</b> Inåtriktat tvärfall mäts i varje körfält inom cirkelbågen. Figuren i VGU anger tillåtna radieband för 2,5, 4,0 och 5,5 % skevning. Brist redovisas när det sämre körfältet ligger under den minsta tillåtna skevningen för radien. Att skevningen är lägre än det normala valet vid 80 km/h och mer redovisas bara i detaljerna.</p>
<p><b>Längdprofil och sikt.</b> Profilen beräknas ur mätbilens höjder och backighet. Vertikalradier skattas på samma sätt som horisontalradier. Alla vägar antas sakna belysning, så svackor bedöms mot kravet för obelyst väg. Stoppsikten beräknas direkt längs profilen med ögonhöjd 1,10 m och hinderhöjd 0,35 m. Sikthinder vid sidan av vägen bedöms bara på provsträckorna (se nedan).</p>
<p><b>Kravnivåer.</b> <i>Nybyggnad</i> använder VGU:s riktvärden för radier och gränsvärdet för stoppsikt (figur 8.1.5.1.5-1). <i>Ombyggnad</i> använder VGU:s gränsvärden när befintlig linjeföring behålls (stoppsikt enligt figur 8.1.5.1.5-3). <i>Hastighetsbeslut</i> använder Trafikverkets riktlinje TDOK 2018:0644 (version 1.0), tabell 1: stoppsikt 85, 110 och 135 m och kurvradie 300, 400 och 500 m vid 80, 90 och 100 km/h. Kurvradien gäller inte mötesfri väg från 100 km/h, och 110 och 120 km/h bedöms med kraven för 100 km/h. Riktlinjen avser huvuddelen av en sträcka och omfattar också vägbredd, säkerhetszon, korsningar och miljö, som inte kan bedömas ur mätdata.</p>
<p><b>Säkerhet.</b> Varje mätvärde har ett osäkerhetsintervall (90 %) från anpassningen, skillnaden mellan riktningarna och felen i testerna nedan. <i>Klarar</i> betyder att hela intervallet klarar kravet, <i>Brist</i> att hela intervallet underskrider det, <i>Osäkert</i> att kravet ligger inom intervallet. Säkerställd nivå räknar osäkra som ej klarade, möjlig nivå räknar dem som klarade.</p>
<p><b>Kontroll mot relationshandling.</b> Metoden har prövats på E12 Västra länken i Umeå (relationshandling 2022, PM till och med 2025) med PMS-mätningar från 2025–2026. Ritningarnas radier, skevning och vertikalradier jämfördes med motorns resultat:</p>
<table><thead><tr><th>Radie ritning (m)</th><th>Motorn (m)</th><th>90 % intervall</th><th>Avvikelse</th></tr></thead><tbody>{''.join(rad_h(r) for r in hh.itertuples())}</tbody></table>
<table><thead><tr><th>Sträcka (km)</th><th>Skevning ritning (%)</th><th>Körfält 1</th><th>Körfält 2</th></tr></thead><tbody>{''.join(rad_t(r) for r in tv.drop_duplicates('km').itertuples())}</tbody></table>
<table><thead><tr><th>Typ</th><th>Rv ritning (m)</th><th>Motorn (m)</th><th>Avvikelse</th></tr></thead><tbody>{''.join(rad_v(r) for r in vv.itertuples())}</tbody></table>
<p>Höjderna i PMS ligger 0,6–0,9 m över ritningens profil, men höjdskillnaderna längs vägen stämmer inom ±0,15 m. Det räcker för sikt och vertikalradier, som bara beror på profilens form. En stor radie (R 4 000) direkt intill en kort, skarp kurva vid en cirkulationsplats skattades för lågt (2 800 m). Felet går åt det försiktiga hållet.</p>
<p><b>Begränsningar.</b> Resultatet är en screening av geometrin, inte en fullständig VGU-granskning. Det säger inget om olycksrisk eller korsningar. Sidoområden, räcken, sikt i plan och vägmärken bedöms bara på provsträckorna (5 km per väg). Kurvor kortare än cirka 20 m kan inte bestämmas exakt ur 20-metersdata och redovisas som osäkra. Vid cirkulationsplatser och korsningar mäter bilen sin egen svängrörelse; dessa zoner tas bort ur analysen.</p>
"""
ss = json.loads((Path(__file__).resolve().parents[1] / 'vagdiagnos/stoppsikt_vgu.json').read_text())
stoppsikt = {k: {str(V): [ss[k][str(V)][f"{g:.1f}"] for g in range(-8, 9)] for V in [40,50,60,70,80,90,100,110,120] if str(V) in ss[k]} for k in ('grans_nybyggnad', 'grans_ombyggnad')}
bakgrund = json.loads((Path(__file__).resolve().parents[1] / 'webb/bakgrund.json').read_text())
bakgrund['lansgrans'] = till3857(bakgrund['lansgrans'])
bakgrund['orter'] = [[n] + till3857([[x, y]])[0] for n, x, y in bakgrund['orter']]
# Tvärsektioner (endast verkliga beräkningar, aldrig syntetiska testdata)
tvar, tvar_meta = {}, {}
BILD_GRANS = 0.01  # andel räckespixlar i bildens närområde som räknas som räcke (kalibrerad mot NVDB på väg 83)
for vn in ORDNING:
    f = Path('/home/claude/work/tvar') / f'vag{vn}.json'
    if not f.exists():
        continue
    T = json.loads(f.read_text())
    for k in T['sektioner']:
        for x in ('x0', 'y0', 'x1', 'y1'):
            k.pop(x, None)
        k['z_mitt'] = round(k['z_mitt'], 2)
        for sd in k['sidor'].values():
            for a, b in list(sd.items()):
                if isinstance(b, float): sd[a] = round(b, 2)
                if isinstance(b, dict):
                    sd[a] = {c: (round(e, 2) if isinstance(e, float) else e) for c, e in b.items()}
    kl = T['kallor']; lm = kl.get('laserdata') or []
    laser = any('punkter' in x for x in lm)
    datum = sorted({str(x.get('datum', ''))[:10] for x in lm if x.get('datum')})
    T['laser'] = laser
    vd_fil = Path(f'/home/claude/data/lan_22/vag_{vn}/export_r1.xlsx')
    if vd_fil.exists():
        vd = las_export(vd_fil).vagdata; q = vd[(vd.start < T['s1']) & (vd.slut > T['s0'])]
        if 'adt' in q and q.adt.notna().any(): T['adt'] = float(q.adt.max())
    T['kalla'] = ('Källor: Lantmäteriet, Markhöjdmodell Nedladdning (grid 1 m)' +
                  (f' och Laserdata Nedladdning, skog ({", ".join(d[:4] for d in datum) or "datum saknas"})' if laser else '; laserdata kunde inte hämtas, så föremål saknas') +
                  ', CC BY 4.0. Vägbredd och ÅDT ur PMSv4.')
    T.pop('kallor', None)
    sf = Path('/home/claude/work/tvar') / f'vag{vn}_sikt.json'
    if sf.exists():
        SI = {round(u['s']): u for u in json.loads(sf.read_text())}
        for k in T['sektioner']:
            u = SI.get(round(k['s']))
            if not u: continue
            k['sikt'] = {}
            for r_ in ('r1', 'r2'):
                q = u.get(r_)
                if not q: continue
                q = dict(q)
                if q.get('hp'):
                    x_, y_ = T3857.transform(*q['hp']); q['hp'] = [round(x_), round(y_)]
                k['sikt'][r_] = q
    # Tvärfall i eget körfält ur PMS 6,4 m-tvärprofil: [tvärfall %, takfall 0/1] per riktning
    for rk, suf in (('1', ''), ('2', '_r2')):
        pf = Path('/home/claude/work/tvar') / f'vag{vn}_tvarprofil{suf}.json'
        if not pf.exists():
            continue
        P = sorted(json.loads(pf.read_text())['sektioner'], key=lambda x: x['s'])
        ps = np.array([x['s'] for x in P])
        for k in T['sektioner']:
            j = int(np.searchsorted(ps, k['s'], side='right')) - 1
            if 0 <= j < len(P) and P[j]['s'] <= k['s'] < P[j]['t'] + 0.01 and P[j].get('tf_eget') is not None:
                k.setdefault('tf', {})[rk] = [P[j]['tf_eget'], int(bool(P[j]['takfall']))]
    ff = Path('/home/claude/work/tvar') / f'vag{vn}_skyltregister.json'
    if ff.exists():   # tolkade vägmärken (verktyg/skyltregister.py); utklippen ligger i vägens huvuddata
        T['skyltar'] = [{k: x[k] for k in ('s', 't', 'kat', 'kod', 'namn', 'r', 'n', 'sida', 'm', 'avst', 'fil', 'id')} for x in json.loads(ff.read_text())['skyltar']]
    rf = Path('/home/claude/work/tvar') / f'vag{vn}_racke.json'
    if rf.exists():
        T['racke'] = json.loads(rf.read_text())
        if T.get('korbana'):
            # Motorväg: sektionen följer riktning 1:s körbana. Mot-riktningens räcken står vid den andra
            # körbanan; på vänster sida (mittremsan) gäller i stället mitträcket.
            m = sorted(T['racke'].get('mitt_r1', []))
            slaget = []
            for a_, b_ in m:
                if slaget and a_ - slaget[-1][1] < 1: slaget[-1][1] = b_
                else: slaget.append([a_, b_, 'Mitträcke'])
            T['racke']['vanster'] = slaget
        T['kalla'] = T['kalla'].replace('Vägbredd och ÅDT ur PMSv4.', 'Vägbredd, ÅDT och räcken (NVDB) ur PMSv4.')
    # Vägbilder (PMS) tolkade med Mask2Former: räcke och stolpar per sida
    bm = Path('/home/claude/work/bilder') / f'vag{vn}'
    if (bm / 'tolkning.json').exists():
        from PIL import Image as PI
        tolk = json.loads((bm / 'tolkning.json').read_text()); idx = json.loads((bm / 'index.json').read_text())
        bilder = {b['fil']: b for b in idx if b['fil'] and b['fil'] in tolk}
        # Riktning 1 tittar framåt (ökande km): närområdet i bilden ligger ca 6-26 m framför kameran.
        # Riktning 2 tittar bakåt: dess högra sida är vänster sida i riktning 1.
        def bevis(s, sida):
            ut = []
            for f, b in bilder.items():
                if b['r'] == 1 and s - 26 <= b['s'] <= s - 6:
                    ut.append((tolk[f][sida], f))
                if b['r'] == 2 and not T.get('korbana') and s + 6 <= b['s'] <= s + 26:
                    ut.append((tolk[f]['hoger' if sida == 'vanster' else 'vanster'], f))
            return ut
        for k in T['sektioner']:
            k['bild'] = {}
            for sida in ('vanster', 'hoger'):
                ev = bevis(k['s'], sida)
                if not ev: continue
                k['bild'][sida] = dict(racke=round(max(e['racke'] for e, _ in ev), 4),
                                       stolpe=int(max(e['stolpobjekt'] for e, _ in ev)),
                                       veg=round(max(e['vegetation'] for e, _ in ev), 3), n=len(ev))
            # bilder att visa: närmaste framåt- och bakåtbild
            f1 = [f for f, b in bilder.items() if b['r'] == 1 and k['s'] - 26 <= b['s'] <= k['s'] - 6]
            f2 = [] if T.get('korbana') else [f for f, b in bilder.items() if b['r'] == 2 and k['s'] + 6 <= b['s'] <= k['s'] + 26]
            k['bildfil'] = [max(f1, key=lambda f: bilder[f]['s']) if f1 else None, min(f2, key=lambda f: bilder[f]['s']) if f2 else None]
        # Tolkningsmasker (128x96, värde 0-3) som körlängdskodade listor; själva fotot hämtas live från PMS
        import numpy as _np
        masker = {}
        for f in bilder:
            mf = bm / 'masker' / (f[:-4] + '.png')
            if not mf.exists(): continue
            a_ = _np.asarray(PI.open(mf)).ravel(); grans_ = _np.flatnonzero(_np.diff(a_)) + 1
            st_ = _np.r_[0, grans_]; ln_ = _np.diff(_np.r_[st_, a_.size])
            masker[f] = [int(x) for p_ in zip(a_[st_], ln_) for x in p_]
        T['masker'] = masker
        T['bildinfo'] = {f: [b['r'], b['ll'], b['datum']] for f, b in bilder.items()}
        T['bilddatum'] = sorted({b['datum'] for b in bilder.values()})
        T['bild_grans'] = BILD_GRANS
        print('vägbildsmasker', len(masker))
    # Packning: gemensam t-axel, marken som heltalsskillnader i cm, vegetation som höjd (dm) per meter
    T['t'] = T['sektioner'][0]['t']
    for k in T['sektioner']:
        k.pop('t', None)
        z = k.get('z')
        if z and all(v is not None for v in z):
            c = [int(round(v * 100)) for v in z]
            k['zb'] = c[0]; k['zd'] = [c[i] - c[i - 1] for i in range(1, len(c))]; k.pop('z')
        if k.get('veg'):
            bins = {}
            for u, h, _zt in k['veg']:
                if abs(u) > 30: continue
                m = int(round(u)); bins[m] = max(bins.get(m, 0), h)
            k['veg'] = [[m, int(round(h * 10))] for m, h in sorted(bins.items()) if h >= 0.5]
    key = f'y{vn}'
    tvar[key] = base64.b64encode(gzip.compress(json.dumps(T, ensure_ascii=False, separators=(',', ':')).encode(), 9)).decode()
    tvar_meta[key] = dict(s0=T['s0'], s1=T['s1'], n=len(T['sektioner']))
    print('tvärsektioner', vn, len(T['sektioner']), len(tvar[key]) // 1024, 'kB')
metod += """<p><b>Tvärsektioner.</b> På en provsträcka beräknas tvärsektioner var 10:e meter, 30 m åt vardera hållet från vägens mitt. Vägens mitt är medelvärdet av mätbilens spår i de två riktningarna. Markhöjden hämtas ur Lantmäteriets markhöjdmodell (1 m grid, laserskannad). Vägkanten är släntens brytpunkt, alltså där profilen kröker mest, nära den vägbredd PMS anger. Höjdmodellen rundar av vägrenen, så en ren lutningsgräns missar ofta kanten. Syns ingen brytpunkt sätts kanten efter PMS vägbredd, och det märks i visaren. Därifrån följs innerslänten ner till dikesbotten (eller till bankens fot) och ytterslänten upp till där marken planar ut. Lutningen anges som 1:n mellan de punkterna. Fasta föremål är laserpunkter från Lantmäteriets laserdata (skog) som ligger mer än 2,5 m över marken, inom ±5 m längs vägen. Föremålets typ skattas ur punktklustrets form: smalt och högt är stolpe, punkter i ett plan (tak) är byggnad, bara högt belägna punkter är ledning, lägre än 4 m är buskage och övrigt träd. Laserdatat är oklassat ovan mark och glest, så typen är en skattning. Där en stolpe eller ett vägmärke syns i vägbilderna på samma plats används den typen. Avståndet mäts vågrätt från vägkanten. Säkerhetszonen följer TDOK 2018:0644. Sidoområdet ingår som två egna kontroller som kan tändas och släckas: <i>släntlutning</i> (innerslänt brantare än 1:2 utan räcke är brist; 1:2–1:2,3 redovisas som osäkert på grund av höjdmodellens mätosäkerhet) och <i>föremål i säkerhetszonen</i> (fast föremål innanför zonen utan räcke är brist). De bedöms bara på provsträckorna. Höjdmodellen jämnar ut diken som är smalare än cirka 1 m, så grunda och smala diken kan underskattas. Stolpar och smala föremål kan saknas i laserdata. Räcken hämtas ur NVDB via PMS, per sida och sträcka. NVDB anger inte räckets läge i sidled, så det ritas strax innanför vägkanten. Bakom ett räcke bedöms inte slänter och föremål, eftersom räcket skyddar mot dem. Laserdatat för skog har bara 1–3 punkter per m², och det räcker inte för att se en räckesbalk. Där NVDB anger räcke finns inga laserpunkter alls på räckeshöjd. På broar visar höjdmodellen marken under bron, så de sektionerna bedöms inte.</p>
<p><b>Vägbilder.</b> PMS framåtbilder (båda körriktningarna, var 20:e meter) har tolkats automatiskt med en öppen segmenteringsmodell, Mask2Former tränad på Mapillary Vistas. Den skiljer på räcke, stolpe, skylt, staket och vegetation. För varje sektion används bilder tagna 6–26 m före sektionen. Andelen räckespixlar i bildens närområde på respektive sida avgör om räcke syns, och resultatet jämförs med NVDB. Själva fotot hämtas direkt från PMS när sektionen visas, och tolkningen läggs ovanpå. Google Street View används bara som länk, eftersom Googles villkor inte tillåter automatisk analys av deras bilder.</p>
<p><b>Riskfaktorer (inte krav i VGU).</b> Tre faktorer som kan förstärka olycksrisken går att slå på under Vad bedöms. De är avstängda från början och ingår inte i bedömningen mot kraven. <i>Kurvföljd:</i> förväntad hastighet (V85) räknas för varje kurva med Lamms modell, V85 = 10<sup>6</sup>/(8 270 + 8,01·CCR) med CCR = 63 700/R gon/km. På raklinjer längre än 250 m antas V85 vara skyltad hastighet + 10 km/h. En sänkning in i kurvan på mer än 20 km/h ger brist och 10–20 km/h osäkert (Lamms kriterium II). <i>Vattensamling:</i> ur PMS vattenarea i spåren (20 m), spårdjup max17 (100 m) och resulterande lutning, alltså roten ur tvärfall² + längslutning², för båda riktningarna. Vattenarea minst 1 dm² eller resulterande lutning under 0,5 % ger brist. Vattenarea 0,5–1 dm², lutning 0,5–1 % eller spårdjup minst 20 mm ger osäkert. <i>Solbländning:</i> solens läge räknas var 10:e minut under ett år (NOAA:s formler) och jämförs med blickriktningen för varje 20-metersavsnitt och körriktning. Blickriktningen räknas i verklig nordriktning, och längslutningen tas med. Bländning räknas när solen står högst 15° över blickriktningen (ovanför skyddar solskyddet) och högst 20° åt sidan. Här ger lågt vintersolsken och morgon- och kvällssol vår och höst ofta 150–350 timmar per år och riktning. Det gäller molnfritt väder, och skugga från terräng och skog ingår inte. Bländningen flaggas bara där den kommer strax före en för snäv kurva eller kort sikt över krön (inom 200 m): minst 250 timmar ger brist och minst 150 timmar osäkert.</p>
<p><b>Sikt i plan och profil.</b> På provsträckorna räknas sikten med siktlinjer i körfältets mitt, från ett öga 1,10 m över vägbanan till ett hinder 0,35 m högt, var 10:e meter och i båda riktningarna, upp till 300 m. Linjen är skymd om marken i höjdmodellen når upp till den (krön, skärning) eller om laserpunkter ovan mark ligger inom 0,5 m från linjen och når dess höjd (träd, buskar, byggnader). Vägbanans höjd tas från PMS-profilen, så att broar blir rätt. Låg vegetation under 1,5 m redovisas separat, eftersom den normalt röjs. Tillgänglig sikt jämförs med stoppsikten för vald kravnivå. Laserdatat är från skanningen för skog (2019–2023), så vegetationen kan ha växt eller röjts sedan dess.</p>
<p><b>Vägmärken och stolpar.</b> Komponenter av klassen stolpe/skylt i bildtolkningen hittas automatiskt i PMS-bilderna. Varje fynd har sedan hämtats i högre upplösning (1 280 × 960 pixlar) och tolkats manuellt till märkestyp med kod enligt vägmärkesförordningen, till exempel A2 Varning för flera farliga kurvor, C31-8 Hastighetsbegränsning 80 km/h, F5 Vägvisare och X1 Markeringspil. Fynden som inte var märken har sorterats bort eller fått sin rätta typ (kantstolpe, ledningsstolpe, belysningsstolpe, baksida av märke). Av 305 automatiska fynd var 60 feltolkningar, bland annat suddade fordon. Några få tolkningar är markerade som osäkra. Läget skattas med en enkel kameramodell och är ungefärligt (±30 % i avstånd). Små och avlägsna märken kan missas, så listan är inte en fullständig inventering. Märkena visas på kartan när man zoomar in och kan klickas för bild och typ. För snäva kurvor (brist eller osäkert, eller R under 400 m) visas var ingen kurvvarning (A1, A2) eller markeringspil (X1) syns inom 300 m före kurvan, som underlag för kontroll av skyltningen.</p>
<p><b>Belysning.</b> Alla vägar antas sakna belysning.</p>"""
# Uppmätta hastigheter (TomTom), om något uttag har lästs in
_tt = []
for namn_l, _lk, _pre, _rd, W_l, *_ in LAN:
    for f in sorted(Path(f'{W_l}/tomtom').glob('vag*.json')):
        g = json.loads(f.read_text()); s = g['segment']; u = g['uttag'][0] if g['uttag'] else {}
        for r in (1, 2):
            q = [x for x in s if x['r'] == r]
            if q:
                _tt.append(f"{vagnamn(int(f.stem[3:]))} i {namn_l}, km {min(x['s0'] for x in q)/1000:.1f}–{max(x['s1'] for x in q)/1000:.1f} mot {'ökande' if r == 1 else 'minskande'} km "
                           f"({len(q)} segment, {', '.join(u.get('period', []))}, {', '.join(u.get('tider', [])).replace('0:00-24:00', 'hela dygnet')}, i median {format(round(float(np.median([x['n'] for x in q]))), ',').replace(',', ' ')} mätningar per segment)")
if _tt:
    metod += f"""<p><b>Uppmätt hastighet (TomTom).</b> Hastighetsfördelningen kommer från TomTom Traffic Stats (Route Analysis), som bygger på anonymiserade GPS-spår från fordon och telefoner. Uttaget ger för varje vägsegment percentilerna 5–95 %, median, medelvärde och antal mätningar. Underlag: {'; '.join(_tt)}. Segmentens punkter har projicerats på mätbilens GNSS-spår (median 2 m från spåret), och varje 20-metersavsnitt får värdet från det segment som täcker avsnittets mitt. Segmenten är 5–800 m långa, så hastigheten inom ett segment är ett medelvärde och en kort inbromsning före en kurva syns bara när kurvan ligger i ett eget segment.
Riskfaktorn <i>Uppmätt hastighet</i> jämför V85 (den hastighet som 85 % av fordonen håller sig under) med den högsta hastighet där geometrin säkert klarar kraven, räknat med de krav som är valda (kurvradie, skevning, krön och svackor, stoppsikt). Brist: V85 ligger minst 10 km/h över den hastighet geometrin möjligen klarar. Osäkert: V85 ligger mer än 5 km/h över den hastighet geometrin säkert klarar. I <i>Kurvföljd</i> ersätts Lamms beräknade hastighetssänkning med den uppmätta (högsta V85 på 300 m före kurvan minus lägsta V85 i kurvan) där kurvan och tillfarten ligger i olika segment. Uppmätt hastighet ingår inte i den klarade hastigheten. Kartläget <i>Uppmätt fart</i> visar V85 minus skyltad hastighet.</p>"""
kf = Path('/home/claude/work/kvalitet/kvalitet.json')
if kf.exists():
    Q = json.loads(kf.read_text()); hq = Q['horisontal']; vq = Q.get('vertikal', {})
    rader_t = ''.join(f"<tr><td>{vagnamn(t['vag']) if 'vagnamn' in globals() else t['vag']}</td><td class='n'>{t['sektioner']}</td><td class='n'>{t['vagbredd_pms']:.1f}</td><td class='n'>{t['tolkad_bredd_median']:.1f}</td><td class='n'>{t.get('stodremsa_per_sida', 0):+.1f}</td><td class='n'>{round(100*t['andel_kant_ej_funnen'])} %</td></tr>" for t in Q['tvarsektioner'])
    rader_p = ''.join(f"<tr><td>{p['vag']}</td><td class='n'>{p['niva_pms_minus_dtm']:+.2f}</td><td class='n'>{p['form_rms_200m']:.3f}</td></tr>" for p in Q['profil'] if p.get('form_rms_200m') is not None)
    metod += f"""<p><b>Kvalitetskontroll.</b> Radierna har kontrollerats mot källor som är oberoende av PMS-kurvaturen. <i>Horisontalradier</i> ({hq['antal']} kurvor, R 50–5 000 m) har jämförts med en cirkel anpassad till mätbilens GNSS-spår. Medianförhållandet är {hq['median_gnss']:.3f}, och {round(100*hq['inom10_gnss'])} % ligger inom ±10 %. De två körriktningarna stämmer inom ±10 % i {round(100*hq['inom10_riktningar'])} % av kurvorna. De största avvikelserna gäller sammansatta kurvor, där cirkelanpassningen ger en medelradie. I {hq['avvikare']} kurvor gav motorn en större radie än GNSS-spåret och båda riktningarna stödde. Därför omfattar osäkerhetsintervallet nu alltid båda riktningarnas radie och en pålitlig GNSS-radie, så att en kurva där källorna är oense redovisas som osäker i stället för klarar.</p>
{precision_html(Q)}
<p><i>Vertikalradier</i> längs hela vägarna ({vq.get('antal', 0)} kurvor) har jämförts med Lantmäteriets höjdmodell i mätbilens spår. Medianförhållandet är {vq.get('median_kvot', float('nan')):.2f}, osäkerhetsintervallen överlappar i {round(100*vq.get('intervall_overlappar', 0))} % och kurvtypen (krön eller svacka) stämmer i {round(100*vq.get('samma_typ', 0))} %. Motorn ger i regel något mindre radier än höjdmodellen, alltså åt det försiktiga hållet. <i>Längdprofilens</i> form stämmer med höjdmodellen (avvikelse efter att nivån tagits bort, över 200 m):</p>
<table><thead><tr><th>Väg</th><th>PMS minus höjdmodell (m)</th><th>Formavvikelse (m)</th></tr></thead><tbody>{rader_p}</tbody></table>
<p><i>Tvärsektionernas</i> släntkrön har jämförts med vägbredden i PMS. På väg 83 sammanfaller de. På övriga vägar ligger släntkrönen i regel 0,5–1 m utanför den belagda ytan på varje sida, och vägbilderna visar att det är grusad stödremsa. Därför skiljer visaren på körbanekant (PMS vägbredd) och släntkrön, och säkerhetszonen räknas från körbanekanten. På motorvägen (E4) följer sektionen körbanan i ökande km, och vänster sida är mittremsan, där kanten oftast inte syns i höjdmodellen.</p>
<p><i>Sidoområdena</i> har kontrollerats mot vägbilderna var 100:e meter. Bildmodellen avgör om det står skog (vegetation som når över horisonten) på varje sida, och det jämförs med om lasern hittar föremål över 2,5 m inom 15 m från körbanekanten. Laser och bild är överens om skog eller öppet i {round(100*np.mean([x['lika'] for x in Q.get('sidoomraden',[])]) ) if Q.get('sidoomraden') else '–'} % av fallen ({sum(x['n'] for x in Q.get('sidoomraden',[]))} sidor). Fall där lasern ser skog men bilden inte gör det är mycket ovanliga ({sum(x['laser_skog_bild_oppet'] for x in Q.get('sidoomraden',[]))} st). De flesta skillnaderna är skog längre bort än 15 m, som syns i bilden men ligger utanför laserns gräns. Avståndet till trädkanten kan inte mätas lika säkert i bilden, eftersom bildmodellen räknar in buskar och sly. Ett stickprov där vägbilder granskades manuellt mot laserns sidomått gav rimliga värden i 10 av 12 fall.</p>
<table><thead><tr><th>Väg</th><th>Sektioner</th><th>PMS belagd bredd, median (m)</th><th>Krön till krön, median (m)</th><th>Stödremsa per sida (m)</th><th>Krön ej funnet</th></tr></thead><tbody>{rader_t}</tbody></table>"""
meta = dict(vagar=vagar, data=data, metod=metod, stoppsikt=stoppsikt, bakgrund=bakgrund, tvar=tvar, tvarMeta=tvar_meta,
            rubrik='Västernorrland och Jämtland',
            kalla='Underlag: Trafikverkets vägytemätningar i PMSv4 (2024–2025) och VGU, TRVINFRA-00396 version 1.0. Resultatet är en screening av geometrin, inte en fullständig VGU-granskning.')
html = MALL.read_text(encoding='utf-8').replace('__DATA__', json.dumps(meta, ensure_ascii=False).replace('</', '<\\/'))
(UT / 'vagdiagnos.html').write_text(html, encoding='utf-8')
print('HTML', len(html) // 1024, 'kB')
if celltabeller:
    alla = pd.concat(celltabeller, ignore_index=True)
    alla.to_csv(UT / 'diagnos_20m.csv', index=False, encoding='utf-8-sig')
