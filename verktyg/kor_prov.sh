#!/bin/bash
# Kör provsträckor: räcken, tvärsektioner, vägbilder. Inloggning via LM_USERNAME/LM_PASSWORD i miljön.
cd /home/claude/motor
PY=/home/claude/venv/bin/python
W=/home/claude/work
while read vn s0 s1; do
  echo "== väg $vn $s0-$s1 $(date +%T)"
  $PY verktyg/hamta_racke.py $vn $s0 $s1 $W/tvar/vag${vn}_racke.json > /dev/null || echo "räcke fel $vn"
  [ -f $W/tvar/vag${vn}.json ] || $PY verktyg/kor_tvarsektion.py $vn $s0 $s1 $W/tvar/vag${vn}.json || echo "tvärsektion fel $vn"
  [ -f $W/bilder/vag${vn}/index.json ] || $PY verktyg/hamta_vagbilder.py $vn $s0 $s1 $W/bilder/vag${vn} || echo "bilder fel $vn"
  echo "== klar $vn $(date +%T)"
done <<LISTA
348 84900 89900
352 62800 67800
4 35200 40200
90 154500 159500
335 46000 51000
86 52600 57600
331 15300 20300
14 26500 31500
87 22700 27700
LISTA
echo ALLT_KLART
