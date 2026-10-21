#!/bin/bash
# Tolka vägbilder i tur och ordning när varje vägs bilder finns
cd /home/claude/motor
for vn in 83 348 352 4 90 335 86 331 14 87; do
  until [ -f /home/claude/work/bilder/vag${vn}/index.json ]; do sleep 20; done
  echo "tolkar $vn $(date +%T)"
  NO_PROXY=localhost no_proxy=localhost /home/claude/venv/bin/python verktyg/tolka_vagbilder.py /home/claude/work/bilder/vag${vn} > /home/claude/work/tolka_${vn}.log 2>&1
done
echo ALLT_TOLKAT
