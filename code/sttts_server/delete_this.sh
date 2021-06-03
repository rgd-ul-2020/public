#!/bin/bash

set -e

COUNT=0
SUM_SEC=0
PCM_SUM=0
SPX_SUM=0
OPS_SUM=0
TXT_SUM=0

while read FILE SECS; do
    PCM_SZ=$(ls -ls "sounds/4_compressed/pcm_alaw/$FILE" | awk "{ print \$6 }")
    SPX_SZ=$(ls -ls "sounds/4_compressed/speex/$FILE.spx" | awk "{ print \$6 }")
    OPS_SZ=$(ls -ls "sounds/4_compressed/opus/$FILE.opus" | awk "{ print \$6 }")
    TXT_SZ=$(ls -ls "sounds/4_compressed/deepspeech/$FILE.txt" | awk "{ print \$6 }")

    COUNT=$(expr $COUNT + 1)
    SUM_SEC=$(echo "${SUM_SEC} + ${SECS}" | bc)
    PCM_SUM=$(echo "${PCM_SUM} + ${PCM_SZ}" | bc)
    SPX_SUM=$(echo "${SPX_SUM} + ${SPX_SZ}" | bc)
    OPS_SUM=$(echo "${OPS_SUM} + ${OPS_SZ}" | bc)
    TXT_SUM=$(echo "${TXT_SUM} + ${TXT_SZ}" | bc)

    printf "%f\t%f\t%f\t%f\t%f\t%d\t%d\t%d\t%d\n" "$SECS"    \
        "$(echo "$PCM_SUM / $SUM_SEC" | bc)" \
        "$(echo "$SPX_SUM / $SUM_SEC" | bc)" \
        "$(echo "$OPS_SUM / $SUM_SEC" | bc)" \
        "$(echo "$TXT_SUM / $SUM_SEC" | bc)" \
        "$(echo "$PCM_SUM / $COUNT" | bc)" \
        "$(echo "$SPX_SUM / $COUNT" | bc)" \
        "$(echo "$OPS_SUM / $COUNT" | bc)" \
        "$(echo "$TXT_SUM / $COUNT" | bc)" 
done < out
