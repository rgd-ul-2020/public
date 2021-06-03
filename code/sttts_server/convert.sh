#!/bin/bash

set -e

#rm -rf sounds/3_normalized
rm -rf sounds/4_compressed

#mkdir sounds/3_normalized
mkdir -p sounds/4_compressed/pcm_alaw
mkdir -p sounds/4_compressed/speex
mkdir -p sounds/4_compressed/opus
mkdir -p sounds/4_compressed/deepspeech
mkdir -p sounds/4_compressed/deepspeech2

IN_DIR=sounds/3_normalized
OUT_DIR=sounds/4_compressed

for F in `ls sounds/2_resampled`; do
#    normalizer "2_resampled/${F}" "3_normalized/${F}" -23
    echo "${F}"
    ffmpeg -i "${IN_DIR}/${F}" -acodec pcm_alaw -ar 8000 "${OUT_DIR}/pcm_alaw/${F}" 2> /dev/null
    speexenc --abr 64 "${IN_DIR}/${F}" "${OUT_DIR}/speex/${F}.spx" # 2> /dev/null
    opusenc --vbr --bitrate 8 "${IN_DIR}/${F}" "${OUT_DIR}/opus/${F}.opus" 2> /dev/null
    cp "../SpeechPlayer/app/src/main/res/raw/txt_$(echo ${F} | cut -c1-3).txt" "${OUT_DIR}/deepspeech/${F}.txt"
    ./stt.py "${IN_DIR}/${F}" > "${OUT_DIR}/deepspeech2/${F}.txt" 2> /dev/null 
done
