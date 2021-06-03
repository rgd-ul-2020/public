#!/usr/bin/python3

import ffmpeg
import os
import sys

from gtts  import gTTS
from pydub import AudioSegment

i = 0

print('[', end='')
for line in sys.stdin:
    stripped  = line.rstrip()
    file_name = 'out/{:04d}_{}.'.format(i, stripped.replace(' ', '_').replace('.', ''))

    mp3_name = file_name + 'mp3'
    wav_name = file_name + 'wav'

    tts = gTTS(text=line, lang='en')
    tts.save(mp3_name)
    
    (ffmpeg
        .input(mp3_name)
        .output(wav_name, format='wav', acodec='pcm_s16le', ac=1, ar='16000')
        .run(quiet=True))

    os.unlink(mp3_name)
    
    print('{}{{"file_name":"{}","phrase":"{}"}}'.format('' if i == 0 else ',', wav_name, stripped), end='')
    print(wav_name, file=sys.stderr)
    i += 1
print(']')
