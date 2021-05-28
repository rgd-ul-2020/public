#!/bin/bash

case $1 in
    f2ch)
        echo "PCM to UINT8:"
        bin/f2ch-enc inputs/test.wav | bin/f2ch-dec outputs/f2ch.wav
        ;;

    alaw)
        echo "PCM to ALAW:"
        bin/alaw-enc inputs/test.wav | bin/alaw-dec outputs/alaw.wav
        ;;

    mulaw)
        echo "PCM to mu-LAW:"
        bin/mulaw-enc inputs/test.wav | bin/mulaw-dec outputs/mulaw.wav
        ;;

    1bit)
        echo "PCM to 1bit:"
        bin/1bit-enc inputs/test.wav | bin/1bit-dec outputs/1bit.wav
        ;;

    2bit)
        echo "PCM to 2bit:"
        bin/2bit-enc inputs/test.wav | bin/2bit-dec outputs/2bit.wav
        ;;

    *)
        echo "Please select one codec."
        ;;
esac
