package ie.lero.proto;

import android.media.AudioFormat;
import android.media.AudioManager;
import android.media.AudioTrack;
import android.util.Log;
import android.view.View;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.LinkedList;
import java.util.Queue;

public class AudioPlayer implements AudioTrack.OnPlaybackPositionUpdateListener
{
    private Queue<ByteBuffer> queue;
    private int SAMPLE_RATE = 8000;

    AudioTrack track;
    int minSize;
    boolean playing;

    public AudioPlayer()
    {
        queue = new LinkedList<>();
        playing = false;

        minSize = AudioTrack.getMinBufferSize(SAMPLE_RATE,
            AudioFormat.CHANNEL_OUT_MONO,
            AudioFormat.ENCODING_PCM_16BIT);

        track = new AudioTrack(AudioManager.STREAM_VOICE_CALL, SAMPLE_RATE,
            AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT,
            minSize, AudioTrack.MODE_STREAM);

        track.setPlaybackPositionUpdateListener(this);
    }

    static short alaw_decode_table[] = new short[] {
        -5504, -5248, -6016, -5760, -4480, -4224, -4992, -4736,
        -7552, -7296, -8064, -7808, -6528, -6272, -7040, -6784,
        -2752, -2624, -3008, -2880, -2240, -2112, -2496, -2368,
        -3776, -3648, -4032, -3904, -3264, -3136, -3520, -3392,
        -22016,-20992,-24064,-23040,-17920,-16896,-19968,-18944,
        -30208,-29184,-32256,-31232,-26112,-25088,-28160,-27136,
        -11008,-10496,-12032,-11520,-8960, -8448, -9984, -9472,
        -15104,-14592,-16128,-15616,-13056,-12544,-14080,-13568,
        -344,  -328,  -376,  -360,  -280,  -264,  -312,  -296,
        -472,  -456,  -504,  -488,  -408,  -392,  -440,  -424,
        -88,   -72,   -120,  -104,  -24,   -8,    -56,   -40,
        -216,  -200,  -248,  -232,  -152,  -136,  -184,  -168,
        -1376, -1312, -1504, -1440, -1120, -1056, -1248, -1184,
        -1888, -1824, -2016, -1952, -1632, -1568, -1760, -1696,
        -688,  -656,  -752,  -720,  -560,  -528,  -624,  -592,
        -944,  -912,  -1008, -976,  -816,  -784,  -880,  -848,
        5504,  5248,  6016,  5760,  4480,  4224,  4992,  4736,
        7552,  7296,  8064,  7808,  6528,  6272,  7040,  6784,
        2752,  2624,  3008,  2880,  2240,  2112,  2496,  2368,
        3776,  3648,  4032,  3904,  3264,  3136,  3520,  3392,
        22016, 20992, 24064, 23040, 17920, 16896, 19968, 18944,
        30208, 29184, 32256, 31232, 26112, 25088, 28160, 27136,
        11008, 10496, 12032, 11520, 8960,  8448,  9984,  9472,
        15104, 14592, 16128, 15616, 13056, 12544, 14080, 13568,
        344,   328,   376,   360,   280,   264,   312,   296,
        472,   456,   504,   488,   408,   392,   440,   424,
        88,    72,   120,   104,    24,     8,    56,    40,
        216,   200,   248,   232,   152,   136,   184,   168,
        1376,  1312,  1504,  1440,  1120,  1056,  1248,  1184,
        1888,  1824,  2016,  1952,  1632,  1568,  1760,  1696,
        688,   656,   752,   720,   560,   528,   624,   592,
        944,   912,  1008,   976,   816,   784,   880,   848
    };

    public void queue(ByteBuffer buffer)
    {
        queue(buffer, 0);
    }

    public void queue(ByteBuffer buffer, int fade_ms)
    {
        ByteBuffer newBuff = ByteBuffer.allocate(2 * buffer.limit());
        newBuff.order(ByteOrder.LITTLE_ENDIAN);

        int samples      = 0;
        int sample_inc   = SAMPLE_RATE / 1000;
        int samples_fade = sample_inc * fade_ms;

        while (buffer.position() < buffer.limit()) {
            short sample = alaw_decode_table[buffer.get() + 128];

            if (samples < samples_fade) {
                sample = (short) ((float) sample * (float) samples / (float) samples_fade);
            }

            newBuff.putShort(sample);
            samples += sample_inc;
        }
        newBuff.flip();

        queue.add(newBuff);

        if (track.getPlayState() != AudioTrack.PLAYSTATE_PLAYING) {
            playNext();
        }
    }

    private void playNext()
    {
        ByteBuffer next = queue.poll();

        if (next == null) {
            playing = false;
            track.stop();
            return;
        }

        playing = true;
        track.play();
        track.setNotificationMarkerPosition(next.limit() / 2);
        track.write(next.array(), next.arrayOffset(), next.limit());
    }

    public boolean isPlaying()
    {
        return playing;
    }

    @Override
    public void onMarkerReached(AudioTrack track)
    {
        track.setNotificationMarkerPosition(0);
        playNext();
    }

    @Override
    public void onPeriodicNotification(AudioTrack track)
    {

    }

    public int getSessionId()
    {
        return track.getAudioSessionId();
    }
}
