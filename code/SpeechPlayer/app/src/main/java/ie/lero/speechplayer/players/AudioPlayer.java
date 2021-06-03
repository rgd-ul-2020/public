package ie.lero.speechplayer.players;

import android.media.AudioFormat;
import android.media.AudioManager;
import android.media.AudioTrack;
import android.media.session.PlaybackState;
import android.util.Log;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.LinkedList;
import java.util.Queue;

public class AudioPlayer implements AudioTrack.OnPlaybackPositionUpdateListener
{
    private InputStream stream;
    private int SAMPLE_RATE = 8000;

    private AudioTrack track;
    private int minSize;
    private boolean playing;

    public AudioPlayer()
    {
        playing = false;

        minSize = AudioTrack.getMinBufferSize(SAMPLE_RATE,
            AudioFormat.CHANNEL_OUT_MONO,
            AudioFormat.ENCODING_PCM_16BIT);

        track = new AudioTrack(AudioManager.STREAM_VOICE_CALL, SAMPLE_RATE,
            AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT,
            minSize, AudioTrack.MODE_STREAM);

        track.setPlaybackPositionUpdateListener(this);
    }

    static private short alaw_decode_table[] = new short[] {
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

    private byte[] translate(byte[] array, int length, int fadeInFrames)
    {
        ByteBuffer buffer = ByteBuffer.allocate(length * 2); // 1 byte to 2 bytes conversion
        buffer.order(ByteOrder.LITTLE_ENDIAN);

        for (int i = 0; i < length; i++) {
            short sample = alaw_decode_table[array[i] + 128];

            if (i < fadeInFrames) {
                sample = (short) (sample * i / fadeInFrames);
            }

            buffer.putShort(sample);
        }

        return buffer.array();
    }

    public void play(InputStream stream) throws IOException
    {
        play(stream, 0);
    }

    public void play(InputStream stream, double loss) throws IOException
    {
        play(stream, loss, 1);
    }

    // jitter in buffers
    public void play(InputStream stream, double loss, double jitter) throws IOException
    {
        playing = true;

        stream.skip(44);

        ByteArrayOutputStream baos = new ByteArrayOutputStream();

        int    read;
        byte[] buffer = new byte[800];
        boolean firstBlock = true;
        while ((read = stream.read(buffer)) > 0) {
            byte[] decoded = translate(buffer, read, firstBlock ? 100 : 0);
            if (Math.random() < loss) { decoded = new byte[(int) (jitter * decoded.length)]; }

            firstBlock = false;
            baos.write(decoded);
        }

        buffer = baos.toByteArray();
        track.play();
        track.write(buffer, 0, buffer.length);
    }

    public boolean isPlaying()
    {
        return track.getState() == AudioTrack.PLAYSTATE_PLAYING;
    }

    @Override
    public void onMarkerReached(AudioTrack track)
    {
        Log.e(AudioPlayer.class.getName(), "Marker: " + track.getPlaybackHeadPosition());
        playing = false;
    }

    @Override
    public void onPeriodicNotification(AudioTrack track)
    {
        Log.e(AudioPlayer.class.getName(), "Periodic: " + track.getPlaybackHeadPosition());
    }

    public int getSessionId()
    {
        return track.getAudioSessionId();
    }
}
