package ie.lero.proto;

import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.AsyncTask;

import java.nio.ShortBuffer;

public class AudioFeed extends AsyncTask<Void, Void, Void>
{
    static abstract class Listener
    {
        protected abstract void onBufferUpdate(ShortBuffer buffer);
    }

    private Listener listener;

    public AudioFeed()
    {
        this.listener = null;
    }

    public void setListener(Listener listener)
    {
        this.listener = listener;
    }

    @Override
    protected Void doInBackground(Void... voids)
    {
        final int SAMPLE_RATE = 16000;
        final int CHANNEL     = AudioFormat.CHANNEL_IN_MONO;
        final int ENCODING    = AudioFormat.ENCODING_PCM_16BIT;

        int bufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL, ENCODING);
        AudioRecord recorder = new AudioRecord(MediaRecorder.AudioSource.MIC, SAMPLE_RATE,
            CHANNEL, ENCODING, bufferSize);

        
        recorder.startRecording();

        ShortBuffer buffer = ShortBuffer.allocate(SAMPLE_RATE / 10);

        while (!isCancelled()) {
            buffer.clear();

            int read = recorder.read(buffer.array(), 0, buffer.limit());
            if (read < 0) {
                continue;
            }
            buffer.limit(read);

            listener.onBufferUpdate(buffer);
        }

        recorder.stop();
        return null;
    }
}
