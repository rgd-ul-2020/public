package ie.lero.speechplayer.activities;

import android.content.Context;
import android.content.Intent;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.media.AudioManager;
import android.os.Environment;
import android.os.Handler;
import android.os.PowerManager;
import android.speech.tts.TextToSpeech;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.text.format.DateFormat;
import android.util.Log;
import android.view.View;
import android.widget.Button;

import com.chibde.visualizer.LineBarVisualizer;
import ie.lero.speechplayer.players.AudioPlayer;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.StringReader;
import java.lang.reflect.Field;
import java.nio.ByteBuffer;
import java.nio.ShortBuffer;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Date;
import java.util.Locale;

import ie.lero.speechplayer.R;

public class MainActivity extends AppCompatActivity implements SensorEventListener {

    static private final String TAG = "MAIN_ACTIVITY";

    private enum SampleType {
        UNINITIALIZED,
        TEXT,
        PCM,
    }

    private class Sample {
        public int id;
        public SampleType type;
        public double loss;

        public Sample(int id, SampleType type, float loss) {
            this.id   = id;
            this.type = type;
            this.loss = loss;
        }

        int getResourceId()
        {
            String format;

            switch (type) {
                case PCM:  format = "pcm_%03d"; break;
                case TEXT: format = "txt_%03d"; break;
                default:   format = "";
            }

            try {
                String fieldName = String.format(format, id);
                Field  field     = R.raw.class.getDeclaredField(fieldName);
                int    fieldInt  = field.getInt(field);
                return fieldInt;
            } catch (NoSuchFieldException e) {
                return -1;
            } catch (IllegalAccessException e) {
                return -2;
            }
        }
    };

    private float volume = 0.5f;

    private AudioPlayer audioPlayer;
    private TextToSpeech tts;

    private ArrayList<Sample> selectedSamples;
    private int selectedIndex;
    private boolean waitingForDelay;

    private SensorManager sensorManager;
    private Sensor proximity;
    private PowerManager.WakeLock wakeLock;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Create the full list
        ArrayList<Sample> samples1 = new ArrayList<>();
        for (int i = 0; i < 166; i++) {
            samples1.add(new Sample(i, SampleType.UNINITIALIZED, 0));
        }

        int max_samples = 70;

        // Pick 70 samples and add to the buckets
        ArrayList<Sample> samples2 = new ArrayList<>();
        for (int i = 0; i < max_samples; i++) {
            Sample sample = samples1.remove((int) (Math.random() * samples1.size()));

            switch (i % 4) {
                case 0:
                    sample.type = SampleType.TEXT;
                    sample.loss = 0;
                    break;

                case 1:
                    sample.type = SampleType.PCM;
                    sample.loss = 0;
                    break;

                case 2:
                    sample.type = SampleType.PCM;
                    sample.loss = 0.05;
                    break;

                case 3:
                    sample.type = SampleType.PCM;
                    sample.loss = 0.10;
                    break;
            }

            samples2.add(sample);
        }

        // Randomize the order.
        selectedSamples = new ArrayList<>();
        for (int i = 0; i < max_samples; i++) {
            selectedSamples.add(samples2.remove((int) (Math.random() * samples2.size())));
        }

        AudioManager am = (AudioManager) getSystemService(AUDIO_SERVICE);
        int stream_type = AudioManager.STREAM_VOICE_CALL;

        setVolumeControlStream(stream_type);
        am.setStreamVolume(stream_type, (int) volume * am.getStreamMaxVolume(stream_type), 0);

        audioPlayer = new AudioPlayer();

        final LineBarVisualizer visualizer = findViewById(R.id.visualizer);
        visualizer.setPlayer(audioPlayer.getSessionId());

        tts = new TextToSpeech(this, new TextToSpeech.OnInitListener()
        {
            @Override
            public void onInit(int status)
            {
                tts.setLanguage(Locale.ENGLISH);
                tts.setSpeechRate(0.75f);
            }
        });

        waitingForDelay = false;

        Button connect_btn = findViewById(R.id.connect_btn);
        connect_btn.setOnClickListener(new View.OnClickListener()
        {
            @Override
            public void onClick(View v)
            {
                if (waitingForDelay || audioPlayer.isPlaying() || tts.isSpeaking()) {
                    return;
                }

                Log.i(TAG, "NEXT CLICKED");
                waitingForDelay = true;

                if (selectedIndex >= selectedSamples.size()) {
                    Log.i(TAG, "END OF TEST");

                    String filename = DateFormat.format("yyyyMMdd-hhmmss", new Date()) + ".txt";

                    File externalStorage = Environment.getExternalStorageDirectory();
                    File outputDir       = new File(externalStorage, "speech_logs");
                    outputDir.mkdir();

                    File file = new File(outputDir, filename);

                    try (
                            OutputStream   os = new FileOutputStream(file);
                            BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(os));
                    ) {
                        for (Sample s : selectedSamples) {
                            bw.write(s.id + "\t");
                            bw.write(((s.type == SampleType.PCM) ? "PCM" : "TXT") + "\t");
                            bw.write( "" + s.loss);
                            bw.newLine();
                        }
                    } catch (Exception e) {
                        Log.e(TAG, "Exception recording order", e);
                    }

                    Log.i(TAG, "File \"" + filename + "\" saved at: " + getFilesDir());

                    Intent intent = new Intent(MainActivity.this, EndActivity.class);
                    MainActivity.this.startActivity(intent);
                    return;
                }

                Handler handler = new Handler();
                handler.postDelayed(new Runnable()
                {
                    @Override
                    public void run()
                    {
                        Log.i(TAG, "DELAY FINISHED");
                        Sample next = selectedSamples.get(selectedIndex);
                        selectedIndex++;
                        InputStream is = getResources().openRawResource(next.getResourceId());

                        switch (next.type) {
                            case PCM:
                                try {
                                    audioPlayer.play(is, next.loss);
                                    visualizer.invalidate();
                                } catch (Exception e) {
                                    Log.e(TAG, "Error loading PCM", e);
                                }
                                break;

                            case TEXT:
                                BufferedReader bis = new BufferedReader(new InputStreamReader(is));
                                try {
                                    String text = bis.readLine();

                                    Bundle params = new Bundle();
                                    params.putInt(TextToSpeech.Engine.KEY_PARAM_STREAM, AudioManager.STREAM_VOICE_CALL);
                                    params.putFloat(TextToSpeech.Engine.KEY_PARAM_VOLUME, volume);
                                    params.putInt(TextToSpeech.Engine.KEY_PARAM_SESSION_ID, audioPlayer.getSessionId());
                                    tts.speak(text, TextToSpeech.QUEUE_ADD, params, text);
                                } catch (Exception e) {
                                    Log.e(TAG, "Error loading TEXT", e);
                                }
                                break;
                        }

                        waitingForDelay = false;
                    }
                }, 2000);

                Button connect_btn = MainActivity.this.findViewById(R.id.connect_btn);
                connect_btn.setClickable(true);
            }
        });

        sensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        if (sensorManager != null) {
            proximity = sensorManager.getDefaultSensor(Sensor.TYPE_PROXIMITY);
        }

        PowerManager powerManager = (PowerManager) getSystemService(Context.POWER_SERVICE);
        if (powerManager != null) {
            wakeLock = powerManager.newWakeLock(PowerManager.PROXIMITY_SCREEN_OFF_WAKE_LOCK, TAG + ":wakelock");
        }
    }

    @Override
    public void onSensorChanged(SensorEvent event)
    {
        if (wakeLock != null) {
            if (event.values[0] <= 3) {
                wakeLock.acquire(5000);

            } else if (wakeLock.isHeld()) {
                wakeLock.release();
            }
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy)
    {
    }

    @Override
    protected void onResume()
    {
        super.onResume();

        if (sensorManager != null) {
            sensorManager.registerListener(this, proximity, SensorManager.SENSOR_DELAY_NORMAL);
        }
    }

    @Override
    protected void onPause()
    {
        super.onPause();

        if (sensorManager != null) {
            sensorManager.unregisterListener(this);
        }
    }
}
