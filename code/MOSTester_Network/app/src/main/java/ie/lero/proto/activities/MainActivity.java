package ie.lero.proto.activities;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.media.AudioManager;
import android.os.Bundle;

import android.os.Handler;
import android.os.PowerManager;
import android.speech.tts.TextToSpeech;
import android.support.annotation.NonNull;
import android.support.annotation.Nullable;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import com.chibde.visualizer.LineBarVisualizer;

import java.net.InetAddress;
import java.nio.ByteBuffer;
import java.util.Locale;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import ie.lero.proto.AudioPlayer;
import ie.lero.proto.Protocol;
import ie.lero.proto.R;
import ie.lero.proto.SingleThreadExecutor;

public class MainActivity extends AppCompatActivity implements SensorEventListener
{
    static private final String TAG = "MAIN_ACTIVITY";

    protected AudioPlayer audioPlayer;
    protected Protocol protocol;
    protected TextToSpeech tts;

    private boolean btn_connect_mode = true;
    private SensorManager sensorManager;
    private PowerManager.WakeLock wakeLock;
    private Sensor proximity;
    private float volume = 0.50f;
    private boolean fade;
    private String remoteAddress;

    @Override
    protected void onCreate(@Nullable final Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Bundle extras = getIntent().getExtras();
        remoteAddress = (String) extras.get("remote_address");

        initialize();
    }

    void initialize()
    {
        initializeProtocol();

        AudioManager am = (AudioManager) getSystemService(AUDIO_SERVICE);
        int stream_type = AudioManager.STREAM_VOICE_CALL;

        setVolumeControlStream(stream_type);
        am.setStreamVolume(stream_type, (int) volume * am.getStreamMaxVolume(stream_type), 0);

        audioPlayer = new AudioPlayer();

        LineBarVisualizer visualizer = findViewById(R.id.visualizer);
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

        Button connect_btn = findViewById(R.id.connect_btn);
        connect_btn.setOnClickListener(new View.OnClickListener()
        {
            @Override
            public void onClick(View v)
            {
                if (audioPlayer.isPlaying() || tts.isSpeaking()) {
                    return;
                }

                Handler handler = new Handler();
                handler.postDelayed(new Runnable()
                {
                    @Override
                    public void run()
                    {
                        protocol.getNext();
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

    protected void initializeProtocol()
    {
        protocol = new Protocol(8080);
        protocol.setListener(new Protocol.Listener()
        {
            @Override
            protected void onCommand(final int command)
            {
                String text = "";

                switch (command) {
                    case Protocol.COMMAND_HANDSHAKE:
                    case Protocol.COMMAND_CONNECT:
                        text = "CONNECTED";
                        fade = true;
                        btn_connect_mode = false;
                        updateButtonInUI(R.id.connect_btn, "NEXT", true);
                        break;

                    case Protocol.COMMAND_DISCONNECT:
                        text = "DISCONNECTED";
                        btn_connect_mode = true;
                        updateButtonInUI(R.id.connect_btn, "CONNECT", true);
                        break;

                    case Protocol.COMMAND_EOF:
                        fade = true;
                        updateButtonInUI(R.id.connect_btn, "NEXT", true);
                        break;

                    case Protocol.COMMAND_EOT:
                        text = "END OF TEST";
                        btn_connect_mode = true;

                        Intent intent = new Intent(MainActivity.this, EndActivity.class);
                        MainActivity.this.startActivity(intent);

                        updateButtonInUI(R.id.connect_btn, "CONNECT", true);
                        break;

                    default:
                        text = "UNKNOWN";
                        break;
                }
            }

            @Override
            protected void onAudio(ByteBuffer buffer)
            {
                if (fade) {
                    audioPlayer.queue(buffer, 100);
                    fade = false;
                } else {
                    audioPlayer.queue(buffer);
                }
            }

            @Override
            protected void onText(String text)
            {
                Bundle params = new Bundle();
                params.putInt(TextToSpeech.Engine.KEY_PARAM_STREAM, AudioManager.STREAM_VOICE_CALL);
                params.putFloat(TextToSpeech.Engine.KEY_PARAM_VOLUME, volume);
                params.putInt(TextToSpeech.Engine.KEY_PARAM_SESSION_ID, audioPlayer.getSessionId());
                tts.speak(text, TextToSpeech.QUEUE_ADD, params, text);
            }
        });
        protocol.listen();
        protocol.connect(remoteAddress);
    }

    private void updateButtonInUI(final int viewId, final String text, final boolean clickable)
    {
        runOnUiThread(new Runnable()
        {
            @Override
            public void run()
            {
                Button btn = MainActivity.this.findViewById(viewId);
                btn.setText(text);
                btn.setClickable(clickable);
            }
        });
    }

    private void updateTextViewInUI(final int viewId, final String text, final boolean append)
    {
        runOnUiThread(new Runnable()
        {
            @Override
            public void run()
            {
                TextView textView = findViewById(viewId);
                if (append) {
                    textView.append(text);
                } else {
                    textView.setText(text);
                }
            }
        });
    }

    @Override
    public void onSensorChanged(SensorEvent sensorEvent)
    {
        if (wakeLock != null) {
            if (sensorEvent.values[0] <= 3) {
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

    protected void onResume()
    {
        super.onResume();

        if (sensorManager != null) {
            sensorManager.registerListener(this, proximity, SensorManager.SENSOR_DELAY_NORMAL);
        }
    }

    protected void onPause()
    {
        super.onPause();

        if (sensorManager != null) {
            sensorManager.unregisterListener(this);
        }
    }
}