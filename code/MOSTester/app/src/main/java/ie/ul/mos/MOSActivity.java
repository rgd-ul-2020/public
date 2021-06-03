package ie.ul.mos;

import android.content.Context;
import android.content.Intent;
import android.content.res.TypedArray;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.media.AudioAttributes;
import android.media.AudioManager;
import android.media.MediaPlayer;
import android.os.Bundle;
import android.os.CountDownTimer;
import android.os.PowerManager;
import android.support.v7.app.AppCompatActivity;
import android.view.View;
import android.widget.Button;
import android.widget.ProgressBar;
import android.widget.RadioGroup;
import android.widget.TextView;
import android.widget.Toast;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;

public class MOSActivity extends AppCompatActivity implements SensorEventListener
{
    private MediaPlayer        mediaPlayer;
    private ArrayList<Integer> audioList;
    private ArrayList<Object>  copyList;
    private HashMap            scoreList;
    private int                currentAudio;
    private boolean            audioPlayed;
    private boolean            isPreview;

    private SensorManager sensorManager;
    private Sensor        proximity;

    private PowerManager          powerManager;
    private PowerManager.WakeLock wakeLock;

    @Override
    protected void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_mos);

        isPreview = (boolean) getIntent().getExtras().get("preview");
        int resArray = isPreview ? R.array.preview : R.array.test01;

        TypedArray resourceList;
        resourceList = getResources().obtainTypedArray(resArray);
        audioList    = new ArrayList<>();
        copyList     = new ArrayList<>();

        int max = resourceList.length();
        for (int i = 0; i < max; ++i) {
            int id = resourceList.getResourceId(i, 0);

            if (id > 0) {
                audioList.add(id);
                copyList.add(id);
            }
        }

        resourceList.recycle();
        Collections.shuffle(audioList);

        scoreList = new HashMap<Integer, Integer>();

        currentAudio = 0;

        loadAudio(currentAudio);

        Button playButton = (Button) findViewById(R.id.playBtn);
        playButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Button playButton = (Button) findViewById(R.id.playBtn);
                int height = playButton.getHeight();
                playButton.setVisibility(View.GONE);

                ProgressBar playProgress = (ProgressBar) findViewById(R.id.playProgress);
                playProgress.setVisibility(View.VISIBLE);
                playProgress.setMinimumHeight(height);

                playProgress.setProgress(0);
                playProgress.setMax(mediaPlayer.getDuration());

                new CountDownTimer(3000, 3000) {
                    @Override
                    public void onTick(long l) {}

                    public void onFinish() {
                        new CountDownTimer(mediaPlayer.getDuration(), 250) {
                            public void onTick(long millisUntilFinished) {
                                ProgressBar playProgress = (ProgressBar) findViewById(R.id.playProgress);
                                playProgress.setProgress(mediaPlayer.getCurrentPosition());
                            }
                            public void onFinish() {
                                audioPlayed = true;

                                ProgressBar playProgress = (ProgressBar) findViewById(R.id.playProgress);
                                playProgress.setProgress(mediaPlayer.getDuration());
                            }
                        }.start();

                        mediaPlayer.start();
                    }
                }.start();
            }
        });

        Button nextButton = (Button) findViewById(R.id.button);
        nextButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (!audioPlayed) {
                    String errmsg = "Please listen to the speech sample first.";
                    Toast.makeText(MOSActivity.this, errmsg, Toast.LENGTH_LONG).show();
                    return;
                }

                if (!saveAnswer()) {
                    String errmsg = "Please rate the speech sample.";
                    Toast.makeText(MOSActivity.this, errmsg, Toast.LENGTH_LONG).show();
                    return;
                }

                unloadAudio();

                currentAudio++;

                if (currentAudio < audioList.size()) {
                    loadAudio(currentAudio);

                } else {
                    endTest();
                }
            }
        });

        sensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        proximity     = sensorManager.getDefaultSensor(Sensor.TYPE_PROXIMITY);

        powerManager = (PowerManager) getSystemService(Context.POWER_SERVICE);
        wakeLock     = powerManager.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "MYTAG");
    }

    protected void loadAudio(int index)
    {
        audioPlayed = false;

        TextView text = (TextView) findViewById(R.id.testProgress);
        text.setText("Test " + (currentAudio + 1) + " of " + audioList.size());

        Button playButton = (Button) findViewById(R.id.playBtn);
        playButton.setVisibility(View.VISIBLE);

        ProgressBar playProgress = (ProgressBar) findViewById(R.id.playProgress);
        playProgress.setVisibility(View.GONE);

        AudioAttributes audioAttrs = new AudioAttributes.Builder()
                .setUsage(AudioAttributes.USAGE_VOICE_COMMUNICATION)
                .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                .build();

        AudioManager audioman = (AudioManager) getSystemService(Context.AUDIO_SERVICE);

        int volume = (int) (0.8 * audioman.getStreamMaxVolume(AudioManager.STREAM_VOICE_CALL));
        audioman.setStreamVolume(AudioManager.STREAM_VOICE_CALL, volume, 0);

        mediaPlayer = MediaPlayer.create(this, audioList.get(index), audioAttrs,
                audioman.generateAudioSessionId());

        ProgressBar progressBar = (ProgressBar) findViewById(R.id.playProgress);
        progressBar.setProgress(currentAudio);
        progressBar.setMax(audioList.size());
    }

    protected boolean saveAnswer()
    {
        RadioGroup group = (RadioGroup) findViewById(R.id.groupMos);
        int selected = group.getCheckedRadioButtonId();
        int answer   = 0;

        if      (selected == R.id.radio5) { answer = 5; }
        else if (selected == R.id.radio4) { answer = 4; }
        else if (selected == R.id.radio3) { answer = 3; }
        else if (selected == R.id.radio2) { answer = 2; }
        else if (selected == R.id.radio1) { answer = 1; }

        if (answer == 0) {
            return false;
        }

        scoreList.put(audioList.get(currentAudio), answer);
        group.clearCheck();

        return true;
    }

    protected void unloadAudio()
    {
        if (mediaPlayer != null) {
            if (mediaPlayer.isPlaying()) {
                mediaPlayer.stop();
            }

            mediaPlayer.release();
        }
    }

    protected void endTest()
    {
        int max = scoreList.size() / 2;

        int[][] resultList = new int[max][2];

        for (int i = 0; i < max; ++i) {
            resultList[i][0] = (int) scoreList.get(copyList.get(2 * i));
            resultList[i][1] = (int) scoreList.get(copyList.get(2 * i + 1));
        }

        Intent intent = new Intent(MOSActivity.this, EndActivity.class);
        intent.putExtra("result_list", resultList);
        intent.putExtra("preview",     isPreview);
        startActivity(intent);
    }

    @Override
    public void onSensorChanged(SensorEvent sensorEvent)
    {
        if (sensorEvent.values[0] <= 3) {
            wakeLock.acquire();

            Toast.makeText(MOSActivity.this, "Perto", Toast.LENGTH_LONG).show();
        } else {
            if (wakeLock.isHeld()) {
                wakeLock.release();
            }

            Toast.makeText(MOSActivity.this, "Longe", Toast.LENGTH_LONG).show();
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int i)
    {
        // do nothing
    }

    protected void onResume()
    {
        super.onResume();
        sensorManager.registerListener(this, proximity, SensorManager.SENSOR_DELAY_NORMAL);
    }

    protected void onPause()
    {
        super.onPause();
        sensorManager.unregisterListener(this);
    }
}
