package ie.lero.proto.activities;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.media.AudioManager;
import android.media.effect.Effect;
import android.speech.tts.TextToSpeech;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

import java.net.InetAddress;
import java.nio.ByteBuffer;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import ie.lero.proto.Protocol;
import ie.lero.proto.R;

public class StartActivity extends AppCompatActivity
{
    boolean hide_cfg = true;
    private Protocol protocol;

    @Override
    protected void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_start);

        boolean permissionsGranted = requestPermissions(new String[]{
            Manifest.permission.INTERNET,
            Manifest.permission.RECORD_AUDIO,
            Manifest.permission.WRITE_EXTERNAL_STORAGE,
            Manifest.permission.WAKE_LOCK,
        });

        if (permissionsGranted) {
            initialize();
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults)
    {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        initialize();
    }

    private boolean requestPermissions(String[] permissions)
    {
        boolean granted = true;

        for (int i = 0; i < permissions.length; i++) {
            int permission = ContextCompat.checkSelfPermission(this, permissions[i]);

            if (permission != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this, new String[]{
                    permissions[i]
                }, i);

                // Don't break yet! We need to request all permissions.
                granted = false;
            }
        }

        return granted;
    }

    protected void initialize()
    {
        initializeLocalAddress();

        Button btn_config = findViewById(R.id.btn_config);
        btn_config.setOnClickListener(new View.OnClickListener()
        {
            @Override
            public void onClick(View view)
            {
                toggleCfg();
            }
        });

        toggleCfg();

        Button btn_start = findViewById(R.id.btn_start);
        btn_start.setOnClickListener(new View.OnClickListener()
        {
            @Override
            public void onClick(View view)
            {
                EditText editText = findViewById(R.id.remote_addr);

                Intent intent = new Intent(StartActivity.this, MainActivity.class);
                intent.putExtra("remote_address", editText.getText().toString());
                StartActivity.this.startActivity(intent);
            }
        });
    }

    protected void initializeLocalAddress()
    {
        final ScheduledExecutorService executorService = Executors.newSingleThreadScheduledExecutor();
        executorService.schedule(new Runnable()
        {
            @Override
            public void run()
            {
                final InetAddress addr = protocol.getLocalAddress();

                if (addr == null) {
                    executorService.schedule(this, 100, TimeUnit.MILLISECONDS);
                    return;
                }

                String text = String.format(getString(R.string.local_addr), addr.getHostName());

                StartActivity.this.runOnUiThread(new Runnable()
                {
                    @Override
                    public void run()
                    {
                        TextView textView = findViewById(R.id.local_addr);
                        String   text     = String.format(getString(R.string.local_addr), addr.getHostName());
                        textView.setText(text);
                    }
                });
            }
        }, 0, TimeUnit.MILLISECONDS);
    }

    void toggleCfg()
    {
        TextView textView = findViewById(R.id.local_addr);
        EditText editText = findViewById(R.id.remote_addr);
        Button   button   = findViewById(R.id.btn_config);

        if (hide_cfg) {
            textView.setVisibility(View.GONE);
            editText.setVisibility(View.GONE);
            button.setText("Configure");
        } else {
            textView.setVisibility(View.VISIBLE);
            editText.setVisibility(View.VISIBLE);
            button.setText("Close");
        }

        hide_cfg = !hide_cfg;
    }
}
