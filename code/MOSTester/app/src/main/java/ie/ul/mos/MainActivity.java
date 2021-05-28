package ie.ul.mos;

import android.content.Context;
import android.content.Intent;
import android.content.res.Resources;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Build;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.text.Html;
import android.util.Log;
import android.view.Gravity;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.TableLayout;
import android.widget.TableRow;
import android.widget.TextView;

/**
 * Created by rafael on 09/10/17.
 */

public class MainActivity extends AppCompatActivity
{
    @Override
    public void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        String    intro_text = "<center>Instructions</center><p>In this experiment you will be evaluating a series of speech samples based on their quality.</p><p>After pressing 'OK', please follow these steps:</p><ul><li>Click play and listen to the speech sample.</li><li>Select a score for the sample.</li><li>Click next to load the next sample.</li></ul><p>Do not close the application after the end of the test.</p><p>Do not readjust the volume levels.</p>";
        TextView text = (TextView) findViewById(R.id.intro_text);

        if (Build.VERSION.SDK_INT >= 24) {
            text.setText(Html.fromHtml(intro_text, Html.FROM_HTML_MODE_LEGACY));
        } else {
            text.setText(Html.fromHtml(intro_text));
        }

        Button button = (Button) findViewById(R.id.intro_button);
        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(MainActivity.this, MOSActivity.class);
                intent.putExtra("preview", true);
                startActivity(intent);
            }
        });

        WindowManager.LayoutParams params = getWindow().getAttributes();
        params.flags |= WindowManager.LayoutParams.FLAG_FULLSCREEN;
        params.flags |= WindowManager.LayoutParams.FLAG_IGNORE_CHEEK_PRESSES;
        getWindow().setAttributes(params);
    }
}
