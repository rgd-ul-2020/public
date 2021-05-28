package ie.ul.mos;

import android.content.Intent;
import android.os.Bundle;
import android.os.PersistableBundle;
import android.support.annotation.Nullable;
import android.support.v7.app.ActionBar;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ProgressBar;
import android.widget.RadioGroup;
import android.widget.TableLayout;
import android.widget.TableRow;
import android.widget.TextView;

import org.w3c.dom.Text;

import java.util.List;

/**
 * Created by rafael on 09/10/17.
 */

public class EndActivity extends AppCompatActivity
{
    @Override
    public void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_end);

        Bundle  bundle     = getIntent().getExtras();
        int[][] resultList = (int[][]) bundle.get("result_list");
        boolean is_preview = (boolean) bundle.get("preview");

        Button button = (Button) findViewById(R.id.startButton);

        if (!is_preview) {
            button.setVisibility(View.GONE);
        } else {
            button.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View view) {
                    Intent intent = new Intent(EndActivity.this, MOSActivity.class);
                    startActivity(intent);
                }
            });
        }

        if (resultList.length == 0) {
            Intent intent = new Intent(this, MainActivity.class);
            startActivity(intent);
        }

        TableLayout table = (TableLayout) findViewById(R.id.result_table);

        Log.d(EndActivity.class.toString(), table.toString());

        for (int i = 0; i < resultList.length; i++) {
            TableRow row    = new TableRow(this);
            TextView index  = new TextView(this);
            TextView female = new TextView(this);
            TextView male   = new TextView(this);

            index.setText(new Integer(i + 1).toString());
            female.setText(new Integer(resultList[i][0]).toString());
            male.setText(new Integer(resultList[i][1]).toString());

            female.setGravity(Gravity.RIGHT);
            male.setGravity(Gravity.RIGHT);

            TableRow.LayoutParams params = new TableRow.LayoutParams(
                    TableRow.LayoutParams.MATCH_PARENT,
                    TableRow.LayoutParams.MATCH_PARENT
            );
            params.setMargins(20, 0, 0, 0);

            female.setLayoutParams(params);
            male.setLayoutParams(params);

            row.addView(index);
            row.addView(female);
            row.addView(male);

            table.addView(row);
        }
    }
}
