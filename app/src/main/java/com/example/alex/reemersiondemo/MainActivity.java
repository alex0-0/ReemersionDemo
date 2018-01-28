package com.example.alex.reemersiondemo;

import android.app.Activity;
import android.content.Intent;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.TextView;

import com.example.alex.reemersiondemo.record.RecordController;
import com.example.alex.reemersiondemo.reemerge.ReemergeController;

public class MainActivity extends Activity {

    private static String TAG = "MainActivity";

    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("native-lib");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Example of a call to a native method
        TextView tv = (TextView) findViewById(R.id.sample_text);
        tv.setText(stringFromJNI());
    }

    public void record(View view) {
        Log.i(TAG, "record activity starts");

        Intent intent = new Intent(this, RecordController.class);
        startActivity(intent);

    }

    public void reemerge(View view) {
        Log.i(TAG, "reemerge activity starts");

        Intent intent = new Intent(this, ReemergeController.class);
        startActivity(intent);
    }

    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */
    public native String stringFromJNI();
}
