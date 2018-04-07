package com.example.alex.reemersiondemo.reemerge;

import android.app.Activity;

import com.example.alex.reemersiondemo.DataManager;
import com.example.alex.reemersiondemo.OrientationManager;

/**
 * Created by alex on 3/8/18.
 *
 * Tell user how to reemerge the situation
 */

public class UserGuider implements OrientationManager.Listener {
    private static final float  TOLERANCE = 30;
    private static final String TURNLEFT = "turn left";
    private static final String TURNRIGHT = "turn right";
    private static final String TURNUP = "turn upward";
    private static final String TURNDOWN = "turn downward";
    private static final String TURNCLOCKWISE = "turn phone clockwise";
    private static final String TURNCOUNTERCLOCKWISE = "turn phone counterclockwise";

    private String guidence = "";

    private float initialAzimuth = 0;
    private float initialRoll = 0;
    private float initialPitch = 0;
    private float targetAzimuth = 0;
    private float targetRoll = 0;
    private float targetPitch = 0;

    private OrientationManager orientationManager;

    //initialization, on default we assume the reference object is in image
    // and this method is called at the instant reference object is found
    public void startGuide(Activity activity) {
        orientationManager = new OrientationManager(activity);
        orientationManager.startListening(this);
        initialAzimuth = orientationManager.getAzimuth();
        initialPitch = orientationManager.getPitch();
        initialRoll = orientationManager.getRoll();
        targetAzimuth = DataManager.getInstance().getAzimuth();
        targetPitch = DataManager.getInstance().getPitch();
        targetRoll = DataManager.getInstance().getRoll();
    }

    public void stopGuide() {
        orientationManager.stopListening();
    }

    public String getGuidence() {
        return guidence;
    }

    private float calAngleDiff(float a, float b) {
       return (Math.abs(a - b) < 180)?Math.abs(a - b) : (360 - Math.abs(a - b));
    }

    private boolean qualifiedAzimuth(float azimuth) {
        return Math.abs(calAngleDiff(azimuth, initialAzimuth) - targetAzimuth) < TOLERANCE;
    }

    private boolean qualifiedRoll(float roll) {
       return Math.abs(calAngleDiff(roll, initialRoll) - targetRoll) < TOLERANCE;
    }

    private boolean qualifiedPitch(float pitch) {
        return Math.abs(calAngleDiff(pitch, initialPitch) - targetPitch) < TOLERANCE;
    }

    @Override
    public void onOrientationChanged(float azimuth, float pitch, float roll) {
        if (!qualifiedRoll(roll))
            guidence = (roll - initialRoll - targetRoll > 0)?TURNCLOCKWISE : TURNCOUNTERCLOCKWISE;
        else if (!qualifiedPitch(pitch))
            guidence = (pitch - initialPitch - targetPitch > 0)?TURNDOWN : TURNUP;
        else if (!qualifiedAzimuth(azimuth))
            guidence = (azimuth - initialAzimuth - targetAzimuth > 0)?TURNLEFT : TURNRIGHT;
        else
            guidence = "";
    }
}
