package com.example.alex.reemersiondemo.reemerge;

import android.os.AsyncTask;

import com.example.alex.reemersiondemo.record.FeatureDetector;

import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;

/**
 * Created by alex on 3/23/18.
 */

public class FeatureMatchTask extends AsyncTask {
    private Runnable r;
    private MatOfDMatch matches;
    private MatOfKeyPoint keypoints;

    //input current and reference image, match image in background thread
    //parms contains:
    //1. rgba Mat
    //2. gray Mat
    //3. feature detector
    //4. feature matcher
    //5. reference feature descriptors
    //6. reference feature keypoints
    //7. runnable callback
    @Override
    protected Object doInBackground(Object[] params) {
        Mat rgba = (Mat)params[0];
        Mat gray = (Mat)params[1];
        FeatureDetector detector = (FeatureDetector) params[2];
        FeatureMatcher matcher = (FeatureMatcher) params[3];
        Mat refDescriptors = (Mat) params[4];
        MatOfKeyPoint refKeyPoints = (MatOfKeyPoint) params[5];
        r = (Runnable)params[6];

        keypoints = new MatOfKeyPoint();
        Mat descriptors = new Mat();

        detector.getFeatures(rgba, gray, keypoints, descriptors);
        if (descriptors.elemSize() > 0) {
            matches = matcher.matchFeatureImage(gray, descriptors, refDescriptors, keypoints, refKeyPoints);
        }

        return null;
    }

    @Override
    protected void onPostExecute(Object o) {
        r.run();
    }

    public MatOfDMatch getMatches() {
        return matches;
    }

    public MatOfKeyPoint getKeypoints() {
        return keypoints;
    }
}
