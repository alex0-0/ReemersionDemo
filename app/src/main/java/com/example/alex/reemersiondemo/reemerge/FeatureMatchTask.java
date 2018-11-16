package com.example.alex.reemersiondemo.reemerge;

import android.os.AsyncTask;

import edu.umb.cs.imageprocessinglib.ImageProcessor;
import edu.umb.cs.imageprocessinglib.model.ImageFeature;

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
//        FeatureDetector detector = (FeatureDetector) params[2];
//        FeatureMatcher matcher = (FeatureMatcher) params[3];
        Mat refDescriptors = (Mat) params[2];
        MatOfKeyPoint refKeyPoints = (MatOfKeyPoint) params[3];
        r = (Runnable)params[4];

        ImageFeature refFeature = new ImageFeature(refKeyPoints, refDescriptors);
        ImageFeature queryFeature = ImageProcessor.extractFeatures(gray);

        keypoints = queryFeature.getObjectKeypoints();//new MatOfKeyPoint();
//        Mat descriptors = imageFeature.getDescriptors();//new Mat();

//        detector.extractFeatures(gray, keypoints, descriptors);
        if (keypoints.elemSize() > 0) {
//            matches = matcher.matchFeature(gray, descriptors, refDescriptors, keypoints, refKeyPoints);
            matches = ImageProcessor.matcheImages(queryFeature, refFeature);
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
