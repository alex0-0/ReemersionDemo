package com.example.alex.reemersiondemo.record;

import android.os.AsyncTask;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Rect;

import java.util.ArrayList;

/**
 * Created by alex on 3/22/18.
 */

public class FeatureDetectTask extends AsyncTask {
    private TensorFlowMultiBoxDetector tfDetector;
    private TensorFlowYoloDetector yoloDetector;
    private Runnable r;
    private ArrayList<Rect> boundRects;
    private MatOfKeyPoint objectKeypoints;
    private Mat descriptors;

    private ArrayList<Recognition> recognitions;

    public FeatureDetectTask() {
        super();
        tfDetector = TensorFlowMultiBoxDetector.getInstance();
        yoloDetector = TensorFlowYoloDetector.getInstance();
    }

    //input rgb and gray image, process image in background thread
    //parms contains:
    //1. rgba Mat
    //2. gray Mat
    //3. feature detector
    //4. runnable callback
    @Override
    protected Object doInBackground(Object[] params) {
        Mat rgba = (Mat)params[0];
        Mat gray = (Mat)params[1];
        FeatureDetector detector = (FeatureDetector) params[2];
        r = (Runnable)params[3];
//        ArrayList<Rect> boundRects = tfDetector.recognizeImage(rgba);
//        recognitions = tfDetector.getRecognitions();
        ArrayList<Rect> boundRects = yoloDetector.recognizeImage(rgba);
        recognitions = yoloDetector.getRecognitions();
        objectKeypoints = new MatOfKeyPoint();
        descriptors = new Mat();
        detector.extractFeatures(gray, objectKeypoints, descriptors);
        this.boundRects = boundRects;
        return null;
    }

    @Override
    protected void onPostExecute(Object o) {
        r.run();
    }

    public ArrayList<Rect> getBoundRects() {
        return boundRects;
    }

    public MatOfKeyPoint getObjectKeypoints() {
        return objectKeypoints;
    }

    public Mat getDescriptors() {
        return descriptors;
    }

    public ArrayList<Recognition> getRecognitions() {
        return recognitions;
    }

}
