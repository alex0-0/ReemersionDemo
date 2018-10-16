package com.example.alex.reemersiondemo.record;

import android.os.AsyncTask;

//import com.example.imageprocessinglib.ImageFeature.FeatureDetector;
import com.example.imageprocessinglib.ImageProcessor;
import com.example.imageprocessinglib.Recognition;

import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Rect;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by alex on 3/22/18.
 */

public class FeatureDetectTask extends AsyncTask {
//    private TensorFlowMultiBoxDetector tfDetector;
//    private TensorFlowYoloDetector yoloDetector;
    private Runnable r;
    private ArrayList<Rect> boundRects;
    private MatOfKeyPoint objectKeypoints;
    private Mat descriptors;

    private ArrayList<Recognition> recognitions;

    public FeatureDetectTask() {
        super();
//        tfDetector = TensorFlowMultiBoxDetector.getInstance();
//        yoloDetector = TensorFlowYoloDetector.getInstance();
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
        ImageProcessor imageProcessor = (ImageProcessor) params[2];
//        FeatureDetector detector = (FeatureDetector) params[2];
        r = (Runnable)params[3];
//        ArrayList<Rect> boundRects = tfDetector.recognizeImage(rgba);
//        recognitions = tfDetector.getRecognitions();
        List<Recognition> recognitions = imageProcessor.recognizeImage(rgba);
//        ImageFeature imageFeature = imageProcessor.extractFeatures(gray);
//        objectKeypoints = imageFeature.getObjectKeypoints();
//        descriptors = imageFeature.getDescriptors();
//        ArrayList<Rect> boundRects = yoloDetector.recognizeImage(rgba);
//        recognitions = yoloDetector.getRecognitions();
//        objectKeypoints = new MatOfKeyPoint();
//        descriptors = new Mat();
//        detector.extractFeatures(gray, objectKeypoints, descriptors);
//        this.boundRects = boundRects;
        ArrayList<Rect> rects = new ArrayList<>();
        for (Recognition r : recognitions) {
            android.graphics.Rect rect = new android.graphics.Rect();
            r.getOriginalLoc().round(rect);
            Rect rec = new Rect(rect.left, rect.top, rect.width(), rect.height());
            rects.add(rec);
        }
        boundRects = rects;
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
