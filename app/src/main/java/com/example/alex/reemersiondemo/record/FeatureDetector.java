package com.example.alex.reemersiondemo.record;

import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Scalar;
import org.opencv.features2d.FastFeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.imgproc.Imgproc;
import org.opencv.xfeatures2d.SURF;

import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/**
 * Created by alex on 1/27/18.
 */

public class FeatureDetector {
    private static final int                        kMaxFeatures = 200;

    private FastFeatureDetector     FAST;
    private SURF                    surf;

    private static final FeatureDetector ourInstance = new FeatureDetector();

    public static FeatureDetector getInstance() {
        return ourInstance;
    }

    private FeatureDetector() {
        FAST = FastFeatureDetector.create();
        surf = SURF.create();
        surf.setHessianThreshold(400);
    }

    public boolean extractFeatures(Mat inputFrame, Mat gray, MatOfKeyPoint keyPoints, Mat descriptors) {
//        FAST.detect(gray, keyPoints);
//        //too many features cause poor performance on mobile
//        if (keyPoints.total() > kMaxFeatures) {
//            List<KeyPoint> listOfKeyPoints = keyPoints.toList();
//            Collections.sort(listOfKeyPoints, new Comparator<KeyPoint>() {
//                @Override
//                public int compare(KeyPoint o1, KeyPoint o2) {
//                    return (int) (o2.response - o1.response);
//                }
//            });
//            keyPoints.fromList(listOfKeyPoints.subList(0, kMaxFeatures));
//        }
        surf.detectAndCompute(gray, new Mat(), keyPoints, descriptors);
//        surf.compute(gray, keyPoints, descriptors);

        return true;
    }
}
