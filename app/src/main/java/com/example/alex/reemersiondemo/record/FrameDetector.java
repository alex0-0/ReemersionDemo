package com.example.alex.reemersiondemo.record;

import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Scalar;
import org.opencv.features2d.FastFeatureDetector;
import org.opencv.features2d.Feature2D;
import org.opencv.features2d.Features2d;
import org.opencv.imgproc.Imgproc;
import org.opencv.xfeatures2d.SURF;

import java.util.ArrayList;

/**
 * Created by alex on 1/27/18.
 */

public class FrameDetector {

    private FastFeatureDetector     FAST;
    private SURF                    surf;

    private static final FrameDetector ourInstance = new FrameDetector();

    public static FrameDetector getInstance() {
        return ourInstance;
    }

    private FrameDetector() {
        FAST = FastFeatureDetector.create();
        surf = SURF.create();
        surf.setHessianThreshold(400);
    }

    public boolean getFeatures(Mat inputFrame, Mat gray, MatOfKeyPoint keyPoints, Mat descriptors) {
        FAST.detect(gray, keyPoints);
//        surf.detectAndCompute(gray, new Mat(), keyPoints, descriptors);
        surf.compute(gray, keyPoints, descriptors);


        //    cv::KeyPointsFilter::retainBest(keypoints, kMaxFeatures);
//
//    if (keypoints.size() > kMaxFeatures)
//    {
//        std::sort(keypoints.begin(), keypoints.end(), [] (const cv::KeyPoint& kp1, const cv::KeyPoint& kp2) {
//            return kp1.response > kp2.response;
//        });
//        keypoints.resize(kMaxFeatures);
//    }

        Mat t = new Mat();
        Imgproc.cvtColor(inputFrame, t, Imgproc.COLOR_BGRA2BGR);
        Features2d.drawKeypoints(t, keyPoints, t, Scalar.all(-1), Features2d.DRAW_RICH_KEYPOINTS);

        Imgproc.cvtColor(t, inputFrame, Imgproc.COLOR_BGR2BGRA);

        return true;
    }
}
