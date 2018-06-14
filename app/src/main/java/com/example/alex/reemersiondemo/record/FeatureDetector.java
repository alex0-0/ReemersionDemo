package com.example.alex.reemersiondemo.record;

import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.features2d.FastFeatureDetector;
import org.opencv.imgproc.Imgproc;
import org.opencv.xfeatures2d.SURF;

import java.util.ArrayList;

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

    public boolean extractFeatures(Mat gray, MatOfKeyPoint keyPoints, Mat descriptors) {
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

    public ArrayList<Mat> distortImage(Mat image) {
        ArrayList<Mat> r = new ArrayList<>();
        r.addAll(scaleImage(image));
        r.addAll(rotateImage(image));

        return r;
    }

    private static final float  stepScale = 0.1f;        //the difference between scales of generating distorted images
    private static final int    numOfScales = 6;    //the number of different scale distorted images

    /**
     * Scale original image to generate a group of distorted image
     * @param image     original image
     * @return          a list containing scaled images
     */
    private ArrayList<Mat> scaleImage(Mat image) {
        ArrayList<Mat> r = new ArrayList<>();
        Size size = image.size();
        double rows = size.height;
        double cols = size.width;
        for (int i = 1; i <= numOfScales/2; i++) {
            Mat largerImage = new Mat();
            Mat smallerImage = new Mat();
            Size newSize = new Size(rows * (1 + i * stepScale), cols * (1 + i * stepScale));
            Imgproc.resize(image, largerImage, newSize);
            newSize = new Size(rows * (1 - i * stepScale), cols * (1 - i * stepScale));
            Imgproc.resize(image, smallerImage, newSize);
            r.add(largerImage);
            r.add(smallerImage);
        }

        return r;
    }

    private static final float stepAngle = 10.0f;        //the step difference between angles of generating distorted images, in degree.
    private static final int    numOfRotations = 6;    //the number of different scale distorted images

    /**
     * Rotate original image to generate a group of distorted image
     * @param image     original image
     * @return          a list containing rotated images
     */
    private ArrayList<Mat> rotateImage(Mat image) {
        ArrayList<Mat> r = new ArrayList<>();
        Size size = image.size();//new Size(image.cols(), image.rows());
        Point center = new Point(size.width/2, size.height/2);
        for (int i = 1; i <= numOfRotations/2; i++) {
            Mat leftRotated = new Mat();
            Mat rightRotated = new Mat();

            //create transformation matrix
            Mat leftMatrix = Imgproc.getRotationMatrix2D(center, -stepAngle * i, 1);
            Mat rightMatrix = Imgproc.getRotationMatrix2D(center, stepAngle * i, 1);
            Imgproc.warpAffine(image, leftRotated, leftMatrix, size);
            Imgproc.warpAffine(image, rightRotated, rightMatrix, size);
            r.add(leftRotated);
            r.add(rightRotated);
        }

        return r;
    }

}
