package com.example.alex.reemersiondemo.record;

import android.app.Activity;
import android.media.Image;
import android.os.Bundle;
import android.util.Log;
import android.view.WindowManager;

import com.example.alex.reemersiondemo.R;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;

public class RecordController extends Activity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final int                        kMaxFeatures = 500;
    //selected area is useful only when features inside is more than kMinFeatures
    private static final int                        kMinFeatures = 100;
    private static final int                        kMinRectLength = 80;

    private static final String                     TAG = "RecordController";
    private CameraBridgeViewBase                    mOpenCvCameraView;

    private Mat                                     mRgba;
    private Mat                                     mGray;
    private ArrayList<Rect>                         boundRects;     //the rectangle on objects
    private ArrayList<ArrayList<KeyPoint>>          featureList;    //features inside one rectangle, corresponding to boundRects
    private MatOfKeyPoint                           objectKeypoints;
    private Mat                                     descriptors;

    private FrameDetector                           detector;



    private BaseLoaderCallback  mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                    try {
                        initializeOpenCVDependencies();

                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
                break;

                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    private void initializeOpenCVDependencies() throws IOException {
        detector = FrameDetector.getInstance();
        objectKeypoints = new MatOfKeyPoint();
        descriptors = new Mat();
        featureList = new ArrayList<>();
    }


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_record_controller);


        mOpenCvCameraView = findViewById(R.id.record_activity_surface_view);
        mOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_3_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        super.onDestroy();
        mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
//        mGray = new Mat();
//        mRgba = new Mat();
    }

    public void onCameraViewStopped() {
//        mGray.release();
//        mRgba.release();
    }

    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();
        boundRects = findRectangleOnObjects(mGray);
        detector.getFeatures(mRgba, mGray, objectKeypoints, descriptors);
        constructFeatureMap();

        //draw rectangle
        for( int i = 0; i < boundRects.size(); i++ ) {
            Scalar color = new Scalar(0,255,0);
            //            drawContours( frame, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, cv::Point() );
            if (i < boundRects.size()) {
                Imgproc.rectangle( mRgba, boundRects.get(i).tl(), boundRects.get(i).br(), color, 2, 8, 0 );
            }
        }
        return mRgba;
    }

    private ArrayList<Rect> findRectangleOnObjects(Mat frame) {
        ArrayList<Rect> boundRects = new ArrayList<>();
        Mat current = new Mat(mGray.size(),0);

        Imgproc.GaussianBlur(mGray, current, new Size(3,3),1);  //TODO: sigmaX is not sure, should test more
        Imgproc.resize(current, current, new Size(106, 80), 0, 0, Imgproc.INTER_CUBIC);
        Imgproc.resize(current, current, new Size(640, 480));

        Mat thresoldOutput = new Mat();
        ArrayList<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        int thresh = 100;
        Imgproc.threshold(current, thresoldOutput, thresh, 255, Imgproc.THRESH_BINARY);
        Imgproc.findContours(thresoldOutput, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE, new Point(0,0));

        ArrayList<MatOfPoint2f> contoursPoly = new ArrayList<>(Collections.nCopies(contours.size(), new MatOfPoint2f()));

        for (int i = 0; i < contours.size(); i++) {
            Imgproc.approxPolyDP(new MatOfPoint2f(contours.get(i).toArray()), contoursPoly.get(i), 3, true);
            Rect tmpRect = Imgproc.boundingRect(contours.get(i));
            if (tmpRect.height > kMinRectLength && tmpRect.width > kMinRectLength) {
                boundRects.add(tmpRect);
            }
        }

        return boundRects;
    }

    private void constructFeatureMap() {
            if (featureList.size() > 0) {
                featureList.clear();
            }

            if (boundRects.size() <= 0 || objectKeypoints.empty()) {
                return;
            }

            KeyPoint[] tmpObjectKeypoints = objectKeypoints.toArray();
            for (int i = 0; i < boundRects.size(); i++) {

                Rect rect = boundRects.get(i);
                ArrayList<KeyPoint> keypoints = new ArrayList<>();
                for (int d = 0; d < objectKeypoints.total(); d++) {
                    KeyPoint p = tmpObjectKeypoints[d];
                    if (rect.contains(p.pt)) {
                        keypoints.add(p);
                    }
                }
                featureList.add(keypoints);
            }
    }
}
