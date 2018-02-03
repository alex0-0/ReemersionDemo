package com.example.alex.reemersiondemo.reemerge;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.view.WindowManager;

import com.example.alex.reemersiondemo.DataManager;
import com.example.alex.reemersiondemo.R;
import com.example.alex.reemersiondemo.record.FrameDetector;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.features2d.Features2d;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;

public class ReemergeController extends Activity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "ReemergeController";
    
    private CameraBridgeViewBase                    mOpenCvCameraView;
    private Mat                                     mRgba;
    private Mat                                     mGray;
    private FrameDetector                           detector;
    private FrameMatcher                            matcher;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                    try {
                        initialize();

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

    private void initialize() throws IOException {
        mOpenCvCameraView.enableView();
        detector = FrameDetector.getInstance();
        matcher = FrameMatcher.getInstance();
    }


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_reemerge_controller);

        mOpenCvCameraView = findViewById(R.id.reemerge_activity_surface_view);
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
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
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
        mGray = new Mat();
        mRgba = new Mat();
    }

    public void onCameraViewStopped() {
        mGray.release();
        mRgba.release();
    }

    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        //if there is no record data
        if (DataManager.getInstance().getTemplateImg() == null)
            return inputFrame.rgba();

        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();
        MatOfKeyPoint keyPoints = new MatOfKeyPoint();
        Mat descriptors = new Mat();
        Mat tDescriptors = DataManager.getInstance().getDescriptors();          //template descriptors
        MatOfKeyPoint tKeyPoints = DataManager.getInstance().getKeyPoints();    //template keypoints
        Mat tGray = DataManager.getInstance().getTemplateImg();                 //template image
        MatOfDMatch goodMatches = new MatOfDMatch();
        Mat imgMatches = new Mat();

        detector.getFeatures(mRgba, mGray, keyPoints, descriptors);
        if (descriptors.elemSize() > 0) {
            goodMatches = matcher.matchFeatureImage(mGray, descriptors, tDescriptors, keyPoints, tKeyPoints);
            //TODO: MatOfByte may have problem, check it and reduce the number of keypoints
//            Features2d.drawMatches(mGray, keyPoints, tGray, tKeyPoints, goodMatches, mGray, Scalar.all(-1), Scalar.all(-1), new MatOfByte(), Features2d.NOT_DRAW_SINGLE_POINTS);
            Features2d.drawMatches(mGray, keyPoints, tGray, tKeyPoints, goodMatches, imgMatches);
        }

        float confidence = (float)goodMatches.total()/tKeyPoints.total();
        Log.i(TAG, "Confidence: \t" + confidence);
        String strConf = "matched keypoints: \t" + goodMatches.total();
        Imgproc.putText(imgMatches, strConf, new Point(200, 200), Core.FONT_HERSHEY_PLAIN, 1.0, new Scalar(0, 0, 255));

        return mGray;
    }
}
