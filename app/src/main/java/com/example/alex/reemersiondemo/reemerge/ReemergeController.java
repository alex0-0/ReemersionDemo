package com.example.alex.reemersiondemo.reemerge;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.view.WindowManager;
import android.widget.Toast;

import com.example.alex.reemersiondemo.DataManager;
import com.example.alex.reemersiondemo.R;
import com.example.alex.reemersiondemo.record.FeatureDetector;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.features2d.Features2d;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;

public class ReemergeController extends Activity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "ReemergeController";
    private static final float  CRITERION = 0.3f;      //judge whether images matched //with RANSAC, 0.30 should be enough, if without, 0.55.

    private CameraBridgeViewBase                    mOpenCvCameraView;
    private Mat                                     mRgba;
    private Mat                                     mGray;
    private FeatureDetector detector;
    private FeatureMatcher matcher;
    private Mat refDescriptors;
    private MatOfKeyPoint refKeyPoints;
    private Mat refGray;
    private Mat targetDescriptors;
    private MatOfKeyPoint targetKeyPoints;
    private Mat targetGray;
    private boolean isSeekingRef;
    private UserGuider userGuider;      //give guidance to user after finding reference object
    private volatile boolean onProcessing;       //judge if the computation of matching is running in background thread
    private MatOfDMatch matches;
    private MatOfKeyPoint keypoints;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                    mOpenCvCameraView.enableFpsMeter();
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
        detector = FeatureDetector.getInstance();
        matcher = FeatureMatcher.getInstance();
        refDescriptors = DataManager.getInstance().getRefDescriptors();
        refKeyPoints = DataManager.getInstance().getRefKeyPoints();
        refGray = DataManager.getInstance().getRefTemplateImg();
        targetDescriptors = DataManager.getInstance().getTargetDescriptors();
        targetKeyPoints = DataManager.getInstance().getTargetKeyPoints();
        targetGray = DataManager.getInstance().getTargetTemplateImg();
        isSeekingRef = true;
        keypoints = new MatOfKeyPoint();
        matches = new MatOfDMatch();
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
        if (mOpenCvCameraView != null) {
            mOpenCvCameraView.disableView();
            mOpenCvCameraView.disableFpsMeter();
        }
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
        mOpenCvCameraView.disableFpsMeter();
    }

    public void onCameraViewStarted(int width, int height) {
        mGray = new Mat();
        mRgba = new Mat();
    }

    public void onCameraViewStopped() {
        mGray.release();
        mRgba.release();
    }

    //default opencv camera callback, process frame
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        //if there is no record data
        if (DataManager.getInstance().getRefTemplateImg() == null)
            return inputFrame.rgba();

        if (!onProcessing) {
            //set flag indicating the computation of matching is on
            onProcessing = true;

            mRgba = inputFrame.rgba();
            mGray = inputFrame.gray();

            Mat refD = (isSeekingRef)? refDescriptors : targetDescriptors;
            MatOfKeyPoint refK = (isSeekingRef)? refKeyPoints : targetKeyPoints;

            //run relative computation in background thread
            final FeatureMatchTask fmTask = new FeatureMatchTask();
            fmTask.execute(mRgba, mGray, detector, matcher, refD, refK, new Runnable() {
                @Override
                public void run() {
                    matches = fmTask.getMatches();
                    keypoints = fmTask.getKeypoints();
                    onProcessing = false;
                }
            });
        }

        Mat imgMatches = new Mat();
        drawOnFrame(imgMatches);
        return imgMatches;
    }

    //draw matched features on frame
    private void drawOnFrame(Mat frame) {

        float confidence;
        if (isSeekingRef) {
            //if the reference object is not found
            Features2d.drawMatches(mGray, keypoints, refGray, refKeyPoints, matches, frame);
            confidence = (float)matches.total()/refKeyPoints.total();
            if (confidence > CRITERION)
                isSeekingRef = false;
        }
        else {
            //if user are finding target object
            if (userGuider == null) {
                //initiate user guider
                userGuider = new UserGuider();
                userGuider.startGuide(this);
            }
            Features2d.drawMatches(mGray, keypoints, targetGray, targetKeyPoints, matches, frame);
            confidence = (float)matches.total()/targetKeyPoints.total();
            String bs = (confidence > CRITERION)?"BINGO!!!" : userGuider.getGuidence();
            Imgproc.putText(frame, bs, new Point(60,60), Core.FONT_HERSHEY_PLAIN, 3.0, new Scalar(255, 0, 0));
            if (userGuider.getGuidence().length() == 0) {
                userGuider.stopGuide();
                this.runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        Toast.makeText(ReemergeController.this, "Target Founded! Angle Correct", Toast.LENGTH_SHORT).show();
                    }
                });
                finish();
            }
        }

        Log.i(TAG, "keypoints: \t" + keypoints.total());
        String strConf = "matched keypoints: \t" + matches.total() + " conf:\t" + confidence;
        Imgproc.putText(frame, strConf, new Point(20, 20), Core.FONT_HERSHEY_PLAIN, 2.0, new Scalar(255, 0, 0));
        Imgproc.resize(frame, frame, mGray.size());
    }
}
