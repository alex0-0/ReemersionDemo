package com.example.alex.reemersiondemo.record;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.view.WindowManager;
import android.widget.Toast;

import com.example.alex.reemersiondemo.DataManager;
import com.example.alex.reemersiondemo.OrientationManager;
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
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;

public class RecordController extends Activity implements CameraBridgeViewBase.CvCameraViewListener2, OrientationManager.Listener {

    private static final int                        kMaxFeatures = 500;
    //selected area is useful only when features inside is more than kMinFeatures
    private static final int                        kMinFeatures = 30;
    private static final int                        kMinRectLength = 80;
    private static final String                     TAG = "RecordController";
    private static final String                     MODEL_PATH = "file:///android_asset/ssd_mobilenet_v1_android_export.pb";

    private TensorFlowMultiBoxDetector              tfDetector;
    private CameraBridgeViewBase                    mOpenCvCameraView;
    private Mat                                     mRgba;
    private Mat                                     mGray;
    private ArrayList<Rect>                         boundRects;         //the rectangle on objects
    private ArrayList<ArrayList<KeyPoint>>          featureList;        //features inside one rectangle, corresponding to boundRects
    private MatOfKeyPoint                           objectKeypoints;
    private Mat                                     descriptors;
    private int                                     selectedIndex;      //record the rectangle user selected
    private Mat                                     ROI;
    private MatOfKeyPoint                           ROIKeypoints;
    private Mat                                     ROIDescriptors;
    private Mat                                     tmpROIGray;
    private FeatureDetector detector;
    private DataManager                             dataManager;
    private OrientationManager                      orientationManager;
    private float                                   initialAzimuth = 0;
    private float                                   initialRoll = 0;
    private float                                   initialPitch = 0;
    private boolean                                 refRecorded = false;        //whether reference object recorded

    private BaseLoaderCallback  mLoaderCallback = new BaseLoaderCallback(this) {
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
        detector = FeatureDetector.getInstance();
        objectKeypoints = new MatOfKeyPoint();
        descriptors = new Mat();
        featureList = new ArrayList<>();
        ROIKeypoints = new MatOfKeyPoint();
        ROIDescriptors = new Mat();
        tmpROIGray = new Mat();
        selectedIndex = -1;
        dataManager = DataManager.getInstance();
        boundRects = new ArrayList<>();
        TensorFlowInferenceInterface tensorflow = new TensorFlowInferenceInterface(getAssets(), MODEL_PATH);
        tfDetector = TensorFlowMultiBoxDetector.getInstance();
        tfDetector.setTensorflow(tensorflow);
        orientationManager = new OrientationManager(this);
        orientationManager.startListening(this);
        mOpenCvCameraView.setOnTouchListener(new View.OnTouchListener() {
            @Override
            public boolean onTouch(View v, MotionEvent event) {
                handleTouch(event);
                return false;
            }
        });
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
        orientationManager.stopListening();
        mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        mGray = new Mat();
        mRgba = new Mat();
    }

    public void onCameraViewStopped() {
        mGray.release();
        mRgba.release();
        orientationManager.stopListening();
    }

    public synchronized Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();
        boundRects = tfDetector.recognizeImage(mRgba);
//        boundRects = findRectangleOnObjects(mGray);
        detector.getFeatures(mRgba, mGray, objectKeypoints, descriptors);
        constructFeatureMap();

        Scalar color = new Scalar(0,255,0);
        //draw rectangle
        for( int i = 0; i < boundRects.size(); i++ ) {
            //            drawContours( frame, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, cv::Point() );
            if (i < boundRects.size() && boundRects.get(i).height > 0) {
                Imgproc.rectangle( mRgba, boundRects.get(i).tl(), boundRects.get(i).br(), color, 3);
            }
        }
        return mRgba;
    }

    //Now I am using tensorflow to find rectrangle on objects //2018.02.19
    private ArrayList<Rect> findRectangleOnObjects(Mat frame) {

        ArrayList<Rect> boundRects = new ArrayList<>();
        Mat current = new Mat(mGray.size(),0);

        Imgproc.GaussianBlur(mGray, current, new Size(3,3),1);  //TODO: sigmaX is not sure, should test more
        Imgproc.resize(current, current, new Size(106, 80), 0, 0, Imgproc.INTER_CUBIC);
        Imgproc.resize(current, current, new Size(mGray.width(), mGray.height()));

        Mat thresholdOutput = new Mat();
        ArrayList<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        int thresh = 100;
        Imgproc.threshold(current, thresholdOutput, thresh, 255, Imgproc.THRESH_BINARY);
        Imgproc.findContours(thresholdOutput, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE, new Point(0,0));

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

    private boolean handleTouch(MotionEvent event) {
        int touchAction = event.getActionMasked();
        double xLocation=-1, yLocation=-1;
        double xOffset = (mOpenCvCameraView.getWidth() - mRgba.cols()) / 2;
        double yOffset = (mOpenCvCameraView.getHeight() - mRgba.rows()) / 2;
        switch (touchAction) {
            case MotionEvent.ACTION_DOWN:
//                android.graphics.Point size = new android.graphics.Point();
//                getWindowManager().getDefaultDisplay().getSize(size);
                xLocation =  event.getX() - xOffset;
                yLocation =  event.getY() - yOffset;
                break;
        }

        Point p=new Point(xLocation,yLocation);
        Log.i(TAG, "touched position:\t" + xLocation + "\t" + yLocation);

        for(int i=0; i < boundRects.size(); i++){
            Rect rect = boundRects.get(i);
            if(rect.contains(p)){
                if (featureList.size() > i && featureList.get(i).size() >= kMinFeatures) {
                    selectedIndex = i;
                    ROI = new Mat(mRgba, boundRects.get(i).clone());
                    Imgproc.cvtColor(ROI, tmpROIGray, Imgproc.COLOR_BGRA2GRAY);
                    detector.getFeatures(ROI, tmpROIGray, ROIKeypoints, ROIDescriptors);

                    //if this is target object
                    //@TODO: target object may not need to get key points?
                    if (refRecorded) {
                        dataManager.storeTargetData(tmpROIGray, ROIKeypoints, ROIDescriptors);
                        //store the angle of target relative to reference object
                        dataManager.storeRelativeAngle(
                                orientationManager.getAzimuth() - initialAzimuth,
                                orientationManager.getRoll() - initialRoll,
                                orientationManager.getPitch() - initialPitch
                        );
                        orientationManager.stopListening();
                        Toast.makeText(this, "Target Recorded!"
                                + "\n" + dataManager.getAzimuth()
                                + "\n" + dataManager.getPitch()
                                + "\n" + dataManager.getRoll(),
                                Toast.LENGTH_SHORT).show();
                        finish();
                        return true;
                    }

                    dataManager.storeRefData(tmpROIGray, ROIKeypoints, ROIDescriptors);
                    initialAzimuth = orientationManager.getAzimuth();
                    initialRoll = orientationManager.getRoll();
                    initialPitch = orientationManager.getPitch();
                    refRecorded = true;

                    Toast.makeText(this, "Rect:" + rect.x + "\t"
                                    + rect.y + "\t"
                                    + rect.width + "\t"
                                    + rect.height + "\t"
                                    + "features:\t"
                                    + featureList.get(i).size() +"\nPlease take shot of target object now",
                            Toast.LENGTH_SHORT).show();

                    return true;
                }
            }
        }
        return false;
    }

    @Override
    public void onOrientationChanged(float azimuth, float pitch, float roll) {
//        System.out.println("pitch:\t" + pitch + "\troll:\t" + roll);

        Log.e(TAG,"azimuth:\t" + azimuth + "\tpitch:\t" + pitch + "\troll:\t" + roll);
    }
}
