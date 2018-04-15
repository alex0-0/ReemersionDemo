package com.example.alex.reemersiondemo.imgmatching;

import android.app.Activity;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.os.Parcel;
import android.os.Parcelable;
import android.util.Log;
import android.widget.ImageView;
import android.widget.TextView;

import com.example.alex.reemersiondemo.R;
import com.example.alex.reemersiondemo.record.FeatureDetector;
import com.example.alex.reemersiondemo.reemerge.FeatureMatcher;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.features2d.Features2d;
import org.opencv.imgproc.Imgproc;

public class MatchDisplay extends Activity {

    public static String TEMPLATE_KEY = "TEMPLATE_KEY";
    public static String QUERY_KEY = "QUERY_KEY";
    private static String TAG = "MatchDisplay";
    private Mat templateImg;
    private Mat queryImg;
    private TextView textView;

    //necessary operation after application load openCV library
    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");
                    initial();
                }
                break;

                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_match_display);
        textView = findViewById(R.id.textView);
        long templateAddr = getIntent().getLongExtra(TEMPLATE_KEY, 0);
        long queryAddr = getIntent().getLongExtra(QUERY_KEY, 0);
        templateImg = new Mat(templateAddr);
        queryImg = new Mat(queryAddr);
    }

    void initial() {
        ImageView imageView = findViewById(R.id.imageView);
        Mat mat = matchImage(templateImg, queryImg);
        Bitmap bm = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(mat, bm);
        imageView.setImageBitmap(bm);
    }

    private Mat matchImage(Mat templateImg, Mat queryImg) {
        MatOfKeyPoint tKPs = new MatOfKeyPoint();
        Mat tDescriptors = new Mat(); //template image's Descriptors
        MatOfKeyPoint qKPs = new MatOfKeyPoint();
        Mat qDescriptors = new Mat(); //query image's Descriptors

        //get feature points
        extractFeatures(templateImg, tKPs, tDescriptors);
        extractFeatures(queryImg, qKPs, qDescriptors);

        //get matches
        MatOfDMatch matches = FeatureMatcher.getInstance().matchFeature(queryImg, qDescriptors, tDescriptors, qKPs, tKPs);
        Mat displayImage = new Mat();
        Features2d.drawMatches(queryImg, qKPs, templateImg, tKPs, matches, displayImage);

        //calculate precision
        long tKPSize = tKPs.total();
        long qKPSize = qKPs.total();
        long matchSize = matches.total();
        float precision = (float)matchSize/tKPSize;
        String text = "Template feature points number: " + tKPSize +
                "\n Query feature points number: " + qKPSize +
                "\n Matched feature points number: " + matchSize +
                "\n Precision(matched feature number/template feature number): " + precision;
        textView.setText(text);


        return displayImage;
    }

    private void extractFeatures(Mat img, MatOfKeyPoint keyPoints, Mat descriptors) {
        Mat gray = new Mat();
        Imgproc.cvtColor(img, gray, Imgproc.COLOR_BGRA2GRAY);
        FeatureDetector.getInstance().extractFeatures(gray, keyPoints, descriptors);
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_3_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        templateImg = null;
        queryImg = null;
    }
}
