package com.example.alex.reemersiondemo.imgmatching;

import android.app.Activity;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageView;
import android.widget.TextView;

import com.example.alex.reemersiondemo.R;
//import com.example.imageprocessinglib.ImageFeature.FeatureDetector;
//import com.example.imageprocessinglib.ImageFeature.FeatureMatcher;
import edu.umb.cs.imageprocessinglib.ImageProcessor;
import edu.umb.cs.imageprocessinglib.feature.FeatureDetector;
import edu.umb.cs.imageprocessinglib.model.Recognition;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.imgproc.Imgproc;

import java.util.List;

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
//        Mat gray = new Mat();
//        Imgproc.cvtColor(templateImg, gray, Imgproc.COLOR_BGRA2GRAY);
////        FeatureDetector.getInstance().extractDistinctFeatures(gray, tKPs, tDescriptors);
//        ImageFeature imageFeatureObject = ImageProcessor.extractDistinctFeatures(gray);
//        gray.release();
//        extractFeatures(queryImg, qKPs, qDescriptors);

        //test TensorFlow
//        ImageProcessor imageProcessor = new ImageProcessor();
//        Bitmap bitmap=Bitmap.createBitmap(templateImg.cols(),  templateImg.rows(), Bitmap.Config.ARGB_8888);
//        Utils.matToBitmap(templateImg, bitmap);
//        ImageProcessorConfig config = new ImageProcessorConfig(bitmap.getWidth(), bitmap.getHeight(), this, 0);
//        imageProcessor.initObjectDetector(config);
////        for (int k=0; k<100; k++) {
//            List<Recognition> recognitionList = imageProcessor.recognizeImage(bitmap);
//            int a = 0;
//        }
        //test the influence of feature points number on matching time
//        for (int i = 100; i <= 500; i += 50) {
//            ImageFeature feature = ImageProcessor.extractORBFeatures(templateImg, i);
//            String log1 = String.format("Comparing %d vs %d FPs ", feature.getSize(),feature.getSize());
//            Log.d("MATCH TEST", log1);
//            long before=System.currentTimeMillis();
//            for (int k=0; k<100; k++) {
//                MatOfDMatch ms = ImageProcessor.matcheImages(feature, feature);
//            }
//            long after=System.currentTimeMillis();
//            String log = String.format("takes %.03f s\n", ((float)(after-before)/1000/100));
//            Log.d("MATCH TEST", log);
//        }

        //get matches
//        MatOfDMatch matches = FeatureMatcher.getInstance().matchFeature(queryImg, qDescriptors, tDescriptors, qKPs, tKPs);
        Mat displayImage = new Mat();
//        Features2d.drawMatches(queryImg, qKPs, templateImg, tKPs, matches, displayImage);

//        Log.d(TAG, "Debug Match Points:");
//        DMatch m[] = matches.toArray();
//        KeyPoint q[] = qKPs.toArray();
//        KeyPoint t[] = tKPs.toArray();
//        for (int i = 0; i < matches.total(); i++) {
//            Log.d(TAG, "Matched Points Info:" +
//                    "query point: " + q[m[i].queryIdx].pt.toString()
//            + "template point: " + t[m[i].trainIdx].pt.toString());
//        }

//        //calculate precision
//        long tKPSize = tKPs.total();
//        long qKPSize = qKPs.total();
//        long matchSize = matches.total();
//        float precision = (float)matchSize/tKPSize;
//        double bonusConfidence = FeatureMatcher.getInstance().bonusConfidenceFromClusteringMatchedPoints(matches, qKPs, tKPs);
//        String text = "Template feature points number: " + tKPSize +
//                "\n Query feature points number: " + qKPSize +
//                "\n Matched feature points number: " + matchSize +
//                "\n Precision(matched feature number/template feature number): " + precision +
//                "\n Bonus confidence: " + bonusConfidence +
//                "\n Sum score(bonus + precision): " + (bonusConfidence + precision);
//        textView.setText(text);
//
//        tKPs.release();
//        tDescriptors.release();
//        qKPs.release();
//        qDescriptors.release();
//        matches.release();

//        return displayImage;
        return queryImg;
    }

    private void extractFeatures(Mat img, MatOfKeyPoint keyPoints, Mat descriptors) {
        Mat gray = new Mat();
        Imgproc.cvtColor(img, gray, Imgproc.COLOR_BGRA2GRAY);
        FeatureDetector.getInstance().extractFeatures(gray, keyPoints, descriptors);
        gray.release();
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
    }
}
