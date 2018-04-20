package com.example.alex.reemersiondemo.imgmatching;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.GridView;

import com.example.alex.reemersiondemo.R;
import com.example.alex.reemersiondemo.record.TensorFlowMultiBoxDetector;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.imgcodecs.Imgcodecs;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.IOException;
import java.util.ArrayList;

public class PicMatching extends Activity {
    private static String TAG = "PicMatching";
    private static final String                     MODEL_PATH = "file:///android_asset/ssd_mobilenet_v1_android_export.pb";
    private ArrayList<Mat> imageList;
    private GridView gridView;
    private TensorFlowMultiBoxDetector tfDetector;

    //necessary operation after application load openCV library
    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");
                    loadImages();
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
        setContentView(R.layout.activity_pic_matching);
        gridView = findViewById(R.id.pic_matching);
        TensorFlowInferenceInterface tensorflow = new TensorFlowInferenceInterface(getAssets(), MODEL_PATH);
        tfDetector = TensorFlowMultiBoxDetector.getInstance();
        tfDetector.setTensorflow(tensorflow);
        gridView.setOnItemClickListener(new AdapterView.OnItemClickListener() {
            public void onItemClick(AdapterView parent, View v, int position, long id) {
                if (position == 0)
                    return;
                match(position);
            }
        });

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

    //load image from drawable file, and set to adapter
    private void loadImages() {
        if (imageList == null) {
            imageList = new ArrayList<>();

            try {
                Mat template = Utils.loadResource(getApplicationContext(), R.drawable.template, Imgcodecs.CV_LOAD_IMAGE_COLOR);
                template = extractTemplate(template);
                imageList.add(template);
                imageList.add(Utils.loadResource(getApplicationContext(), R.drawable.a, Imgcodecs.CV_LOAD_IMAGE_COLOR));
                imageList.add(Utils.loadResource(getApplicationContext(), R.drawable.b, Imgcodecs.CV_LOAD_IMAGE_COLOR));
                imageList.add(Utils.loadResource(getApplicationContext(), R.drawable.c, Imgcodecs.CV_LOAD_IMAGE_COLOR));
                imageList.add(Utils.loadResource(getApplicationContext(), R.drawable.d, Imgcodecs.CV_LOAD_IMAGE_COLOR));
                imageList.add(Utils.loadResource(getApplicationContext(), R.drawable.e, Imgcodecs.CV_LOAD_IMAGE_COLOR));
                imageList.add(Utils.loadResource(getApplicationContext(), R.drawable.f, Imgcodecs.CV_LOAD_IMAGE_COLOR));
                imageList.add(Utils.loadResource(getApplicationContext(), R.drawable.g, Imgcodecs.CV_LOAD_IMAGE_COLOR));
                GridViewAdapter adapter = new GridViewAdapter(getApplicationContext(), imageList);
                gridView.setAdapter(adapter);
            } catch (IOException e) {
                Log.e(TAG, Log.getStackTraceString(e));
            }
        }
    }

    /**
     * Extract object from image by TensorFlow
     * @param image
     * @return objectImage. If unrecognizable, return original image.
     */
    private Mat extractTemplate(Mat image) {
        ArrayList<Rect> boundRect = tfDetector.recognizeImage(image);

        //assume image only contains template
        if (boundRect.size() > 0) {
            return new Mat(image, boundRect.get(0));
        }
        return image;
    }

    private void match(int pos) {
        if (imageList == null || imageList.size() <= 1)
            return;
        Intent intent = new Intent(this, MatchDisplay.class);
        long templateImgAddr = imageList.get(0).getNativeObjAddr();
        long queryImgAddr = imageList.get(pos).getNativeObjAddr();
        intent.putExtra(MatchDisplay.TEMPLATE_KEY, templateImgAddr);
        intent.putExtra(MatchDisplay.QUERY_KEY, queryImgAddr);
        startActivity(intent);
    }
}
