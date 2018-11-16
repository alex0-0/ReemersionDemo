package com.example.alex.reemersiondemo.imgmatching;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Matrix;
import android.graphics.RectF;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.GridView;

import com.example.alex.reemersiondemo.R;

import edu.umb.cs.imageprocessinglib.ObjectDetector;
import edu.umb.cs.imageprocessinglib.feature.FeatureDetector;
import edu.umb.cs.imageprocessinglib.ImageProcessor;
import edu.umb.cs.imageprocessinglib.model.BoxPosition;
import edu.umb.cs.imageprocessinglib.model.Recognition;
//import edu.umb.cs.imageprocessinglib.feature.FeatureDetector;
//import com.example.alex.reemersiondemo.record.TensorFlowMultiBoxDetector;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class PicMatching extends Activity {
    private static String TAG = "PicMatching";
    private static final String                     MODEL_PATH = "file:///android_asset/ssd_mobilenet_v1_android_export.pb";
    private ArrayList<Mat> imageList;
    private GridView gridView;
//    private TensorFlowMultiBoxDetector tfDetector;

    private ObjectDetector detector;
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
//        TensorFlowInferenceInterface tensorflow = new TensorFlowInferenceInterface(getAssets(), MODEL_PATH);
//        tfDetector = TensorFlowMultiBoxDetector.getInstance();
//        tfDetector.setTensorflow(tensorflow);
        gridView.setOnItemClickListener(new AdapterView.OnItemClickListener() {
            public void onItemClick(AdapterView parent, View v, int position, long id) {
                if (position == 0 || !v.isShown())
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
                Mat test = Utils.loadResource(getApplicationContext(), R.drawable.test_10, Imgcodecs.CV_LOAD_IMAGE_COLOR);
                imageList.add(test);
                template = extractTemplate(test);
                imageList.add(template);
                //uncomment next line to see more clearly how distortion works
//                imageList.addAll(FeatureDetector.getInstance().distortImage(Utils.loadResource(getApplicationContext(), R.drawable.j, Imgcodecs.CV_LOAD_IMAGE_COLOR)));
                imageList.addAll(FeatureDetector.getInstance().distortImage(template));
                imageList.add(Utils.loadResource(getApplicationContext(), R.drawable.a, Imgcodecs.CV_LOAD_IMAGE_COLOR));
                imageList.add(Utils.loadResource(getApplicationContext(), R.drawable.b, Imgcodecs.CV_LOAD_IMAGE_COLOR));
                imageList.add(Utils.loadResource(getApplicationContext(), R.drawable.c, Imgcodecs.CV_LOAD_IMAGE_COLOR));
                imageList.add(Utils.loadResource(getApplicationContext(), R.drawable.d, Imgcodecs.CV_LOAD_IMAGE_COLOR));
                imageList.add(Utils.loadResource(getApplicationContext(), R.drawable.e, Imgcodecs.CV_LOAD_IMAGE_COLOR));
                imageList.add(Utils.loadResource(getApplicationContext(), R.drawable.f, Imgcodecs.CV_LOAD_IMAGE_COLOR));
                imageList.add(Utils.loadResource(getApplicationContext(), R.drawable.g, Imgcodecs.CV_LOAD_IMAGE_COLOR));
                imageList.add(Utils.loadResource(getApplicationContext(), R.drawable.h, Imgcodecs.CV_LOAD_IMAGE_COLOR));
                imageList.add(Utils.loadResource(getApplicationContext(), R.drawable.i, Imgcodecs.CV_LOAD_IMAGE_COLOR));
//                imageList.add(Utils.loadResource(getApplicationContext(), R.drawable.j, Imgcodecs.CV_LOAD_IMAGE_COLOR));
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
        if (detector == null) {
            detector = new ObjectDetector();
            detector.init(this);
        }
        List<Recognition> recognitions = detector.recognizeImage(image);
        ArrayList<Rect> boundRect = new ArrayList<>();
        for (Recognition r : recognitions) {
            BoxPosition rect = r.getScaledLocation(image.cols()/(float)r.getModelSize(), image.rows()/(float)r.getModelSize());
            Rect rec = new Rect(rect.getLeftInt(), rect.getTopInt(), rect.getWidthInt(), rect.getHeightInt());
            boundRect.add(rec);
        }

        //assume image only contains template
        if (boundRect.size() > 0) {
            return new Mat(image, boundRect.get(0));
//            BoxPosition bp = recognitions.get(0).getLocation();
//            Matrix matrix = new Matrix();
//            Matrix invertMatrix = new Matrix();
//            matrix.postScale(detector.cropSize/(float)image.cols(), detector.cropSize/(float)image.rows());
//            boolean a = matrix.invert(invertMatrix);
////            matrix.postScale(image.cols()/detector.cropSize, image.rows()/detector.cropSize);
//            RectF rectF = new RectF(bp.getLeft(), bp.getTop(), bp.getRight(), bp.getBottom());
//            invertMatrix.mapRect(rectF);
////            RectF rectF = recognitions.get(0).rectF;
//            Rect rect = new Rect((int)rectF.left, (int)rectF.top, (int)rectF.width(), (int)rectF.height());
//            return new Mat(image, rect);
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
