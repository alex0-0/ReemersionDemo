package com.example.imageprocessinglib;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.RectF;

import com.example.imageprocessinglib.ImageFeature.FeatureDetector;
import com.example.imageprocessinglib.ImageFeature.FeatureMatcher;
import com.example.imageprocessinglib.ImageProcessorConfig.*;
import com.example.imageprocessinglib.tensorflow.Classifier;
import com.example.imageprocessinglib.tensorflow.TensorFlowMultiBoxDetector;
import com.example.imageprocessinglib.tensorflow.TensorFlowObjectDetectionAPIModel;
import com.example.imageprocessinglib.tensorflow.TensorFlowYoloDetector;
import com.example.imageprocessinglib.utils.ImageUtils;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;

import java.io.IOException;
import java.util.LinkedList;
import java.util.List;

public class ImageProcessor {
    static String TAG = "IMAGE_PROCESSOR";
    private int oriWidth;
    private int oriHeight;

    // Configuration values for the prepackaged multibox model.
    private static final int MB_INPUT_SIZE = 224;
    private static final int MB_IMAGE_MEAN = 128;
    private static final float MB_IMAGE_STD = 128;
    private static final String MB_INPUT_NAME = "ResizeBilinear";
    private static final String MB_OUTPUT_LOCATIONS_NAME = "output_locations/Reshape";
    private static final String MB_OUTPUT_SCORES_NAME = "output_scores/Reshape";
    private static final String MB_MODEL_FILE = "file:///android_asset/multibox_model.pb";
    private static final String MB_LOCATION_FILE = "file:///android_asset/multibox_location_priors.txt";

    private static final int TF_OD_API_INPUT_SIZE = 300;
    private static final String TF_OD_API_MODEL_FILE = "file:///android_asset/ssd_mobilenet_v1_android_export.pb";
    private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/coco_labels_list.txt";

    // Configuration values for tiny-yolo-voc. Note that the graph is not included with TensorFlow and
    // must be manually placed in the assets/ directory by the user.
    // Graphs and models downloaded from http://pjreddie.com/darknet/yolo/ may be converted e.g. via
    // DarkFlow (https://github.com/thtrieu/darkflow). Sample command:
    // ./flow --model cfg/tiny-yolo-voc.cfg --load bin/tiny-yolo-voc.weights --savepb --verbalise
    private static final String YOLO_MODEL_FILE = "file:///android_asset/graph-tiny-yolo-voc.pb";
    private static final int YOLO_INPUT_SIZE = 416;
    private static final String YOLO_INPUT_NAME = "input";
    private static final String YOLO_OUTPUT_NAMES = "output";
    private static final int YOLO_BLOCK_SIZE = 32;

    // Minimum detection confidence to track a detection.
    private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.6f;
    private static final float MINIMUM_CONFIDENCE_MULTIBOX = 0.1f;
    private static final float MINIMUM_CONFIDENCE_YOLO = 0.25f;

    private static final DetectorMode MODE = DetectorMode.TF_OD_API;
    private static final boolean MAINTAIN_ASPECT = MODE == DetectorMode.YOLO;
    private Matrix cropToFrameTransform;
    private Matrix frameToCropTransform;
    private Bitmap croppedBitmap;

    private Classifier detector;
    private float minConfidence = 0.5f;

    /*
    init TensorFlow
     */
    public void initObjectDetector(final ImageProcessorConfig config) {
        oriWidth = config.getOriWidth();
        oriHeight = config.getOriHeight();
        AssetManager assetManager = config.getContext().getAssets();

//        //if(tracker==null)
//        tracker = new MultiBoxTracker(this);

        int cropSize = TF_OD_API_INPUT_SIZE;
        if (MODE == DetectorMode.YOLO) {
            detector =
                    TensorFlowYoloDetector.create(
                            assetManager,
                            YOLO_MODEL_FILE,
                            YOLO_INPUT_SIZE,
                            YOLO_INPUT_NAME,
                            YOLO_OUTPUT_NAMES,
                            YOLO_BLOCK_SIZE);
            cropSize = YOLO_INPUT_SIZE;
            minConfidence = MINIMUM_CONFIDENCE_YOLO;
        } else if (MODE == DetectorMode.MULTIBOX) {
            detector =
                    TensorFlowMultiBoxDetector.create(
                            assetManager,
                            MB_MODEL_FILE,
                            MB_LOCATION_FILE,
                            MB_IMAGE_MEAN,
                            MB_IMAGE_STD,
                            MB_INPUT_NAME,
                            MB_OUTPUT_LOCATIONS_NAME,
                            MB_OUTPUT_SCORES_NAME);
            cropSize = MB_INPUT_SIZE;
            minConfidence = MINIMUM_CONFIDENCE_MULTIBOX;
        } else {
            try {
                detector = TensorFlowObjectDetectionAPIModel.create(
                        assetManager, TF_OD_API_MODEL_FILE, TF_OD_API_LABELS_FILE, TF_OD_API_INPUT_SIZE);
                cropSize = TF_OD_API_INPUT_SIZE;
                minConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
            } catch (final IOException e) {
                e.printStackTrace();
            }
        }

        croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Bitmap.Config.ARGB_8888);

        frameToCropTransform =
                ImageUtils.getTransformationMatrix(
                        oriWidth, oriHeight,
                        cropSize, cropSize,
                        config.getRotation(), MAINTAIN_ASPECT);
        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);
    }

    /*
    Process image recognition and save cropped object image
     */
    public List<Recognition> recognizeImage(Bitmap bitmap) {
        final Canvas canvas = new Canvas(croppedBitmap);
        canvas.drawBitmap(bitmap, frameToCropTransform, null);
        final List<Recognition> ret = new LinkedList<Recognition>();

        List<Recognition> recognitions = detector.recognizeImage(croppedBitmap);
        for (Recognition r : recognitions) {
            if (r.getConfidence() < minConfidence)
                continue;
            RectF location = r.getLocation();
            cropToFrameTransform.mapRect(location);
            r.setOriginalLoc(location);
            Bitmap cropped = Bitmap.createBitmap(bitmap, (int)location.left, (int)location.top, (int)location.width(), (int)location.height());
            r.setObjectImage(cropped);
            ret.add(r);
        }
        return ret;
    }

    /*
    Process image recognition and save cropped object image
     */
    public List<Recognition> recognizeImage(Mat img) {
        Bitmap tBM = Bitmap.createBitmap(oriWidth, oriHeight, Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(img, tBM);
        return recognizeImage(tBM);
    }

    /*
    Extract image feature points
     */
    static public ImageFeatureObject extractDistinctFeatures(Mat img) {
        MatOfKeyPoint kps = new MatOfKeyPoint();
        Mat des = new Mat();
        FeatureDetector.getInstance().extractDistinctFeatures(img, kps, des);
        return new ImageFeatureObject(kps, des);
    }

    /*
    Extract image feature points
     */
    static public ImageFeatureObject extractFeatures(Mat img) {
        MatOfKeyPoint kps = new MatOfKeyPoint();
        Mat des = new Mat();
        FeatureDetector.getInstance().extractFeatures(img, kps, des);
        return new ImageFeatureObject(kps, des);
    }

    static public ImageFeatureObject extractFeatures(Bitmap bitmap) {
        Mat img = new Mat();
        Utils.bitmapToMat(bitmap, img);
        return extractFeatures(img);
    }

    /*
    Match two images
     */
    static public MatOfDMatch matcheImages(ImageFeatureObject qIF, ImageFeatureObject tIF) {
        return FeatureMatcher.getInstance().matchFeature(qIF.getDescriptors(), tIF.getDescriptors(), qIF.getObjectKeypoints(), tIF.getObjectKeypoints());
    }

    static public MatOfDMatch matcheImages(Mat queryImg, Mat temImg) {
        MatOfKeyPoint kps = new MatOfKeyPoint();
        ImageFeatureObject qIF = extractFeatures(queryImg);
        ImageFeatureObject tIF = extractFeatures(temImg);
        return matcheImages(qIF, tIF);
    }

    static public MatOfDMatch matcheImages(Bitmap queryImg, Bitmap temImg) {
        ImageFeatureObject qIF = extractFeatures(queryImg);
        ImageFeatureObject tIF = extractFeatures(temImg);
        return matcheImages(qIF, tIF);
    }
}
