package com.example.alex.reemersiondemo.record;

import android.graphics.RectF;
import android.os.Trace;

import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.PriorityQueue;

/**
 * Created by alex on 2/17/18.
 */

public class TensorFlowMultiBoxDetector {

    // Only return this many results.
    private static final int MAX_RESULTS = 100;
    private static final String     INPUT_NAME = "image_tensor";
    private static final String     OUTPUT_NAME_LOCATION_NAME = "detection_boxes";
    private static final String     OUTPUT_NAME_SCORE_NAME = "detection_scores";
    private static final String     OUTPUT_NAME_CLASSES = "detection_classes";
    private static final String     OUTPUT_NAME_NUM_DETECTIONS = "num_detections";
    private static final int        INPUT_SIZE = 300;

    private TensorFlowInferenceInterface tensorflow;

    private static final TensorFlowMultiBoxDetector ourInstance = new TensorFlowMultiBoxDetector();

    public static TensorFlowMultiBoxDetector getInstance() {
        return ourInstance;
    }

    public void setTensorflow(TensorFlowInferenceInterface tensorflow) {
        this.tensorflow = tensorflow;
    }

    private TensorFlowMultiBoxDetector() {
    }

    public ArrayList<Rect> recognizeImage(final Mat bitmap) {
        // Log this method so that it can be analyzed with systrace.
        Trace.beginSection("recognizeImage");

        Trace.beginSection("preprocessBitmap");
        // Preprocess the image data from 0-255 int to normalized float based
        // on the provided parameters.
        Mat rgbMap = bitmap.clone();
        Imgproc.cvtColor(rgbMap, rgbMap, Imgproc.COLOR_RGBA2RGB);
        Imgproc.resize(rgbMap, rgbMap, new Size(300, 300), 0, 0, Imgproc.INTER_CUBIC);
        long t = rgbMap.total();
        long c = rgbMap.channels();
        byte[] byteValues = new byte[(int)rgbMap.total()*rgbMap.channels()];
        rgbMap.get(0,0,byteValues);
        Trace.endSection(); // preprocessBitmap

        // Copy the input data into TensorFlow.
        Trace.beginSection("feed");
        tensorflow.feed(INPUT_NAME, byteValues, 1, rgbMap.rows(), rgbMap.cols(), 3);
        Trace.endSection();
        String[] outputNames = new String[]{OUTPUT_NAME_LOCATION_NAME, OUTPUT_NAME_SCORE_NAME, OUTPUT_NAME_CLASSES, OUTPUT_NAME_NUM_DETECTIONS};

        // Run the inference call.
        Trace.beginSection("run");
        tensorflow.run(outputNames, false);
        Trace.endSection();

        // Copy the output Tensor back into the output array.
        Trace.beginSection("fetch");
        float[] outputLocations = new float[MAX_RESULTS * 4];
        float[] outputScores = new float[MAX_RESULTS];
        float[] outputClasses = new float[MAX_RESULTS];
        float[] outputNumDetections = new float[1];
        tensorflow.fetch(outputNames[0], outputLocations);
        tensorflow.fetch(outputNames[1], outputScores);
        tensorflow.fetch(outputNames[2], outputClasses);
        tensorflow.fetch(outputNames[3], outputNumDetections);
        Trace.endSection();


        // Find the best detections.
        final PriorityQueue<Recognition> pq =
                new PriorityQueue<Recognition>(
                        1,
                        new Comparator<Recognition>() {
                            @Override
                            public int compare(final Recognition lhs, final Recognition rhs) {
                                // Intentionally reversed to put high confidence at the head of the queue.
                                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                            }
                        });

        // Scale them back to the input size.
        for (int i = 0; i < outputScores.length; ++i) {
            final RectF detection =
                    new RectF(
                            outputLocations[4 * i] * INPUT_SIZE,
                            outputLocations[4 * i + 1] * INPUT_SIZE,
                            outputLocations[4 * i + 2] * INPUT_SIZE,
                            outputLocations[4 * i + 3] * INPUT_SIZE);
            pq.add(new Recognition("" + i, null, outputScores[i], detection));
        }

        final ArrayList<Recognition> recognitions = new ArrayList<Recognition>();
        final ArrayList<Rect> rects = new ArrayList<>();

        //calculate the ratio between the original image and resized image
        float widthRatio = (float)bitmap.height() / INPUT_SIZE;
        float heightRatio = (float)bitmap.width() / INPUT_SIZE;

        for (int i = 0; i < Math.min(pq.size(), MAX_RESULTS); ++i) {
            Recognition recognition = pq.poll();
            recognitions.add(recognition);
            RectF rectF = recognition.getLocation();
            //store original object's position in original image, the dimension is flipped over, i.e., width as height and height as width
            rects.add(new Rect(
                    (int)(rectF.top * heightRatio),
                    (int)(rectF.left * widthRatio),
                    (int)((rectF.bottom - rectF.top) * heightRatio),
                    (int)((rectF.right - rectF.left) * widthRatio)
                    )
            );
        }
        Trace.endSection(); // "recognizeImage"
        return rects;
    }

}
