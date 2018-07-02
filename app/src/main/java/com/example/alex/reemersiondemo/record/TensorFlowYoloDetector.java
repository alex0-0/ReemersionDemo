/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package com.example.alex.reemersiondemo.record;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.Trace;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;

import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

/** An object detector that uses TF and a YOLO model to detect objects. */
public class TensorFlowYoloDetector {

    private static final float MINIMUM_CONFIDENCE_YOLO = 0.25f;
    // Only return this many results with at least this confidence.
    private static final int MAX_RESULTS = 5;

    private static final int NUM_CLASSES = 20;

    private static final int NUM_BOXES_PER_BLOCK = 5;
//    private static final String YOLO_MODEL_FILE = "file:///android_asset/graph-tiny-yolo-voc.pb";     //move this line to RecordController
    private static final int YOLO_INPUT_SIZE = 416;
    private static final String YOLO_INPUT_NAME = "input";
    private static final String YOLO_OUTPUT_NAMES = "output";
    private static final int YOLO_BLOCK_SIZE = 32;
    private TensorFlowInferenceInterface tensorflow;

    private static final TensorFlowYoloDetector ourInstance = new TensorFlowYoloDetector();

    public static TensorFlowYoloDetector getInstance() {
        return ourInstance;
    }

    public void setTensorflow(TensorFlowInferenceInterface tensorflow) {
        this.tensorflow = tensorflow;
    }

    // TODO(andrewharp): allow loading anchors and classes
    // from files.
    private static final double[] ANCHORS = {
            1.08, 1.19,
            3.42, 4.41,
            6.63, 11.38,
            9.42, 5.11,
            16.62, 10.52
    };

    private static final String[] LABELS = {
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor"
    };

    private boolean logStats = false;

    private TensorFlowYoloDetector() {}

    private float expit(final float x) {
        return (float) (1. / (1. + Math.exp(-x)));
    }

    private void softmax(final float[] vals) {
        float max = Float.NEGATIVE_INFINITY;
        for (final float val : vals) {
            max = Math.max(max, val);
        }
        float sum = 0.0f;
        for (int i = 0; i < vals.length; ++i) {
            vals[i] = (float) Math.exp(vals[i] - max);
            sum += vals[i];
        }
        for (int i = 0; i < vals.length; ++i) {
            vals[i] = vals[i] / sum;
        }
    }

    public ArrayList<Rect> recognizeImage(final Mat bitmap) {
        // Log this method so that it can be analyzed with systrace.
        Trace.beginSection("recognizeImage");

        Trace.beginSection("preprocessBitmap");
        // Preprocess the image data from 0-255 int to normalized float based
        // on the provided parameters.
        Mat rgbMap = bitmap.clone();
        Imgproc.cvtColor(rgbMap, rgbMap, Imgproc.COLOR_RGBA2RGB);
        Imgproc.resize(rgbMap, rgbMap, new Size(YOLO_INPUT_SIZE, YOLO_INPUT_SIZE), 0, 0, Imgproc.INTER_CUBIC);

        //get every pixel value
        byte[] byteValues = new byte[(int)rgbMap.total()*rgbMap.channels()];
        rgbMap.get(0,0, byteValues);
//TODO: compare the value array to previous one, see if previous detect contains negtive value
        float[] floatValues = new float[byteValues.length];
        //convert byte value to float
        for (int i = 0; i < byteValues.length; i++)
            floatValues[i] = byteValues[i]/255.0f;

        Trace.endSection(); // preprocessBitmap

        // Copy the input data into TensorFlow.
        Trace.beginSection("feed");
        tensorflow.feed(YOLO_INPUT_NAME, floatValues, 1, rgbMap.width(), rgbMap.height(), 3);
        Trace.endSection();

        String[] outputNames = new String[]{YOLO_OUTPUT_NAMES};
        // Run the inference call.
        Trace.beginSection("run");
        tensorflow.run(outputNames, logStats);
        Trace.endSection();

        // Copy the output Tensor back into the output array.
        Trace.beginSection("fetch");
        final int gridWidth = rgbMap.width() / YOLO_BLOCK_SIZE;
        final int gridHeight = rgbMap.height() / YOLO_BLOCK_SIZE;
        final float[] output =
                new float[gridWidth * gridHeight * (NUM_CLASSES + 5) * NUM_BOXES_PER_BLOCK];
        tensorflow.fetch(outputNames[0], output);
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

        for (int y = 0; y < gridHeight; ++y) {
            for (int x = 0; x < gridWidth; ++x) {
                for (int b = 0; b < NUM_BOXES_PER_BLOCK; ++b) {
                    final int offset =
                            (gridWidth * (NUM_BOXES_PER_BLOCK * (NUM_CLASSES + 5))) * y
                                    + (NUM_BOXES_PER_BLOCK * (NUM_CLASSES + 5)) * x
                                    + (NUM_CLASSES + 5) * b;

                    final float xPos = (x + expit(output[offset + 0])) * YOLO_BLOCK_SIZE;
                    final float yPos = (y + expit(output[offset + 1])) * YOLO_BLOCK_SIZE;

                    final float w = (float) (Math.exp(output[offset + 2]) * ANCHORS[2 * b + 0]) * YOLO_BLOCK_SIZE;
                    final float h = (float) (Math.exp(output[offset + 3]) * ANCHORS[2 * b + 1]) * YOLO_BLOCK_SIZE;

                    final RectF rect =
                            new RectF(
                                    Math.max(0, xPos - w / 2),
                                    Math.max(0, yPos - h / 2),
                                    Math.min(rgbMap.width() - 1, xPos + w / 2),
                                    Math.min(rgbMap.height() - 1, yPos + h / 2));
                    final float confidence = expit(output[offset + 4]);

                    int detectedClass = -1;
                    float maxClass = 0;

                    final float[] classes = new float[NUM_CLASSES];
                    for (int c = 0; c < NUM_CLASSES; ++c) {
                        classes[c] = output[offset + 5 + c];
                    }
                    softmax(classes);

                    for (int c = 0; c < NUM_CLASSES; ++c) {
                        if (classes[c] > maxClass) {
                            detectedClass = c;
                            maxClass = classes[c];
                        }
                    }

                    final float confidenceInClass = maxClass * confidence;
                    if (confidenceInClass > 0.01) {
                        pq.add(new Recognition("" + offset, LABELS[detectedClass], confidenceInClass, rect));
                    }
                }
            }
        }

        final ArrayList<Recognition> recognitions = new ArrayList<Recognition>();
        final ArrayList<Rect> rects = new ArrayList<>();

        //calculate the ratio between the original image and resized image
        float widthRatio = (float)bitmap.width() / YOLO_INPUT_SIZE;
        float heightRatio = (float)bitmap.height() / YOLO_INPUT_SIZE;

        for (int i = 0; i < Math.min(pq.size(), MAX_RESULTS); ++i) {
            Recognition recognition = pq.poll();
            if (recognition.getConfidence() < MINIMUM_CONFIDENCE_YOLO)
                continue;
            recognitions.add(recognition);
            RectF rectF = recognition.getLocation();
            //store original object's position in original image
            rects.add(new Rect(
                            (int)(rectF.left * widthRatio),
                            (int)(rectF.top * heightRatio),
                            (int)((rectF.right - rectF.left) * widthRatio),
                            (int)((rectF.bottom - rectF.top) * heightRatio)
                    )
            );
        }
        Trace.endSection(); // "recognizeImage"

        return rects;
    }
}
