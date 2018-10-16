package com.example.imageprocessinglib;

import android.content.Context;

public class ImageProcessorConfig {
    private int oriWidth;
    private int oriHeight;
    private Context context;
    private int rotation;       //image rotation

    public ImageProcessorConfig(int oriWidth, int oriHeight, Context context, int rotation) {
        this.oriWidth = oriWidth;
        this.oriHeight = oriHeight;
        this.context = context;
        this.rotation = rotation;
    }

    // Which detection model to use: by default uses Tensorflow Object Detection API frozen
    // checkpoints.  Optionally use legacy Multibox (trained using an older version of the API)
    // or YOLO.
    public enum DetectorMode {
        TF_OD_API, MULTIBOX, YOLO;
    }

    private static final DetectorMode MODE = DetectorMode.TF_OD_API;

    /*
    get original image width
     */
    public int getOriWidth() {
        return oriWidth;
    }

    /*
    get original image height
     */
    public int getOriHeight() {
        return oriHeight;
    }

    public static DetectorMode getMODE() {
        return MODE;
    }

    public int getRotation() {
        return rotation;
    }

    public Context getContext() {
        return context;
    }

}

