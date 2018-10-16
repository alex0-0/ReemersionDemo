package com.example.imageprocessinglib;

import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;

public class ImageFeature {
    private final MatOfKeyPoint objectKeypoints;
    private final Mat descriptors;

    public ImageFeature(MatOfKeyPoint objectKeypoints, Mat descriptors) {
        this.objectKeypoints = objectKeypoints;
        this.descriptors = descriptors;
    }

    public MatOfKeyPoint getObjectKeypoints() {
        return objectKeypoints;
    }

    public Mat getDescriptors() {
        return descriptors;
    }
}
