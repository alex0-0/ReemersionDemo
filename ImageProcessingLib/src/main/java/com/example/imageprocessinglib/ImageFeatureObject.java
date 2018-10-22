package com.example.imageprocessinglib;

import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;

public class ImageFeatureObject {
    private final MatOfKeyPoint objectKeypoints;
    private final Mat descriptors;

    public ImageFeatureObject(MatOfKeyPoint objectKeypoints, Mat descriptors) {
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
