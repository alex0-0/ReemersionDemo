package com.example.alex.reemersiondemo;

import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;

/**
 * Created by alex on 1/29/18.
 */

public class DataManager {
    private Mat templateImg;
    private Mat descriptors;
    private MatOfKeyPoint keyPoints;

    private static final DataManager ourInstance = new DataManager();

    public static DataManager getInstance() {
        return ourInstance;
    }

    private DataManager() {
    }

    public void storeNecessaryData(Mat templateImg, MatOfKeyPoint keyPoints, Mat descriptors) {
        this.templateImg = templateImg.clone();
        this.keyPoints = new MatOfKeyPoint(keyPoints.clone());
        this.descriptors = descriptors.clone();
    }

    public Mat getTemplateImg() {
        return templateImg;
    }

    public Mat getDescriptors() {
        return descriptors;
    }

    public MatOfKeyPoint getKeyPoints() {
        return keyPoints;
    }

}
