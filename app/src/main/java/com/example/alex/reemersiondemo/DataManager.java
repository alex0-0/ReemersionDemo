package com.example.alex.reemersiondemo;

import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;

/**
 * Created by alex on 1/29/18.
 */

public class DataManager {
    private Mat refTemplateImg;
    private Mat refDescriptors;
    private MatOfKeyPoint refKeyPoints;
    private Mat targetTemplateImg;
    private Mat targetDescriptors;
    private MatOfKeyPoint targetKeyPoints;

    private float azimuth = 0;
    private float roll = 0;
    private float pitch = 0;

    private static final DataManager ourInstance = new DataManager();

    public static DataManager getInstance() {
        return ourInstance;
    }

    private DataManager() {
    }

    public void storeRefData(Mat templateImg, MatOfKeyPoint keyPoints, Mat descriptors) {
        this.refTemplateImg = templateImg.clone();
        this.refKeyPoints = new MatOfKeyPoint(keyPoints.clone());
        this.refDescriptors = descriptors.clone();
    }

    public void storeTargetData(Mat templateImg, MatOfKeyPoint keyPoints, Mat descriptors) {
        this.targetTemplateImg = templateImg.clone();
        this.targetKeyPoints = new MatOfKeyPoint(keyPoints.clone());
        this.targetDescriptors = descriptors.clone();
    }

    public void storeRelativeAngle(float azimuth, float roll, float pitch) {
        this.azimuth = azimuth;
        this.roll = roll;
        this.pitch = pitch;
    }

    public Mat getRefTemplateImg() {
        return refTemplateImg;
    }

    public Mat getRefDescriptors() {
        return refDescriptors;
    }

    public MatOfKeyPoint getRefKeyPoints() {
        return refKeyPoints;
    }

    public Mat getTargetTemplateImg() {
        return targetTemplateImg;
    }

    public Mat getTargetDescriptors() {
        return targetDescriptors;
    }

    public MatOfKeyPoint getTargetKeyPoints() {
        return targetKeyPoints;
    }

    public float getAzimuth() {
        return azimuth;
    }

    public float getRoll() {
        return roll;
    }

    public float getPitch() {
        return pitch;
    }

}
