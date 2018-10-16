package com.example.imageprocessinglib.tensorflow;


import android.graphics.Bitmap;

import com.example.imageprocessinglib.Recognition;

import java.util.List;

/**
 * Generic interface for interacting with different recognition engines.
 */
public interface Classifier {
    List<Recognition> recognizeImage(Bitmap bitmap);

    void enableStatLogging(final boolean debug);

    String getStatString();

    void close();
}
