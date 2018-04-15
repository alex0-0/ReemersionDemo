package com.example.alex.reemersiondemo.imgmatching;

import android.content.Context;
import android.graphics.Bitmap;
import android.support.annotation.NonNull;
import android.support.annotation.Nullable;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.ImageView;

import com.example.alex.reemersiondemo.R;

import org.opencv.android.Utils;
import org.opencv.core.Mat;

import java.util.ArrayList;

/**
 * Created by alex on 4/13/18.
 */

public class GridViewAdapter extends ArrayAdapter {
    private Context context;
    ViewHolder viewHolder;
    ArrayList<Bitmap> bitmapArrayList;

    public GridViewAdapter(Context context, ArrayList<Mat> imageList) {
        super(context, R.layout.adapter_image, imageList);
        this.bitmapArrayList = new ArrayList<>();
        for (Mat mat : imageList) {
            Bitmap bm = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(mat, bm);
            bitmapArrayList.add(bm);
        }
        this.context = context;
    }

    @Override
    public int getCount() {
        return bitmapArrayList.size();
    }

    @Override
    public long getItemId(int position) {
        return position;
    }

    @NonNull
    @Override
    public View getView(int position, @Nullable View convertView, @NonNull ViewGroup parent) {
        if (convertView == null) {
            viewHolder = new ViewHolder();
            convertView = LayoutInflater.from(getContext()).inflate(R.layout.adapter_image, parent, false);
            viewHolder.imageView = (ImageView)convertView.findViewById(R.id.list_image);
            convertView.setTag(viewHolder);
        }
        else {
            viewHolder = (ViewHolder)convertView.getTag();
        }
//        viewHolder.imageView.setVisibility(View.GONE);
        viewHolder.imageView.setImageBitmap(bitmapArrayList.get(position));
        return convertView;
    }

    private static class ViewHolder {
        ImageView imageView;
    }
}
