package com.example.alex.reemersiondemo.reemerge;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.DMatch;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FlannBasedMatcher;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

/**
 * Created by alex on 1/28/18.
 */

public class FeatureMatcher {
//    static private final double kConfidence = 0.99;
//    static private final double kDistance = 3.0;
    static private final double kRatio = 0.7;

    private FlannBasedMatcher flannMatcher;

    private DescriptorMatcher BFMatcher;

    private static final FeatureMatcher ourInstance = new FeatureMatcher();

    public static FeatureMatcher getInstance() {
        return ourInstance;
    }

    private FeatureMatcher() {
        BFMatcher = DescriptorMatcher.create("BruteForce");
    }

    public MatOfDMatch matchFeature(Mat input, Mat queryDescriptor, Mat template, MatOfKeyPoint keypoints1, MatOfKeyPoint keypoints2) {
        ArrayList<MatOfDMatch> matches1 = new ArrayList<>();
        ArrayList<MatOfDMatch>  matches2 = new ArrayList<>();

        BFMatcher.knnMatch(queryDescriptor, template, matches1, 2);
        BFMatcher.knnMatch(template, queryDescriptor, matches2, 2);

        ratioTest(matches1);
        ratioTest(matches2);

        MatOfDMatch symMatches = symmetryTest(matches1, matches2);
        MatOfDMatch ransacMatches = new MatOfDMatch();

        if (symMatches.total() > 20) {
            ransacTest(symMatches, keypoints1, keypoints2, ransacMatches);
            return ransacMatches;
        }
        return symMatches;
    }



//if the two best matches are relatively close in distance,
//then there exists a possibility that we make an error if we select one or the other.
private int ratioTest(ArrayList<MatOfDMatch> matches) {
        ArrayList<MatOfDMatch> updatedMatches = new ArrayList<>();
        int removed=0;
        // for all matches
        for (int i = 0; i < matches.size(); i++) {
            MatOfDMatch matchIterator = matches.get(i);
            // if 2 NN has been identified
            if (matchIterator.total() > 1) {
                DMatch[] match = matchIterator.toArray();
                // check distance ratio
                if (match[0].distance/match[1].distance <= kRatio) {
                    updatedMatches.add(matchIterator);
                    continue;
                }
            }
            // does not have 2 neighbours or distance ratio is higher than threshold
            removed++;
        }
        matches = updatedMatches;

        return removed;
    }



    private MatOfDMatch symmetryTest(ArrayList<MatOfDMatch> matches1, ArrayList<MatOfDMatch> matches2) {

        ArrayList<DMatch> symMatchList = new ArrayList<>();

        for (int i = 0; i < matches1.size(); i++) {
            MatOfDMatch matchIterator1 = matches1.get(i);
            if (matchIterator1.total() < 2) {
                continue;
            }
            DMatch match1 = matchIterator1.toArray()[0];

            for (int d = 0; d < matches2.size(); d++) {
                MatOfDMatch matchIterator2 = matches2.get(d);
                if (matchIterator2.total() < 2)
                    continue;
                DMatch match2 = matchIterator2.toArray()[0];
                // Match symmetry test
                if (match1.queryIdx == match2.trainIdx && match2.queryIdx == match1.trainIdx) {
                    // add symmetrical match
                    symMatchList.add(match1);
                    break; // next match in image 1 -> image 2
                }
            }
        }

        MatOfDMatch symMatches = new MatOfDMatch();
        symMatches.fromList(symMatchList);
        return symMatches;
    }

    //refer to: https://en.wikipedia.org/wiki/Random_sample_consensus
// Identify good matches using RANSAC
    private void ransacTest(MatOfDMatch matches,
                           MatOfKeyPoint keypoints1,
                           MatOfKeyPoint keypoints2,
                           MatOfDMatch outMatches)
    {
//        return matches;
        // get keypoint coordinates of good matches to find homography and remove outliers using ransac
        List<Point> pts1 = new ArrayList<Point>();
        List<Point> pts2 = new ArrayList<Point>();
        LinkedList<DMatch> good_matches = new LinkedList<>(Arrays.asList(matches.toArray()));
        for(int i = 0; i<good_matches.size(); i++){
            pts1.add(keypoints1.toList().get(good_matches.get(i).queryIdx).pt);
            pts2.add(keypoints2.toList().get(good_matches.get(i).trainIdx).pt);
        }

        // convertion of data types - there is maybe a more beautiful way
        Mat outputMask = new Mat();
        MatOfPoint2f pts1Mat = new MatOfPoint2f();
        pts1Mat.fromList(pts1);
        MatOfPoint2f pts2Mat = new MatOfPoint2f();
        pts2Mat.fromList(pts2);

        // Find homography - here just used to perform match filtering with RANSAC, but could be used to e.g. stitch images
        // the smaller the allowed reprojection error (here 15), the more matches are filtered
        Mat Homog = Calib3d.findHomography(pts1Mat, pts2Mat, Calib3d.RANSAC, 15, outputMask, 2000, 0.995);

        // outputMask contains zeros and ones indicating which matches are filtered
        LinkedList<DMatch> better_matches = new LinkedList<DMatch>();
        for (int i = 0; i < good_matches.size(); i++) {
            if (outputMask.get(i, 0)[0] != 0.0) {
                better_matches.add(good_matches.get(i));
            }
        }
        outMatches.fromList(better_matches);
    }

}
