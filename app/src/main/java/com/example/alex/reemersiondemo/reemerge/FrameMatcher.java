package com.example.alex.reemersiondemo.reemerge;

import org.opencv.core.DMatch;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FlannBasedMatcher;
import java.util.ArrayList;

/**
 * Created by alex on 1/28/18.
 */

public class FrameMatcher {
    static private final double kConfidence = 0.99;
    static private final double kDistance = 3.0;
    static private final double kRatio = 0.7;

    private FlannBasedMatcher flannMatcher;

    private DescriptorMatcher BFMatcher;

    private static final FrameMatcher ourInstance = new FrameMatcher();

    public static FrameMatcher getInstance() {
        return ourInstance;
    }

    private FrameMatcher() {
        BFMatcher = DescriptorMatcher.create("BruteForce");
    }

    public ArrayList<DMatch> matchFeatureImage(Mat input, Mat queryDescriptor, Mat template, MatOfKeyPoint keypoints1, MatOfKeyPoint keypoints2) {
        ArrayList<MatOfDMatch> matches1 = new ArrayList<>();
        ArrayList<MatOfDMatch>  matches2 = new ArrayList<>();

        BFMatcher.knnMatch(queryDescriptor, template, matches1, 2);
        BFMatcher.knnMatch(template, queryDescriptor, matches2, 2);

        ratioTest(matches1);
        ratioTest(matches2);

        MatOfDMatch symMatches = symmetryTest(matches1, matches2);
        MatOfDMatch ransacMatches = new MatOfDMatch();

        if (symMatches.total() > 20) {
            Mat fundemental = ransacTest(symMatches, keypoints1, keypoints2, ransacMatches);
            return new ArrayList<>(ransacMatches.toList());
        }
        return new ArrayList<>(symMatches.toList());
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

            for (int d = 0; d < matches1.size(); d++) {
                MatOfDMatch matchIterator2 = matches2.get(d);
                if (matchIterator2.total() < 2)
                    continue;

                DMatch match1 = matchIterator1.toArray()[0];
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

// Identify good matches using RANSAC
// Return fundemental matrix
    private Mat ransacTest(MatOfDMatch matches,
                           MatOfKeyPoint keypoints1,
                           MatOfKeyPoint keypoints2,
                           MatOfDMatch outMatches)
    {
        return matches;
//        // Convert keypoints into Point2f
//        ArrayList<Point> points1 = new ArrayList<>();
//        ArrayList<Point> points2 = new ArrayList<>();
//        DMatch[] matchList = matches.toArray();
//
//        for (int i = 0; i < matches.total(); i++) {
//            DMatch it = matchList[i];
//            KeyPoint[] kp1 = keypoints1.toArray();
//            KeyPoint[] kp2 = keypoints2.toArray();
//        // Get the position of left keypoints
//        double x= kp1[it.queryIdx].pt.x;
//        double y= kp1[it.queryIdx].pt.y;
//        points1.add(new Point(x,y));
//
//        // Get the position of right keypoints
//        x= kp2[it.trainIdx].pt.x;
//        y= kp2[it.trainIdx].pt.y;
//        points2.add(new Point(x,y));
//    }
//        // Compute F matrix using RANSAC
//        Mat inliers = new Mat();
//        MatOfPoint2f p1 = new MatOfPoint2f();
//        p1.fromList(points1);
//        MatOfPoint2f p2= new MatOfPoint2f();
//        p2.fromList(points2);
//
//        Mat fundemental = Calib3d.findFundamentalMat(
//                p1,
//                p2,
//                Calib3d.FM_RANSAC,
//                3.0,     // distance to epipolar line
//                kConfidence,    // confidence probability
//                inliers);
//
//        // extract the surviving (inliers) matches
//
//
//        int[] itIn= inliers;
//
//        std::vector<cv::DMatch>::const_iterator
//
//            itM= matches.begin();
//
//        // for all matches
//
//        for (int i = 0;itIn!= inliers.end(); ++itIn, ++itM) {
//
//            if (*itIn) {
//
//                outMatches.push_back(*itM);
//
//            }
//
//        }
//
////    if (refineF) {
//
//        if (true) {
//
//            // The F matrix will be recomputed with
//
//            // all accepted matches
//
//            // Convert keypoints into Point2f
//
//            // for final F computation
//
//            points1.clear();
//
//            points2.clear();
//
//            for (std::vector<cv::DMatch>::
//
//            const_iterator it= outMatches.begin();
//
//            it!= outMatches.end(); ++it) {
//
//                // Get the position of left keypoints
//
//                float x= keypoints1[it->queryIdx].pt.x;
//
//                float y= keypoints1[it->queryIdx].pt.y;
//
//                points1.push_back(cv::Point2f(x,y));
//
//                // Get the position of right keypoints
//
//                x= keypoints2[it->trainIdx].pt.x;
//
//                y= keypoints2[it->trainIdx].pt.y;
//
//                points2.push_back(cv::Point2f(x,y));
//
//            }
//
//            // Compute 8-point F from all accepted matches
//
//            fundemental= cv::findFundamentalMat(
//
//                    cv::Mat(points1),cv::Mat(points2), // matches
//
//                    CV_FM_8POINT); // 8-point method
//
//        }
//        return fundemental;
    }


}
