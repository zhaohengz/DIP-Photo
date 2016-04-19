#pragma once
#include <opencv2\opencv.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\nonfree\nonfree.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <vector>

class ImageStitch
{
public:
	ImageStitch();
	~ImageStitch();
	double distance(float* a, float* b);
	void HomoTransform(cv::Mat & H, cv::Point2f & src, cv::Point2f & dst);
	void exec(cv::Mat& img1, cv::Mat& img2, cv::Mat& res);
	void RANSAC(std::vector<cv::DMatch>& matches, double p_min, double err_max, cv::Mat& homo_matrix);
	void findConsensus(std::vector<cv::DMatch>& matches, cv::Mat& H, double err_max, std::vector<int>& inliers);

	cv::SIFT sift1, sift2;
	std::vector<cv::KeyPoint> key_points1, key_points2;
	cv::Mat descriptors1, descriptors2;
};

