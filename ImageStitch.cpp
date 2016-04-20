#include "ImageStitch.h"
#include <fstream>
#include <cstdlib>
#include <ctime>

using namespace cv;
using namespace std;

ImageStitch::ImageStitch()
{
}


ImageStitch::~ImageStitch()
{
}


double ImageStitch::distance(float* a, float* b)
{
	double dis = 0;
	for (int i = 0; i < 128; i++)
	{
		double temp = a[i] - b[i];
		dis += temp * temp;
	}
	return dis;
}

void ImageStitch::HomoTransform(cv::Mat& H, cv::Point2f& src, cv::Point2f& dst)
{
	double z = H.at<double>(2, 0) * src.x + H.at<double>(2, 1) * src.y + H.at<double>(2, 2);
	dst.x = (H.at<double>(0, 0) * src.x + H.at<double>(0, 1) * src.y + H.at<double>(0, 2)) / z;
	dst.y = (H.at<double>(1, 0) * src.x + H.at<double>(1, 1) * src.y + H.at<double>(1, 2)) / z;
}

void ImageStitch::exec(cv::Mat & img1, cv::Mat & img2, cv::Mat& res)
{
	Mat mascara;


	sift1(img1, mascara, key_points1, descriptors1);
	sift2(img2, mascara, key_points2, descriptors2);

	vector<DMatch>matches;
	for (int i = 0; i < key_points1.size(); i++)
	{
		float* a = descriptors1.ptr<float>(i);
		double max_2 = -1;
		int idx_2, idx_1;
		double max_1 = -1;
		for (int j = 0; j < key_points2.size(); j++)
		{
			float* b = descriptors2.ptr<float>(j);

			double dis = distance(a, b);
			if (dis < max_1 || max_1 == -1)
			{
				max_2 = max_1;
				idx_2 = idx_1;
				max_1 = dis;
				idx_1 = j;
			}
			else if (dis < max_2 || max_2 == -1)
			{
				max_2 = dis;
				idx_2 = j;
			}
		}

		//std::oo << max_1 << ' ' << max_2 << std::endl;
		if ((max_1 / max_2) <= 0.7)
		{
			matches.push_back(DMatch(i, idx_1, 0, sqrt(max_1)));
		}
	}

	Mat homo_matrix;
	RANSAC(matches, 0.01, 3, homo_matrix);
	Mat homo_inverted = homo_matrix.clone();
	invert(homo_matrix, homo_inverted);

	Point2f img2_lu, img2_ld, img2_ru, img2_rd;    //left up, left down, ...
	Point2f temp(0, 0);
	HomoTransform(homo_inverted, temp, img2_lu);
	temp = Point(0, img2.rows - 1);
	HomoTransform(homo_inverted, temp, img2_ld);
	temp = Point(img2.cols - 1, 0);
	HomoTransform(homo_inverted, temp, img2_ru);
	temp = Point(img2.cols - 1, img2.rows - 1);
	HomoTransform(homo_inverted, temp, img2_rd);
	res= Mat(cvSize(max(img2_ru.x, img2_rd.x), max(img1.rows, img2.rows)), CV_8UC3);

	for (int i = 0; i < res.rows; i++)
	{
		for (int j = 0; j < res.cols; j++)
		{
			if (i < img1.rows && j < img1.cols)
			{
				res.at<Vec3b>(i, j) = img1.at<Vec3b>(i, j);
			}
			else
			{
				Point2f temp(j, i);
				Point2f hpt;
				HomoTransform(homo_matrix, temp, hpt);
				int u = hpt.x;
				int v = hpt.y;
				if (u >= 0 && u < img2.cols && v >= 0 && v < img2.rows)
				{
					res.at<Vec3b>(i, j) = img2.at<Vec3b>(v, u);
				}
				else
				{
					res.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
				}
			}
		}
	}

//imshow("dfd", res);
	//cvWaitKey(0);
	imwrite("stitch.jpg", res);
	/*cvDestroyAllWindows();

	namedWindow("SIFT_matches");
	Mat img_matches;
	//在输出图像中绘制匹配结果  
	drawMatches(img1, key_points1,         //第一幅图像和它的特征点  
		img2, key_points2,      //第二幅图像和它的特征点  
		matches,       //匹配器算子  
		img_matches,      //匹配输出图像  
		Scalar(255, 255, 255));     //用白色直线连接两幅图像中的特征点  
	imshow("SIFT_matches", img_matches);
	waitKey(0);*/
}

void ImageStitch::RANSAC(std::vector<cv::DMatch>& matches, double p_min, double err_max, cv::Mat & homo_matrix)
{
	srand((int)time(NULL));
	Mat* M = NULL;
	double p, in_frac = 0.25;
	int inlier_max = 0;
	int k = 0;
	int m = 4;
	p = pow(1.0 - pow(in_frac, m), k);
	vector<Point2f> left;
	vector<Point2f> right;
	vector<int> inliers;
	while (p > p_min)
	{
		left.resize(0);
		right.resize(0);
		int i = 0;
		while (i < 4)
		{
			int x = rand() % matches.size();
			Point2f pt = key_points1[matches[x].queryIdx].pt;
			bool flag = true;
			for (auto& it : left)
			{
				if (pt == it)
				{
					flag = false;
					break;
				}
			}
			if (flag)
			{
				i++;
				left.push_back(pt);
				right.push_back(key_points2[matches[x].trainIdx].pt);
			}
		}
		
		Mat H = findHomography(left, right);
		findConsensus(matches, H, err_max, inliers);
		if (inliers.size() > inlier_max)
		{
			homo_matrix = H.clone();
			inlier_max = inliers.size();
			in_frac = (double)(inliers.size()) / matches.size();
		}

		p = pow(1.0 - pow(in_frac, m), ++k);
	}
	/*for (auto& it : matches)
	{
		left.push_back(key_points1[it.queryIdx].pt);
		right.push_back(key_points2[it.trainIdx].pt);
	}

	homo_matrix = findHomography(left, right, CV_RANSAC);*/

	left.resize(0);
	right.resize(0);
	findConsensus(matches, homo_matrix, err_max, inliers);
	vector<DMatch> new_matches;
	for (auto& i : inliers)
	{
		left.push_back(key_points1[matches[i].queryIdx].pt);
		right.push_back(key_points2[matches[i].trainIdx].pt);
		new_matches.push_back(matches[i]);
	}
	Mat H = findHomography(left, right);
	homo_matrix = H.clone();
	matches = new_matches;
}

void ImageStitch::findConsensus(std::vector<cv::DMatch>& matches, cv::Mat& H, double err_max, std::vector<int>& inliers)
{
	assert(H.type() == CV_64F);
	inliers.resize(0);
	
	for (int i = 0; i < matches.size(); i++)
	{
		auto it = matches.at(i);
		Point2f src = key_points1[it.queryIdx].pt;
		Point2f dst = key_points2[it.trainIdx].pt;
		Point2f Hsrc;
		double z = H.at<double>(2, 0) * src.x + H.at<double>(2, 1) * src.y + H.at<double>(2, 2);
		Hsrc.x = (H.at<double>(0, 0) * src.x + H.at<double>(0, 1) * src.y + H.at<double>(0, 2)) / z;
		Hsrc.y = (H.at<double>(1, 0) * src.x + H.at<double>(1, 1) * src.y + H.at<double>(1, 2)) / z;
		double dis = sqrt((Hsrc.x - dst.x) * (Hsrc.x - dst.x) + (Hsrc.y - dst.y) * (Hsrc.y - dst.y));
		if (dis <= err_max)
		{
			inliers.push_back(i);
		}
	}
}
