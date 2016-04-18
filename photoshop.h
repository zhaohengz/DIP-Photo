#ifndef PHOTOSHOP_H
#define PHOTOSHOP_H

#include <QtWidgets/QMainWindow>
#include "ui_photoshop.h"
#include <opencv2\opencv.hpp>
#include <vector>
#include <Eigen/SparseLU>


class Photoshop : public QMainWindow
{
	Q_OBJECT

public:
	Photoshop(QWidget *parent = 0);
	~Photoshop();



public slots:
	void loadImage();
	void changeBrightness(int value);
	void changeContrast(int value);
	void changeGamma(int value);
	void histogramEqualize();
	void histogramMatch();
	void poissonMatting();
	void reset();
	void imageOverride();
	void contrastStretch();

private:
	void maskMerge(cv::Mat& des, cv::Mat& src, cv::Mat& mask);
	void updateImage();
	void lookUp();
	void contrastStretch(int min, int max);
	void computeGradientX(const cv::Mat& img, cv::Mat& gradient);
	void computeGradientY(const cv::Mat& img, cv::Mat& gradient);
	void computeLaplacianX(const cv::Mat& img, cv::Mat& laplacian);
	void computeLaplacianY(const cv::Mat& img, cv::Mat& laplacian);
	int SolveLinearSystemByEigen(Eigen::SparseMatrix<float>& smA, double * pColumnB, double * pX, int m_nRow, int m_nColumn);

	Ui::PhotoshopClass ui;
	int _LUT[255];				// Look Up Table
	int _min_contrast, _max_contrast;
	QString _image_name;
	cv::Mat _origin_image, _image_proc, _image_show;
	std::vector<cv::Mat> _candidates;
};

#endif // PHOTOSHOP_H
