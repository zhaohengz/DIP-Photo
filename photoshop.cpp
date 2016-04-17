#include "photoshop.h"
#include <QFileDialog>
#include <QMessageBox>
#include <QString>
#include <iostream>
#include <ctime>

using namespace cv;
using namespace std;
using Eigen::MatrixXd;

int mouse_state = 0;

void poisson_on_mouse(int event, int x, int y, int flags, void* param)
{
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		mouse_state = 1;
	}

	if (event == CV_EVENT_MOUSEMOVE)
	{
		if (mouse_state == 1)
		{
			auto vec_param = (std::vector<void*> *)(param);
			auto candidate = (Mat*)((*vec_param)[0]);
			Mat* image_show = (Mat*)((*vec_param)[1]);
			Mat* image_proc = (Mat*)((*vec_param)[2]);
			Mat* binMask = (Mat*)((*vec_param)[3]);
			*image_show = image_proc->clone();
			if (x >=0 && y >= 0 && x + candidate->cols < image_show->cols && y + candidate->rows < image_show->rows)
			{
				candidate->copyTo((*image_show)(cv::Rect(Point(x, y), Point(x + candidate->cols, y + candidate->rows))), *binMask);
				imshow("PoissonMatting", *image_show);
			}
		}
	}

	if (event == CV_EVENT_LBUTTONUP)
	{
		mouse_state = 0;
		int *px, *py;
		auto vec_param = (std::vector<void*> *)(param);
		px = (int*)((*vec_param)[4]);
		py = (int*)((*vec_param)[5]);
		auto candidate = (Mat*)((*vec_param)[0]);
		Mat* image_show = (Mat*)((*vec_param)[1]);
		if (x >= 0 && y >= 0 && x + candidate->cols < image_show->cols && y + candidate->rows < image_show->rows)
		{
			*px = x;
			*py = y;
		}

	}
}


Photoshop::Photoshop(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);
	connect(ui.loadImage, SIGNAL(clicked()), this, SLOT(loadImage()));
	connect(ui.brightnessSlider, SIGNAL(sliderReleased()), this, SLOT(imageOverride()));
	connect(ui.brightnessSlider, SIGNAL(valueChanged(int)), this, SLOT(changeBrightness(int)));
	connect(ui.contrastSlider, SIGNAL(sliderReleased()), this, SLOT(imageOverride()));
	connect(ui.contrastSlider, SIGNAL(valueChanged(int)), this, SLOT(changeContrast(int)));
	connect(ui.gammaSlider, SIGNAL(sliderReleased()), this, SLOT(imageOverride()));
	connect(ui.gammaSlider, SIGNAL(valueChanged(int)), this, SLOT(changeGamma(int)));
	connect(ui.pushButtonStretch, SIGNAL(clicked()), this, SLOT(contrastStretch()));
	connect(ui.pushButtonHistogramEqualize, SIGNAL(clicked()), this ,SLOT(histogramEqualize()));
	connect(ui.pushButtonHistogramMatch, SIGNAL(clicked()), this, SLOT(histogramMatch()));
	connect(ui.pushButtonPoisson, SIGNAL(clicked()), this, SLOT(poissonMatting()));
	connect(ui.reset, SIGNAL(clicked()), this, SLOT(reset()));
}

Photoshop::~Photoshop()
{

}

void Photoshop::loadImage()
{
	_image_name = QFileDialog::getOpenFileName(this, tr("选择图像"), "", tr("Images (*.png *.jpg *.jpeg *.bmp)"));
	_min_contrast = 0;
	_max_contrast = 255;

	if (_image_name.isEmpty())
	{
		return;
	}
	else
	{
		_origin_image = cv::imread(_image_name.toStdString().c_str());
		_image_proc = _origin_image.clone();
		_image_show = _origin_image.clone();
		updateImage();
	}
}

void Photoshop::updateImage()
{
	cv::Mat temp;
	cv::cvtColor(_image_show, temp, CV_BGR2RGB);
	QImage img((const unsigned char*)(temp.data), temp.cols, temp.rows, temp.cols * temp.channels(), QImage::Format_RGB888);

	ui.imageShow->setPixmap(QPixmap::fromImage( 
		img.scaled(ui.imageShow->width(), ui.imageShow->height(), Qt::KeepAspectRatio)));
}

void Photoshop::contrastStretch(int min, int max)
{
	for (int i = 0; i < 256; i++)
	{
		_LUT[i] = (max - min) * (double(i - _min_contrast)) / (_max_contrast - _min_contrast) + min;
		_LUT[i] = std::max(_LUT[i], 0);
		_LUT[i] = std::min(_LUT[i], 255);
	}

	lookUp();
	updateImage();
	_min_contrast = min;
	_max_contrast = max;
	imageOverride();
}

void Photoshop::lookUp()
{
	for (int i = 0; i < _image_proc.rows; i++)
	{
		for (int j = 0; j < _image_proc.cols; j++)
		{
			Vec3b* temp = _image_show.ptr<Vec3b>(i, j);
			Vec3b* origin = _image_proc.ptr<Vec3b>(i, j);
			for (int k = 0; k < 3; k++)
			{
				temp->val[k] = _LUT[origin->val[k]];
			}
		}
	}
}

void Photoshop::changeBrightness(int value)
{
	for (int i = 0; i < 256; i++)
	{
		_LUT[i] = i - 127 + value;
		_LUT[i] = max(_LUT[i], 0);
		_LUT[i] = min(_LUT[i], 255);
	}
	lookUp();
	updateImage();
}

void Photoshop::changeContrast(int value)
{
	double ratio = exp(double(value) / 100);
	for (int i = 0; i < 256; i++)
	{
		_LUT[i] = ratio * (i - 127) + 127;
		_LUT[i] = max(_LUT[i], 0);
		_LUT[i] = min(_LUT[i], 255);
	}

	lookUp();
	updateImage();
}

void Photoshop::changeGamma(int value)
{
	double gamma = exp(double(value) / 100);
	for (int i = 0; i < 256; i++)
	{
		_LUT[i] = 255 * pow(double(i) / 255, double(1) / gamma);
		_LUT[i] = max(_LUT[i], 0);
		_LUT[i] = min(_LUT[i], 255);
	}

	lookUp();
	updateImage();
}

void Photoshop::histogramEqualize()
{
	int pixel_num = _image_proc.rows * _image_proc.cols * _image_proc.channels();
	memset(_LUT, 0, sizeof(_LUT));
	for (int i = 0; i < _image_proc.rows; i++)
	{
		for (int j = 0; j < _image_proc.cols; j++)
		{
			Vec3b temp = _image_proc.at<Vec3b>(i, j);
			for (int k = 0; k < 3; k++)
			{
				_LUT[temp.val[k]]++;
			}
		}
	}

	int sum = 0;
	for (int i = 0; i < 256; i++)
	{
		sum += _LUT[i];
		_LUT[i] = 255 * double(sum) / pixel_num;
	}
	
	lookUp();
	updateImage();
	imageOverride();
}

void Photoshop::histogramMatch()
{
	QString candidate_name = QFileDialog::getOpenFileName(this, tr("选择图像"), "", tr("Images (*.png *.jpg *.jpeg *.bmp)"));

	if (candidate_name.isEmpty())
	{
		return;
	}
	else
	{
		_candidates.resize(0);
		double p_candidate[256];
		memset(p_candidate, 0, sizeof(p_candidate));
		_candidates.push_back(cv::imread(candidate_name.toStdString().c_str()));
		auto candidate = _candidates.begin();
		int candidate_num = candidate->rows * candidate->cols * candidate->channels();
		for (int i = 0; i < candidate->rows; i++)
		{
			for (int j = 0; j < candidate->cols; j++)
			{
				Vec3b temp = candidate->at<Vec3b>(i, j);
				
				for (int k = 0; k < 3; k++)
				{
					p_candidate[temp.val[k]]++;
				}
			}
		}
		
		p_candidate[0] /= candidate_num;
		for (int i = 1; i < 256; i++)
		{
			p_candidate[i] /= candidate_num;
			p_candidate[i] += p_candidate[i - 1];
		}


		int pixel_num = _image_proc.rows * _image_proc.cols * _image_proc.channels();
		memset(_LUT, 0, sizeof(_LUT));
		for (int i = 0; i < _image_proc.rows; i++)
		{
			for (int j = 0; j < _image_proc.cols; j++)
			{
				Vec3b temp = _image_proc.at<Vec3b>(i, j);
				for (int k = 0; k < 3; k++)
				{
					_LUT[temp.val[k]]++;
				}
			}
		}

		int sum = 0;
		int j = 0;
		for (int i = 0; i < 256; i++)
		{
			sum += _LUT[i];
			double temp = double(sum) / pixel_num;
			while (p_candidate[j] < temp && j < 256)
			{
				j++;
			}
			if (j < 256)
			{
				_LUT[i] = j;
			}
			else
			{
				for (int k = i; k < 256; k++)
				{
					_LUT[k] = 255;
				}
				break;
			}
		}
		

		lookUp();
		updateImage();
		imageOverride();
	}
}

void Photoshop::reset()
{
	ui.contrastSlider->setValue(0);
	ui.brightnessSlider->setValue(127);
	ui.gammaSlider->setValue(0);
	_min_contrast = 0;
	_max_contrast = 255;
	ui.spinBoxMinContrast->setValue(_min_contrast);
	ui.spinBoxMaxContrast->setValue(_max_contrast);
	_image_proc = _origin_image.clone();
	_image_show = _origin_image.clone();
	updateImage();
}

void Photoshop::imageOverride()
{
	_image_proc = _image_show.clone();
}

void Photoshop::contrastStretch()
{
	contrastStretch(ui.spinBoxMinContrast->value(), ui.spinBoxMaxContrast->value());
}

void Photoshop::poissonMatting()
{
	QString candidate_name = QFileDialog::getOpenFileName(this, tr("选择图像"), "", tr("Images (*.png *.jpg *.jpeg *.bmp)"));

	if (candidate_name.isEmpty())
	{
		return;
	}
	else
	{
		_candidates.resize(0);
		_candidates.push_back(cv::imread(candidate_name.toStdString().c_str()));
		Mat* candidate = &(_candidates[0]);
		Mat bgdModel, fgdModel, mask;
		mask = cvCreateMat(candidate->rows, candidate->cols, CV_8UC1);
		mask.setTo(GC_PR_FGD);
		for (int i = 0; i < 0.1 * mask.rows; i++)
		{
			for (int j = 0; j < mask.cols; j++)
			{
				mask.at<unsigned char>(i, j) = GC_PR_BGD;
			}
		}

		for (int i = 0; i < mask.rows; i++)
		{
			for (int j = 0; j < 0.1 * mask.cols; j++)
			{
				mask.at<unsigned char>(i, j) = GC_PR_BGD;
				mask.at<unsigned char>(i, mask.cols - j - 1) = GC_PR_BGD;
			}
		}
		grabCut(*candidate, mask, Rect(0, 0, candidate->cols, candidate->rows), bgdModel, fgdModel, 1);
		Mat binMask = mask & 1;
		Mat res;
		candidate->copyTo(res, binMask);
		_image_show = _image_proc.clone();
		candidate->copyTo(_image_show(cv::Rect(Point(0, 0), Point(candidate->cols, candidate->rows))), binMask);
		cvNamedWindow("PoissonMatting");
		std::vector<void*> params;
		int x, y;
		params.push_back((void*)candidate);
		params.push_back((void*)(&_image_show));
		params.push_back((void*)(&_image_proc));
		params.push_back((void*)(&binMask));
		params.push_back((void*)&x);
		params.push_back((void*)&y);
		cvSetMouseCallback("PoissonMatting", poisson_on_mouse, (void*)(&params));
		imshow("PoissonMatting", _image_show);
		cvWaitKey(0);
		cvDestroyWindow("PoissonMatting");

		cout << x << ' ' << y << endl;
		Mat dest, destGradientX, destGradientY;
		Mat patchGradientX, patchGradientY;
		_image_show(cv::Rect(Point(x, y), Point(x + candidate->cols, y + candidate->rows))).copyTo(dest);
		patchGradientX = Mat(candidate->size(), CV_8UC3);
		patchGradientY = Mat(candidate->size(), CV_8UC3);
		destGradientX = Mat(dest.size(), CV_8UC3);
		destGradientY = Mat(dest.size(), CV_8UC3);
		Mat test = _image_show.clone();
		computeGradientX(_image_show, test);
		imshow("test", test);
		cvWaitKey(0);
		cvDestroyAllWindows();
		
		computeGradientX(*candidate, patchGradientX);
		computeGradientY(*candidate, patchGradientY);
		computeGradientX(dest, destGradientX);
		computeGradientY(dest, destGradientY);

		Mat laplacianX = destGradientX.clone();
		Mat laplacianY = destGradientY.clone();
		patchGradientX.copyTo(laplacianX, binMask);
		patchGradientY.copyTo(laplacianY, binMask);

		computeLaplacianX(laplacianX, laplacianX);
		computeLaplacianY(laplacianY, laplacianY);

		Mat lap = laplacianX + laplacianY;
		Mat ans = dest.clone();
		int pixNum = dest.rows * dest.cols;

		double* columnB = new double[pixNum];
		double* pX = new double[pixNum];

		cout << pixNum << endl;
		Eigen::SparseMatrix<float> smA(pixNum, pixNum);
		for (int i = 0; i < dest.rows; i++)
		{
			for (int j = 0; j < dest.cols; j++)
			{
				int place = i * dest.cols + j;
				if (i == 0 || j == 0 || i == dest.rows - 1 || j == dest.cols - 1)
				{
					smA.insert(place, place) = 1;
					continue;
				}
				smA.insert(place, place) = -4;
				smA.insert(place, (i - 1) * dest.cols + j) = 1;
				smA.insert(place, (i + 1) * dest.cols + j) = 1;
				smA.insert(place, i * dest.cols + j - 1) = 1;
				smA.insert(place, i * dest.cols + j + 1) = 1;
			}
		}
		smA.makeCompressed();
		for (int k = 0; k < 3; k++)
		{
			for (int i = 0; i < dest.rows; i++)
			{
				for (int j = 0; j < dest.cols; j++)
				{
					int place = i * dest.cols + j;
					if (i == 0 || j == 0 || i == dest.rows - 1 || j == dest.cols - 1)
					{
						columnB[place] = dest.at<Vec3b>(i, j).val[k];
						continue;
					}
					columnB[place] = lap.at<Vec3b>(i, j).val[k];
				}
			}
			cout << (int)time(NULL) << endl;
			SolveLinearSystemByEigenQR(smA, columnB, pX, pixNum, pixNum);
			cout << (int)time(NULL) << endl;

			for (int i = 0; i < dest.rows; i++)
			{
				for (int j = 0; j < dest.cols; j++)
				{
					int place = i * dest.cols + j;
					ans.at<Vec3b>(i, j).val[k] = pX[place];
				}
			}
		}
		ans.copyTo(_image_show(cv::Rect(Point(x, y), Point(x + candidate->cols, y + candidate->rows))));
		
		delete[] columnB;
		delete[] pX;
		//std::cout << smA << std::endl;
			
		updateImage();
		imageOverride();
	}
}

void Photoshop::computeGradientX(Mat& img, Mat& gradient)
{
	Mat kernel = Mat::zeros(1, 3, CV_8S);
	kernel.at<char>(0, 2) = 1;
	kernel.at<char>(0, 1) = -1;
	filter2D(img, gradient, CV_8UC3, kernel);
}

void Photoshop::computeGradientY(Mat& img, Mat& gradient)
{
	Mat kernel = Mat::zeros(3, 1, CV_8S);
	kernel.at<char>(2, 0) = 1;
	kernel.at<char>(1, 0) = -1;
	filter2D(img, gradient, CV_8UC3, kernel);
}

void Photoshop::computeLaplacianX(cv::Mat & img, cv::Mat & laplacian)
{
	Mat kernel = Mat::zeros(1, 3, CV_8S);
	kernel.at<char>(0, 0) = -1;
	kernel.at<char>(0, 1) = 1;
	filter2D(img, laplacian, CV_8UC3, kernel);
}

void Photoshop::computeLaplacianY(cv::Mat & img, cv::Mat & laplacian)
{
	Mat kernel = Mat::zeros(3, 1, CV_8S);
	kernel.at<char>(0, 0) = -1;
	kernel.at<char>(1, 0) = 1;
	filter2D(img, laplacian, CV_8UC3, kernel);
}

int Photoshop::SolveLinearSystemByEigenQR(Eigen::SparseMatrix<float>& smA, double* pColumnB, double* pX, int m_nRow, int m_nColumn)
{

	Eigen::SparseQR<Eigen::SparseMatrix<float>, Eigen::COLAMDOrdering<int>> linearSolver;
	linearSolver.compute(smA);

	Eigen::VectorXf vecB(m_nRow);
	for (int i = 0; i < m_nRow; ++i)
	{
		vecB[i] = pColumnB[i];
	}
	Eigen::VectorXf vecX = linearSolver.solve(vecB);

	for (int i = 0; i < m_nColumn; ++i)
	{
		pX[i] = vecX[i];
	}

	return 1;
}
