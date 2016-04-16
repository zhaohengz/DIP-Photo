#include "photoshop.h"
#include <QFileDialog>
#include <QMessageBox>
#include <QString>
#include <iostream>

using namespace cv;
using namespace std;


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
	connect(ui.pushButtonHistogramEqualize, SIGNAL(clicked()), SLOT(histogramEqualize()));
	connect(ui.pushButtonHistogramMatch, SIGNAL(clicked()), SLOT(histogramMatch()));
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
	cout << min << ' ' << max << endl;
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
