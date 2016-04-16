#ifndef PHOTOSHOP_H
#define PHOTOSHOP_H

#include <QtWidgets/QMainWindow>
#include "ui_photoshop.h"
#include <opencv2\opencv.hpp>
#include <vector>

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
	void reset();
	void imageOverride();
	void contrastStretch();

private:
	void updateImage();
	void lookUp();
	void contrastStretch(int min, int max);

	Ui::PhotoshopClass ui;
	int _LUT[255];				// Look Up Table
	int _min_contrast, _max_contrast;
	QString _image_name;
	cv::Mat _origin_image, _image_proc, _image_show;
	std::vector<cv::Mat> _candidates;
};

#endif // PHOTOSHOP_H
