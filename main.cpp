#include "photoshop.h"
#include "ImageStitch.h"
#include <QtWidgets/QApplication>

using namespace cv;

int main(int argc, char *argv[])
{
	Mat img1 = imread("part1.jpeg");
	Mat img2 = imread("part2.jpeg");
	Mat res;
	ImageStitch imgSti;
	imgSti.exec(img1, img2, res);
	
	QApplication a(argc, argv);
	Photoshop w;
	w.show();
	return a.exec();
}
