#include "photoshop.h"
#include "ImageStitch.h"
#include <QtWidgets/QApplication>

using namespace cv;

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	Photoshop w;
	w.show();
	return a.exec();
}
