#include "photoshop.h"
#include <QtWidgets/QApplication>


int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	Photoshop w;
	w.show();
	return a.exec();
}
