#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include "FeatureExtraction.h"
#include "MomentsHelper.h"

using namespace cv;
using namespace std;
using namespace fe;

void generateData()
{
	cout << "===Generate data!===" << endl;
}

void trainNetwork()
{
	cout << "===Train network!===" << endl;
}

void precisionTest()
{
	cout << "===Precision test!===" << endl;
}

void recognizeImage()
{
	cout << "===Recognize single image!===" << endl;
}

int main(int argc, char** argv)
{
	string key;
	do 
	{
		cout << "===Enter next walues to do something:===" << endl;
		cout << "  '1' - to generate data." << endl;
		cout << "  '2' - to train network." << endl;
		cout << "  '3' - to check recognizing precision." << endl;
		cout << "  '4' - to recognize single image." << endl;
		cout << "  'exit' - to close the application." << endl;
		cin >> key;
		cout << endl;
		if (key == "1") {
			generateData();
		}
		else if (key == "2") {
			trainNetwork();
		}
		else if (key == "3") {
			precisionTest();
		}
		else if (key == "4") {
			recognizeImage();
		}
		cout << endl;
	} while (key != "exit");
	return 0;
}