#include <iostream>
#include <FeatureExtraction.h>
#include "Visualisation.h"

using namespace std;

const std::string image_path = "..\\numbers.bmp";

int main() 
{
	cout << "hello world!" << endl;
	cout << fe::GetTestString().c_str() << endl;

    size_t pm_n_count;
    size_t pm_px_radius;

    auto pm = fe::CreatePolynomialManager();
    auto bp = fe::CreateBlobProcessor();

    cout << endl;
    cout << "******** Basis Info ********" << endl;
    cout << pm->GetType() << endl;
    cout << "****************************" << endl;
    cout << endl;

    cout << "Enter polynomial count and radius: ";
    cin >> pm_n_count >> pm_px_radius;
    cout << endl;

    pm->InitBasis(pm_n_count, pm_px_radius);

    ShowPolynomials("Polynomials", pm->GetBasis());
    cv::waitKey();
    cv::destroyAllWindows();

    cv::Mat raw_numbers = cv::imread(image_path, CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat numbers;
    cv::threshold(raw_numbers, numbers, 127, 255, CV_THRESH_BINARY_INV);

    Show64FC1Mat("numbers", numbers);
    cv::waitKey();
    cv::destroyAllWindows();

    cout << endl;
    cout << "*** Blob Processor Info ****" << endl;
    cout << bp->GetType() << endl;
    cout << "****************************" << endl;
    cout << endl;

    std::vector < cv::Mat > blobs;
    std::vector < cv::Mat > normalized_blobs;
    std::vector < cv::Mat > recovered_blobs;
    std::vector < fe::ComplexMoments > blob_decompositions;

    bp->DetectBlobs(numbers, blobs);
    bp->NormalizeBlobs(blobs, normalized_blobs, pm_px_radius);

    blob_decompositions.resize(blobs.size());
    recovered_blobs.resize(blobs.size());

    for (size_t i = 0; i < blobs.size(); ++i)
    {
        pm->Decompose(normalized_blobs[i], blob_decompositions[i]);
    }

    for (size_t i = 0; i < blobs.size(); ++i)
    {
        pm->Recovery(blob_decompositions[i], recovered_blobs[i]);

        ShowBlobDecomposition(std::to_string(i + 1),
                              normalized_blobs[i], recovered_blobs[i]);
        cv::waitKey();
        cv::destroyAllWindows();
    }

	system("pause");
	return 0;
}