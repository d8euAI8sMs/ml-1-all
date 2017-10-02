#include "windows.h"

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include "FeatureExtraction.h"
#include "MomentsHelper.h"
#include "MomentsRecognizerImpl.h"

using namespace cv;
using namespace std;
using namespace fe;

static const string labeled_data        = "..\\data\\labeled_data";
static const string ground_data         = "..\\data\\ground_data";
static const string test_data           = "..\\data\\test_data";
static const string cv_ground_data      = "..\\data\\cv_ground_data";
static const string cv_test_data        = "..\\data\\cv_test_data";
static const string ground_data_moments = "..\\data\\ground_data_moments.dat";
static const string test_data_moments   = "..\\data\\test_data_moment.dat";
static const string cv_ground_data_moments = "..\\data\\cv_ground_data_moments.dat";
static const string cv_test_data_moments   = "..\\data\\cv_test_data_moment.dat";
static const string serialized_network  = "..\\data\\network.dat";

void generateData()
{
	cout << "===Generate data!===" << endl;

    bool result;

    cout << "  Split data onto ground and test... ";

    result = MomentsHelper::DistributeData(labeled_data,
                                           ground_data,
                                           test_data,
                                           50);
    if (result) cout << "success" << endl;
    else        cout << "failed" << endl;

    auto bp = CreateBlobProcessor();
    auto pm = CreatePolynomialManager();

    map < string, vector < ComplexMoments > > moments;

    cout << "  Initialize polynomial basis... " << endl;

    size_t base_n_max, base_diameter;

    cout << "    [input] Base order and diameter (pixels): [order diameter] ";
    cin >> base_n_max >> base_diameter;

    pm->InitBasis(base_n_max, base_diameter);

    cout << "    success " << endl;

    cout << "  Calculate ground data moments... ";

    result = MomentsHelper::GenerateMoments(ground_data, bp, pm, moments);

    if (result) cout << "success" << endl;
    else        cout << "failed" << endl;

    cout << "  Save ground data moments... ";

    result = MomentsHelper::SaveMoments(ground_data_moments, moments);

    if (result) cout << "success" << endl;
    else        cout << "failed" << endl;

    cout << "  Calculate test data moments... ";

    moments.clear();

    result = MomentsHelper::GenerateMoments(test_data, bp, pm, moments);

    if (result) cout << "success" << endl;
    else        cout << "failed" << endl;

    cout << "  Save test data moments... ";

    result = MomentsHelper::SaveMoments(test_data_moments, moments);

    if (result) cout << "success" << endl;
    else        cout << "failed" << endl;
}

void trainNetwork()
{
	cout << "===Train network!===" << endl;

    bool result;

    MomentsRecognizerImpl mr;
    map < string, vector < ComplexMoments > > moments;

    cout << "  Read training data moments... ";

    result = MomentsHelper::ReadMoments(ground_data_moments, moments);

    if (result) cout << "success" << endl;
    else        cout << "failed" << endl;

    vector < int > ann_config;
    size_t num_of_hiddne_layers;
    float ann_precision; float ann_speed;
    size_t ann_max_iters;

    cout << "  Train network... " << endl;

    cout << "    [input] Number of hidden layers: [N] ";
    cin >> num_of_hiddne_layers;
    ann_config.resize(num_of_hiddne_layers);

    cout << "    [input] Number of neurons in each layer: [M1 M2 M3 ... MN] ";
    for (size_t i = 0; i < num_of_hiddne_layers; ++i)
    {
        cin >> ann_config[i];
    }

    cout << "    [input] Max iteration number and expected precision: [uint float] ";
    cin >> ann_max_iters; cin >> ann_precision;

    cout << "    [input] Training speed: [float] ";
    cin >> ann_speed;

    result = mr.Train(moments, ann_config, ann_max_iters, ann_precision, ann_speed);

    if (result) cout << "    success" << endl;
    else        cout << "    failed" << endl;

    cout << "  Save network... ";

    result = mr.Save(serialized_network);

    if (result) cout << "success" << endl;
    else        cout << "failed" << endl;
}

void precisionTest()
{
	cout << "===Precision test!===" << endl;

    bool result;

    MomentsRecognizerImpl mr;
    map < string, vector < ComplexMoments > > moments;

    cout << "  Read network... ";

    result = mr.Read(serialized_network);

    if (result) cout << "success" << endl;
    else        cout << "failed" << endl;

    cout << "  Read test data moments... ";

    result = MomentsHelper::ReadMoments(test_data_moments, moments);

    if (result) cout << "success" << endl;
    else        cout << "failed" << endl;

    cout << "  Perform precision test of network prediction results... ";

    double precision = mr.PrecisionTest(moments);

    cout << "success" << endl;

    cout << "    Result: " << precision << endl;
}

void recognizeImage()
{
	cout << "===Recognize single image!===" << endl;

    bool result;

    MomentsRecognizerImpl mr;
    vector < Mat > blobs, nblobs;
    vector < ComplexMoments > moments;
    string path;

    auto bp = CreateBlobProcessor();
    auto pm = CreatePolynomialManager();

    cout << "  Read network... ";

    result = mr.Read(serialized_network);

    if (result) cout << "success" << endl;
    else        cout << "failed" << endl;

    cout << "  Reading image... " << endl;
    cout << "    [input] Image path: [path] ";
    cin >> path;

    Mat img = imread(path, CV_LOAD_IMAGE_GRAYSCALE);

    if (img.data == NULL)
    {
        cout << "    failure" << endl;
        cout << "    Cannot continue... aborting" << endl;
        return;
    }
    else cout << "    success" << endl;

    threshold(img, img, 127, 255, CV_THRESH_BINARY_INV);

    cout << "  Initializing polynomial basis... " << endl;

    size_t base_n_max, base_diameter;

    cout << "    [input] Base order and diameter (pixels): [order diameter] ";
    cin >> base_n_max >> base_diameter;

    pm->InitBasis(base_n_max, base_diameter);

    cout << "    success " << endl;

    cout << "  Decomposing image... " << endl;

    bp->DetectBlobs(img, blobs);
    bp->NormalizeBlobs(blobs, nblobs, base_diameter);

    moments.resize(blobs.size());

    cout << "    Detected " << blobs.size() << " images" << endl;
    cout << "    Decomposing ";

    for (size_t i = 0; i < blobs.size(); ++i)
    {
        pm->Decompose(nblobs[i], moments[i]);
        cout << (i + 1) << " ";
    }

    cout << endl << "    success" << endl;

    cout << "  Recognizing images... " << endl;
    cout << "    Recognized: ";

    for (size_t i = 0; i < blobs.size(); ++i)
    {
        string recognition_resilt = mr.Recognize(moments[i]);
        cout << recognition_resilt << " ";
        Mat blob;
        resize(nblobs[i], blob, cv::Size(200, 200));
        imshow("Recognized as: " + recognition_resilt, blob);
        waitKey(); destroyAllWindows();
    }

    cout << endl << "    success" << endl;
}

bool deleteFilesInDirs(vector < string > paths)
{
    size_t length = 1;
    for each (auto & str in paths)
    {
        length += str.length() + 1;
    }
    char * buffer = new char[length];
    buffer[length - 1] = '\0';
    size_t idx = 0;
    for each (auto & str in paths)
    {
        memcpy_s(buffer + idx, length, str.data(), str.length());
        buffer[idx + str.length()] = '\0';
        idx += str.length() + 1;
    }
    SHFILEOPSTRUCT fo = { };
    fo.pFrom = buffer;
    fo.fFlags = FOF_SILENT | FOF_NOCONFIRMATION;
    fo.wFunc = FO_DELETE;

    bool result = (SHFileOperation(&fo) == 0);

    for each (auto & str in paths)
    {
        result &= CreateDirectory(str.c_str(), NULL);
    }

    return result;
}

void cvTest()
{
	cout << "===Preform CV-test!===" << endl;

    bool result;

    size_t num_of_partitions;

    auto bp = CreateBlobProcessor();
    auto pm = CreatePolynomialManager();

    map < string, vector < ComplexMoments > > moments;

    cout << "  Initialize polynomial basis... " << endl;

    size_t base_n_max, base_diameter;

    cout << "    [input] Base order and diameter (pixels): [order diameter] ";
    cin >> base_n_max >> base_diameter;

    pm->InitBasis(base_n_max, base_diameter);

    cout << "    success " << endl;

    vector < int > ann_config;
    size_t num_of_hiddne_layers;
    float ann_precision; float ann_speed;
    size_t ann_max_iters;

    cout << "  Initialize network parameters... " << endl;

    cout << "    [input] Number of hidden layers: [N] ";
    cin >> num_of_hiddne_layers;
    ann_config.resize(num_of_hiddne_layers);

    cout << "    [input] Number of neurons in each layer: [M1 M2 M3 ... MN] ";
    for (size_t i = 0; i < num_of_hiddne_layers; ++i)
    {
        cin >> ann_config[i];
    }

    cout << "    [input] Max iteration number and expected precision: [uint float] ";
    cin >> ann_max_iters; cin >> ann_precision;

    cout << "    [input] Training speed: [float] ";
    cin >> ann_speed;

    cout << "    success" << endl;

    cout << "  Prepare for CV-test... " << endl;
    cout << "    [input] Number of partitions: [uint] ";
    cin >> num_of_partitions;
    cout << "    success" << endl;

    double precision = 0, mean_precision = 0, sqmean_precision = 0;

    for (size_t i = 0; i < num_of_partitions; ++i)
    {
        cout << "  Performing " << (i + 1) << " test... " << endl;

        cout << "    Clean up... ";

        result = deleteFilesInDirs({ cv_ground_data, cv_test_data });

        if (result) cout << "success" << endl;
        else        cout << "failed" << endl;

        cout << "    Split data onto ground and test... " << endl;

        double percent = ((double)rand() / RAND_MAX) * 50 + 25;

        cout << "      Percent is " << percent << endl;

        result = MomentsHelper::DistributeData(labeled_data,
                                               cv_ground_data,
                                               cv_test_data,
                                               percent);

        if (result) cout << "      success" << endl;
        else        cout << "      failed" << endl;

        cout << "    Calculate ground data moments... ";

        result = MomentsHelper::GenerateMoments(cv_ground_data, bp, pm, moments);

        if (result) cout << "success" << endl;
        else        cout << "failed" << endl;

        cout << "    Save ground data moments... ";

        result = MomentsHelper::SaveMoments(cv_ground_data_moments, moments);

        if (result) cout << "success" << endl;
        else        cout << "failed" << endl;

        cout << "    Calculate test data moments... ";

        moments.clear();

        result = MomentsHelper::GenerateMoments(cv_test_data, bp, pm, moments);

        if (result) cout << "success" << endl;
        else        cout << "failed" << endl;

        cout << "    Save test data moments... ";

        result = MomentsHelper::SaveMoments(cv_test_data_moments, moments);

        if (result) cout << "success" << endl;
        else        cout << "failed" << endl;

        MomentsRecognizerImpl mr;

        moments.clear();

        cout << "    Read training data moments... ";

        result = MomentsHelper::ReadMoments(cv_ground_data_moments, moments);

        if (result) cout << "success" << endl;
        else        cout << "failed" << endl;

        cout << "    Train network... ";

        result = mr.Train(moments, ann_config, ann_max_iters, ann_precision, ann_speed);

        if (result) cout << "success" << endl;
        else        cout << "failed" << endl;

        moments.clear();

        cout << "    Read test data moments... ";

        result = MomentsHelper::ReadMoments(cv_test_data_moments, moments);

        if (result) cout << "success" << endl;
        else        cout << "failed" << endl;

        cout << "    Perform precision test of network prediction results... ";

        precision = mr.PrecisionTest(moments);
        mean_precision += precision;
        sqmean_precision += precision * precision;

        cout << "success" << endl;

        cout << "      Result: " << precision << endl;

        moments.clear();

        cout << "    Read training data moments... ";

        result = MomentsHelper::ReadMoments(cv_test_data_moments, moments);

        if (result) cout << "success" << endl;
        else        cout << "failed" << endl;

        cout << "    Train network... ";

        result = mr.Train(moments, ann_config, ann_max_iters, ann_precision, ann_speed);

        if (result) cout << "success" << endl;
        else        cout << "failed" << endl;

        moments.clear();

        cout << "    Read test data moments... ";

        result = MomentsHelper::ReadMoments(cv_ground_data_moments, moments);

        if (result) cout << "success" << endl;
        else        cout << "failed" << endl;

        cout << "    Perform precision test of network prediction results... ";

        precision = mr.PrecisionTest(moments);
        mean_precision += precision;
        sqmean_precision += precision * precision;

        cout << "success" << endl;

        cout << "      Result: " << precision << endl;
    }

    mean_precision /= num_of_partitions * 2;
    sqmean_precision /= num_of_partitions * 2;
    sqmean_precision = std::sqrt(sqmean_precision);

    cout << "  success" << endl;

    cout << "  Mean precision is " << mean_precision << endl;
    cout << "  Mean square precision is " << sqmean_precision << endl;
}

void cleanup()
{
	cout << "===Cleanup!===" << endl;

    bool result;
    string query;

    cout << "  Clean up... ";

    result = deleteFilesInDirs({ ground_data, test_data, cv_ground_data, cv_test_data });

    if (result) cout << "success" << endl;
    else        cout << "failed" << endl;
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
		cout << "  '5' - to check recognition algorithm precision." << endl;
		cout << "  '6' - to cleanup." << endl;
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
		else if (key == "5") {
			cvTest();
		}
		else if (key == "6") {
			cleanup();
		}
		cout << endl;
	} while (key != "exit");
	return 0;
}
