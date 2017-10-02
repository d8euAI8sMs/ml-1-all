#include "MomentsRecognizerImpl.h"

bool MomentsRecognizerImpl::Train(
    std::map<std::string, std::vector<fe::ComplexMoments>> moments,
    std::vector<int> layers,
    int max_iters,
    float eps,
    float speed)
{
    this->statModel = StatModel::ANN;

    this->pAnn = cv::ml::ANN_MLP::create();

    size_t num_of_inputs = moments.begin()->second.front().re.rows * 2; // re & im
    size_t num_of_outputs = moments.size(); // number of classes

    std::vector < int > all_layers = { (int)num_of_inputs };
    all_layers.insert(all_layers.end(), layers.begin(), layers.end());
    all_layers.push_back(num_of_outputs);

    pAnn->setLayerSizes(all_layers);

    this->pAnn->setBackpropMomentumScale(0.1);
    this->pAnn->setBackpropWeightScale(0.1);
    this->pAnn->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM, 0., 0.);

    cv::TermCriteria term_criteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, max_iters, eps);
    this->pAnn->setTermCriteria(term_criteria);
    this->pAnn->setTrainMethod(cv::ml::ANN_MLP::RPROP, speed);

    this->values.resize(num_of_outputs);

    size_t num_of_samples = 0;
    for (auto it = moments.begin(); it != moments.end(); ++it)
    {
        num_of_samples += it->second.size();
    }

    cv::Mat input(num_of_samples, num_of_inputs, CV_32FC1);
    cv::Mat output = cv::Mat::zeros(num_of_samples, num_of_outputs, CV_32FC1);
    size_t sample_idx = 0, class_idx = 0;
    for (auto it = moments.begin(); it != moments.end(); ++it, ++class_idx)
    {
        for (size_t i = 0; i < it->second.size(); ++i, ++sample_idx)
        {
            MomentsToInput(it->second[i]).copyTo(input.rowRange(sample_idx, sample_idx + 1));
            output.at<float>(sample_idx, class_idx) = 1.0f;
        }
        this->values[class_idx] = it->first;
    }

    return this->pAnn->train(input, cv::ml::SampleTypes::ROW_SAMPLE, output);
}

bool MomentsRecognizerImpl::TrainKNearest(
    std::map<std::string, std::vector<fe::ComplexMoments>> moments, size_t max_k)
{
    this->statModel = StatModel::KNEAREST;

    this->pKNearest = cv::ml::KNearest::create();

    size_t num_of_inputs = moments.begin()->second.front().re.rows * 2; // re & im
    size_t num_of_outputs = moments.size(); // number of classes

    this->pKNearest->setDefaultK(max_k);
    this->pKNearest->setIsClassifier(true);

    this->values.resize(num_of_outputs);

    size_t num_of_samples = 0;
    for (auto it = moments.begin(); it != moments.end(); ++it)
    {
        num_of_samples += it->second.size();
    }

    cv::Mat input(num_of_samples, num_of_inputs, CV_32FC1);
    cv::Mat output = cv::Mat::zeros(num_of_samples, 1, CV_32SC1);
    size_t sample_idx = 0, class_idx = 0;
    for (auto it = moments.begin(); it != moments.end(); ++it, ++class_idx)
    {
        for (size_t i = 0; i < it->second.size(); ++i, ++sample_idx)
        {
            MomentsToInput(it->second[i]).copyTo(input.rowRange(sample_idx, sample_idx + 1));
            output.at<int>(sample_idx, 0) = class_idx;
        }
        this->values[class_idx] = it->first;
    }

    cv::Mat var_type(num_of_inputs + 1, 1, CV_8UC1);
    var_type.setTo(cv::Scalar::all(cv::ml::VAR_ORDERED));
    var_type.at<uchar>(num_of_inputs) = cv::ml::VAR_CATEGORICAL;

    return this->pKNearest->train(
        cv::ml::TrainData::create(input, cv::ml::SampleTypes::ROW_SAMPLE, output, cv::noArray(), cv::noArray(), cv::noArray(), var_type));
}

std::string MomentsRecognizerImpl::Recognize(fe::ComplexMoments & moments)
{
	cv::Mat output;
    if (this->statModel == StatModel::ANN)
    {
	    pAnn->predict(MomentsToInput(moments), output);
    }
    else
    {
	    pKNearest->predict(MomentsToInput(moments), output);
    }
	return OutputToValue(output);
}

cv::Mat MomentsRecognizerImpl::MomentsToInput(fe::ComplexMoments & moments)
{
    cv::Mat mat(1, moments.re.rows * 2, CV_32FC1);
    for (size_t i = 0; i < moments.re.rows; ++i)
    {
        mat.at<float>(i) = moments.re.at<double>(i) + 1;
        mat.at<float>(i + moments.re.rows) = moments.im.at<double>(i) + 1;
    }
    return mat;
}

std::string MomentsRecognizerImpl::OutputToValue(cv::Mat output)
{
    if (this->statModel == StatModel::KNEAREST)
    {
        return this->values[(int)output.at<float>(0)];
    }
    size_t max_pos = 0;
    for (size_t i = 0; i < output.cols; ++i)
    {
        if (output.at<float>(i) > output.at<float>(max_pos))
        {
            max_pos = i;
        }
    }
    return this->values[max_pos];
}

bool MomentsRecognizerImpl::Save(std::string filename)
{
	cv::FileStorage fs(filename, cv::FileStorage::WRITE);
	if (!fs.isOpened()) {
		return false;
	}
    fs << "statModel" << (int)statModel;
    if (statModel == StatModel::ANN)
    {
	    pAnn->write(fs);
    }
    else
    {
	    pKNearest->write(fs);
    }
	fs << "values" << values;
	fs.release();
	return true;
}

bool MomentsRecognizerImpl::Read(std::string filename)
{
	cv::FileStorage fs(filename, cv::FileStorage::READ);
	if (!fs.isOpened()) {
		return false;
	}
    this->statModel = (StatModel)(int)fs["statModel"];
    if (this->statModel == StatModel::ANN)
    {
	    pAnn = cv::ml::ANN_MLP::create();
	    pAnn->read(fs.root());
    }
    else
    {
	    pKNearest = cv::ml::KNearest::create();
	    pKNearest->read(fs.root());
    }
	values.clear();
	for (auto iter = fs["values"].begin(); iter != fs["values"].end(); iter++) {
		values.push_back(*iter);
	}
	fs.release();
	return true;
}