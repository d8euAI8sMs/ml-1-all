#pragma once

#include "MomentsRecognizer.h"

class MomentsRecognizerImpl : public MomentsRecognizer
{

public:

    enum class StatModel
    {
        ANN,
        KNEAREST
    };

private:

    cv::Ptr < cv::ml::KNearest > pKNearest;
    StatModel statModel;

public:

    MomentsRecognizerImpl()
    {
    }

    virtual ~MomentsRecognizerImpl()
    {
    }

protected:

    virtual cv::Mat MomentsToInput(fe::ComplexMoments & moments);

    virtual std::string OutputToValue(cv::Mat output);

public:

    virtual bool Train(
        std::map<std::string, std::vector<fe::ComplexMoments>> moments,
        std::vector<int> layers,
        int max_iters,
        float eps,
        float speed);

    virtual std::string Recognize(fe::ComplexMoments & moments);

    virtual bool Save(std::string filename);

    virtual bool Read(std::string filename);

    bool TrainKNearest(std::map<std::string, std::vector<fe::ComplexMoments>> moments, size_t max_k);
};
