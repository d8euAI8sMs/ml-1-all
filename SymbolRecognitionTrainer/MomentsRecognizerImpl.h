#pragma once

#include "MomentsRecognizer.h"

class MomentsRecognizerImpl : public MomentsRecognizer
{

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
};
