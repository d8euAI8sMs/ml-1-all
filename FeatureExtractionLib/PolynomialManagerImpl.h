#pragma once

#include "PolynomialManager.h"

namespace fe
{

    class PolynomialManagerImpl : public PolynomialManager
    {

    public:

        PolynomialManagerImpl()
        {
        }

		virtual ~PolynomialManagerImpl()
        {
        }

    public:

        // PolynomialManager

		virtual void Decompose(cv::Mat blob, ComplexMoments & decomposition);

        virtual void Recovery(ComplexMoments & decomposition, cv::Mat & recovery);

		virtual void InitBasis(int n_max, int diameter);

		virtual std::string GetType();
    };
}
