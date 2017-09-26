#include "PolynomialManagerImpl.h"
#include "RadialFunctions.h"

#define M_PI 3.1415926535897932

std::string fe::PolynomialManagerImpl::GetType()
{
    return "Walsh Polynomial Manager by Alexander Vasilevsky";
}

void fe::PolynomialManagerImpl::InitBasis(int n_max, int diameter)
{
    double delta = 2. / diameter;
    this->polynomials.resize(n_max + 1);
    for (size_t n = 0; n <= n_max; ++n)
    {
        this->polynomials[n].resize(n + 1);
        for (size_t i = 0; i <= n; ++i)
        {
            this->polynomials[n][i] = std::make_pair(cv::Mat(diameter, diameter, CV_64FC1), cv::Mat(diameter, diameter, CV_64FC1));
            this->polynomials[n][i].first.setTo(cv::Scalar(0));
            this->polynomials[n][i].second.setTo(cv::Scalar(0));
        }
        for (size_t r = 0; r < diameter / 2; ++r)
        {
            double radial = rf::RadialFunctions::Walsh(r * delta, n, n_max);
            size_t rot_count = (size_t) (2 * M_PI * r * delta * diameter) * 2;
            for (size_t th = 0; th < rot_count; ++th)
            {
                double theta = 2 * th * M_PI / rot_count;
                for (size_t i = 0; i <= n; ++i)
                {
                    double sine = std::sin(theta * i) * radial;
                    double cosine = std::cos(theta * i) * radial;
                    double & color_re = this->polynomials[n][i].first.at<double>(r * std::cos(theta) + diameter / 2, r * std::sin(theta) + diameter / 2);
                    color_re = cosine;
                    double & color_im = this->polynomials[n][i].second.at<double>(r * std::cos(theta) + diameter / 2, r * std::sin(theta) + diameter / 2);
                    color_im = sine;
                }
            }
        }
    }
}
