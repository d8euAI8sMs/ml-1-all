#include "PolynomialManagerImpl.h"
#include "RadialFunctions.h"

#define M_PI 3.1415926535897932

#include <limits>

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

void fe::PolynomialManagerImpl::Decompose(cv::Mat blob, ComplexMoments & decomposition)
{
    size_t basis_mat_count = 0;
    for (size_t i = 0; i < this->polynomials.size(); ++i)
    {
        for (size_t j = 0; j < this->polynomials[i].size(); ++j)
        {
            ++basis_mat_count;
        }
    }
    decomposition.re = cv::Mat(1, basis_mat_count, CV_64FC1);
    decomposition.im = cv::Mat(1, basis_mat_count, CV_64FC1);

    cv::Mat other;
    blob.convertTo(other, CV_64FC1);

    // convert from 0..255 to -1..+1
    other /= 255.0 / 2.0;
    other -= 1.0;

    size_t basis_idx = 0;
    for (size_t i = 0; i < this->polynomials.size(); ++i)
    {
        for (size_t j = 0; j < this->polynomials[i].size(); ++j)
        {
            double base_norm_re = this->polynomials[i][j].first.dot(this->polynomials[i][j].first);
            double base_norm_im = this->polynomials[i][j].second.dot(this->polynomials[i][j].second);
            if (abs(base_norm_re) > std::numeric_limits<double>::epsilon())
            {
                decomposition.re.at<double>(0, basis_idx) = other.dot(this->polynomials[i][j].first) / base_norm_re;
            }
            if (abs(base_norm_im) > std::numeric_limits<double>::epsilon())
            {
                decomposition.im.at<double>(0, basis_idx) = other.dot(this->polynomials[i][j].second) / base_norm_im;
            }
            ++basis_idx;
        }
    }
}

void fe::PolynomialManagerImpl::Recovery(ComplexMoments & decomposition, cv::Mat & recovery)
{
    recovery = cv::Mat(this->polynomials[0][0].first.rows, this->polynomials[0][0].first.cols, CV_64FC1);
    recovery.setTo(cv::Scalar(0));
    size_t basis_idx = 0;
    for (size_t i = 0; i < this->polynomials.size(); ++i)
    {
        for (size_t j = 0; j < this->polynomials[i].size(); ++j)
        {
            recovery += this->polynomials[i][j].first * decomposition.re.at<double>(basis_idx)
                      + this->polynomials[i][j].second * decomposition.im.at<double>(basis_idx);
            ++basis_idx;
        }
    }
}
