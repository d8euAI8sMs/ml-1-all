#include "BlobProcessorImpl.h"

#define M_PI 3.1415926535897932

std::string fe::BlobProcessorImpl::GetType()
{
    return "Simple Blob Processor by Alexander Vasilevsky";
}

void fe::BlobProcessorImpl::DetectBlobs(cv::Mat image, std::vector<cv::Mat> & blobs)
{
    std::vector < std::vector < cv::Point > > contours;
    std::vector < cv::Vec4i > hierarchy;
    cv::findContours(image, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
    cv::Point2f contour_center; float contour_radius;
    for(int i = 0, j = 0; i >= 0; i = hierarchy[i][0])
    {
        cv::minEnclosingCircle(contours[i], contour_center, contour_radius);
        blobs.emplace_back((int)std::ceil(contour_radius * 2), (int)std::ceil(contour_radius * 2), CV_8UC1);
        blobs[j].setTo(cv::Scalar(0));
        cv::drawContours(blobs[j], contours, i, cv::Scalar(255), CV_FILLED, 8, hierarchy, 100, -contour_center + cv::Point2f(contour_radius, contour_radius));
        ++j;
    }
}

static void GetMoments(cv::Mat & blob,
                       cv::Point2f & mass_center,
                       double & effective_radius,
                       double & theta)
{
    double normalized_color;
    double radius_mean = 0;
    double radius_dispersion = 0;
    double m00 = 0, m10 = 0, m01 = 0;
    double m11 = 0, m20 = 0, m02 = 0;
    double x, y;
    for (size_t r = 0; r < blob.rows; ++r)
    {
        for (size_t c = 0; c < blob.cols; ++c)
        {
            normalized_color = blob.at<uchar>(c, r) / 255.0;
            m00 += normalized_color;
            m10 += c * normalized_color;
            m01 += r * normalized_color;
        }
    }
    mass_center = cv::Point2f(m10 / m00, m01 / m00);
    for (size_t r = 0; r < blob.rows; ++r)
    {
        for (size_t c = 0; c < blob.cols; ++c)
        {
            normalized_color = blob.at<uchar>(c, r) / 255.0;
            x = c - mass_center.x; y = r - mass_center.y;
            radius_mean += std::sqrt(x * x + y * y) * normalized_color;
            radius_dispersion += (x * x + y * y) * normalized_color;
            m20 += x * x * normalized_color;
            m11 += x * y * normalized_color;
            m02 += y * y * normalized_color;
        }
    }
    radius_mean /= m00;
    radius_dispersion -= radius_mean * radius_mean * m00;
    radius_dispersion /= m00;
    effective_radius = radius_mean + 3.0 /* 1..3 */ * std::sqrt(radius_dispersion);
    theta = std::atan2(2 * m11, m20 - m02) / 2.0;
}

void fe::BlobProcessorImpl::NormalizeBlobs
(
    std::vector < cv::Mat > & blobs,
    std::vector < cv::Mat > & normalized_blobs,
    int side
)
{
    normalized_blobs.resize(blobs.size());
    cv::Point2f mass_center;
    double effective_radius;
    double theta;
    int start_row, end_row;
    int start_col, end_col;
    for (size_t i = 0; i < blobs.size(); ++i)
    {
        cv::Mat pre_scale;
        cv::Mat blured;
        cv::Mat scaled;

        GetMoments(blobs[i], mass_center, effective_radius, theta);

        pre_scale = cv::Mat::zeros(std::ceil(effective_radius) * 2,
                                   std::ceil(effective_radius) * 2,
                                   CV_8UC1);

        blobs[i].copyTo
        (
            pre_scale.colRange(effective_radius - mass_center.x, effective_radius - mass_center.x + blobs[i].cols)
                     .rowRange(effective_radius - mass_center.y, effective_radius - mass_center.y + blobs[i].rows)
        );

        // blur the image in case of downscale to filter out higher harmonics
        if (side < effective_radius * 2)
        {
            cv::GaussianBlur(pre_scale, blured, cv::Size(), 1.0 / 2.0 * effective_radius * effective_radius / (side / 2.0) / (side / 2.0));
            cv::resize(blured, scaled, cv::Size(side, side));
        }
        else
        {
            cv::resize(pre_scale, scaled, cv::Size(side, side));
        }

        // rotate the image
        if (std::abs(theta) > (M_PI / 100.0))
        {
            cv::Mat rot_mat = cv::getRotationMatrix2D(cv::Point2f(side / 2.0, side / 2.0), theta, 1.0);
            cv::warpAffine(scaled, normalized_blobs[i], rot_mat, cv::Size(side, side));
        }
        else // do nothing if theta is too small
        {
            normalized_blobs[i] = scaled;
        }
    }
}
