#include "BlobProcessorImpl.h"

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

void fe::BlobProcessorImpl::NormalizeBlobs
(
    std::vector < cv::Mat > & blobs,
    std::vector < cv::Mat > & normalized_blobs,
    int side
)
{
    normalized_blobs.resize(blobs.size());
    for (size_t i = 0; i < blobs.size(); ++i)
    {
        cv::resize(blobs[i], normalized_blobs[i], cv::Size(side, side));
    }
}
