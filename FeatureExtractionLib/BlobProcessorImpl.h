#pragma once

#include "IBlobProcessor.h"

namespace fe
{
	class BlobProcessorImpl : public IBlobProcessor
	{

    public:

        virtual ~BlobProcessorImpl()
        {
        }

	public:

		virtual void DetectBlobs(cv::Mat image, std::vector<cv::Mat> & blobs);

		virtual void NormalizeBlobs(
			std::vector<cv::Mat> & blobs,
			std::vector<cv::Mat> & normalized_blobs,
			int side
		);

		virtual std::string GetType();
	};
};
