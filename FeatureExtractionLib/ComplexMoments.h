#pragma once
#include <opencv2/opencv.hpp>
#include "ExportMacro.h"

namespace fe {
	/*��������� ����������� ����������� �������*/
	class ComplexMoments
	{
	public:
		/*�������� �����*/
		cv::Mat re;
		/*������ �����*/
		cv::Mat im;
		/*������*/
		cv::Mat abs;
		/*����*/
		cv::Mat phase;

		FEATURE_DLL_API ComplexMoments();
		FEATURE_DLL_API virtual ~ComplexMoments();
	};
}

