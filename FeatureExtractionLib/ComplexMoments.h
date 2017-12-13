#pragma once
#include <opencv2/opencv.hpp>
#include "ExportMacro.h"

namespace fe {
	/*Структура описывающая комплексные моменты*/
	class ComplexMoments
	{
	public:
		/*Реальные части*/
		cv::Mat re;
		/*Мнимые части*/
		cv::Mat im;
		/*Модули*/
		cv::Mat abs;
		/*Фазы*/
		cv::Mat phase;

		FEATURE_DLL_API ComplexMoments();
		FEATURE_DLL_API virtual ~ComplexMoments();
	};
}

