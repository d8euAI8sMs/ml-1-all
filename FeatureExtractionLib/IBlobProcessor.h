#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

namespace fe {
	/*»нтерфейс через который можно взаимодействовать с обработчиком смежных областей*/
	__interface IBlobProcessor
	{
	public:
		/**Ќайти смежные области на изображении
		 * @param image - изображение дл€ поиска смежных областей.
		 *				 должно иметь тип CV_8UC1
		 * @param blobs - буфер дл€ записи неотмасштабированных смежных областей
		 */
		virtual void DetectBlobs(cv::Mat image, std::vector<cv::Mat> & blobs) = 0;

		/**ѕривести размер смежных областей к единому масштабу.
		 * @param blobs - смежные области
		 * @param normilized_blobs - буфер дл€ записи смежных областей единого размера.
		 * @param side - сторона квадрата на котором будет отрисована нормализованна€ смежна€ область.
		 */
		virtual void NormalizeBlobs(
			std::vector<cv::Mat> & blobs,
			std::vector<cv::Mat> & normalized_blobs,
			int side
		);

		/**ѕолучить описание используемого обработчика смежных областей.
		 * @return описание
		 */
		virtual std::string GetType();
	};
};