#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

namespace fe {
	/*��������� ����� ������� ����� ����������������� � ������������ ������� ��������*/
	__interface IBlobProcessor
	{
	public:
		/**����� ������� ������� �� �����������
		 * @param image - ����������� ��� ������ ������� ��������.
		 *				 ������ ����� ��� CV_8UC1
		 * @param blobs - ����� ��� ������ �������������������� ������� ��������
		 */
		virtual void DetectBlobs(cv::Mat image, std::vector<cv::Mat> & blobs) = 0;

		/**�������� ������ ������� �������� � ������� ��������.
		 * @param blobs - ������� �������
		 * @param normilized_blobs - ����� ��� ������ ������� �������� ������� �������.
		 * @param side - ������� �������� �� ������� ����� ���������� ��������������� ������� �������.
		 */
		virtual void NormalizeBlobs(
			std::vector<cv::Mat> & blobs,
			std::vector<cv::Mat> & normalized_blobs,
			int side
		);

		/**�������� �������� ������������� ����������� ������� ��������.
		 * @return ��������
		 */
		virtual std::string GetType();
	};
};