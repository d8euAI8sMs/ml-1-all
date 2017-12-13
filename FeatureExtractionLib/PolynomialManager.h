#pragma once

#include <opencv2/opencv.hpp>
#include "ComplexMoments.h"

namespace fe {
	/**��������������� ���� ������ ���������. ��� ���������.
	 * ����������� �������� �������� � ���������� �������. ������ ������ - ���������� �������, ������ ������ - ������� �������.
	 * ������� ����������� ����� std::pair - � ���� first �������� �������� ����� ��������, � ���� second ������.
	 * ������ ����� ������������ � cv::Mat ���� CV_64FC1.
	 */
	typedef std::vector<std::vector<std::pair<cv::Mat, cv::Mat>>> OrthoBasis;
	/**����� ���������� �� �������������� � ���������� ~ exp(jm*fi)*/
	class PolynomialManager
	{
	protected:
		/**��������*/
		OrthoBasis polynomials;
	public:
		/**��������� �������� � ��� �� ���������
		 * @param blob - �������� (������� �������), ������ ���� ���� CV_8UC1
		 * @param decomposition - ����� ��� ������ ����������
		 */
		virtual void Decompose(cv::Mat blob, ComplexMoments & decomposition) = 0;

		/**������������ �������� �� ����������.
		 * @param decomposition - ���������� �������� � ���.
		 * @param recovery - ����� ��� ������ ���������������� �����������. 
		 *					 ���������������� ����������� ����� ��� CV_64FC1
		 */
		virtual void Recovery(ComplexMoments & decomposition, cv::Mat & recovery) = 0;

		/**������������������� ����� ������������� ��������� ~ exp(jm*fi)
		 * @param n_max - ������������ ���������� ������� ���������.
		 * @param diameter - ������� ����������, �� ������� ����� ������������� ��������, �������.
		 */
		virtual void InitBasis(int n_max, int diameter) = 0;

		/**�������� �������� ������� ��� ������ � ����������.
		 * @return - �������� �������.
		 */
		virtual std::string GetType() = 0;

		/**�������� ����� ������������� ���������. 
		 * @return ����� ������������� ���������. ������ ������� ����������� std::pair<cv::Mat, cv::Mat>.
		 *		   � ���� first �������� �������� ����� ��������, � ���� second ������. ������ ����� ����� ��� CV_64FC1.
		 */
		virtual OrthoBasis GetBasis();

		virtual ~PolynomialManager();
	};
};

