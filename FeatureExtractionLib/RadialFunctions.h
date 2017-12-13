#pragma once
#include <opencv2/opencv.hpp>

namespace rf {
	class RadialFunctions {
	protected:
		static int walsh_n_max;
		static cv::Mat walsh_matrix;
		static cv::Mat WalshGenerator(cv::Mat walsh, int n);
		static double Legendre(double x, int n);
	public:
		/*���������� ����� �������� ������� �������� n,m
		 * @param rad - ������ �� ������� ����������� ��������, 0 <= rad <= 1.
		 * @param n - ���������� ������� ��������, n > 0.
		 * @param m - ������� ������� ��������, m > 0, n-m ������ ���� ������.
		 * @return �������� �������� � ����� rad.
		 */
		static double Zernike(double rad, int n, int m);

		/**������� ����� � ������� n � ������ n_max.
		 * @param rad - ������ �� ������� ����������� ��������, 0 <= rad <= 1.
		 * @param n - ����� �������.
		 * @param n_max - ���������� ��������� � ������, ������ ���� �������� ������.
		 * @return �������� �������� � ����� rad.
		 */
		static double Walsh(double rad, int n, int n_max);

		/**��������� �������� ���������� �������� ������
		 * @param rad - ������ �� ������� ����������� ��������, 0 <= rad <= 1.
		 * @param n - ������� �������� ������.
		 * @return �������� �������� � ����� rad.
		 */
		static double ShiftedLegendre(double rad, int n);

		/**��������� �������� ���������� �������� ��������
		 * @param rad - ������ �� ������� ����������� ��������, 0 <= rad <= 1.
		 * @param n - ������� �������� ��������.
		 * @return �������� �������� � ����� rad.
		 */
		static double ShiftedChebyshev(double rad, int n);

		virtual ~RadialFunctions();
	};
}