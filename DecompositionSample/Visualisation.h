#pragma once
#include <opencv2/opencv.hpp>

/*������� �������� ������ ������� ��� ����������� �������� �����������*/
#define MIDDLE_LEVEL 127
/*������������ �������� ������ ������� ��� ����������� �������� �����������*/
#define MAX_LEVEL 255

/**������� �� ����� ����������� ���� CV_64FC1. ��� ��������� ���� ������� waitKey
 * @param wnd_name - �������� ���� ��� �����������.
 * @param polynomial - ����������� ���� CV_64FC1 ��� �����������
 */
void Show64FC1Mat(std::string wnd_name, cv::Mat mat64fc1);

/**������� �� ����� ������� ������� � �� ��������������� �� ���������� �� �������������� ������.
 * ��� ��������� ���� ������� waitKey
 * @param wnd_name - �������� ���� ��� �����������.
 * @param blob - ������� �������, ������� ������������ �� �������������� ������. ������ ����� ��� CV_8UC1.
 * @param decomposition - �������� ��������������� �� ���������� �� �������������� ������.
 *						  ������ ����� CV_64FC1.
 */
void ShowBlobDecomposition(std::string wnd_name, cv::Mat blob, cv::Mat decomposition);


/**������� �� ����� ����� ���������. ��� ��������� ���� ������� waitKey.
 * @param wnd_name - �������� ��� ���� �����������.
 * @param polynomials - ����� ����������� ��������� ��� �����������. 
 *						������ ������� ����������� ����� �������� ���� CV_64FC1.
 */
void ShowPolynomials(std::string wnd_name, std::vector<std::vector<std::pair<cv::Mat, cv::Mat>>> & polynomials);