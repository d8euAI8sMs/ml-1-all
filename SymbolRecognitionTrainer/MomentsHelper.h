#pragma once
#include "FeatureExtraction.h"

/************************************************************************/
/* �������� ������� ����� � ���������                                   */
/************************************************************************/
/*
	����� � ��������� ����� ��������� ���������:
	� ����� ����� ����� �������� � ���������. � ������ ������� ����� ��������� ����������
	������ � ���� �� ������� � ������� *.png � ���� "value.txt" � ��������� ��������� �������.
*/

/**�������� ��� ������ � ���������.*/
class MomentsHelper
{
public:
	/**����������� �� ���������.*/
	MomentsHelper();
	/**���������� �� ���������.*/
	virtual ~MomentsHelper();

	/**������������� ������� �� ��������.
	 * @param path - ���� �� ����� � ���������.
	 * @param blob_processor - ���������� ������� ��������, ������� ����� �������������� ��� ��������� ��������.
	 * @param poly_manager - �������� ���������, ������� ����� �������������� ��� ��������� ��������.
	 * @param res - ����� ��� ������ ��������������� ��������. ��� ������������� ������.
	 *	���� - �������� ������� (�������� "5")
	 *  �������� - ����� ���������� ������� �������� ����� �������.
	 * @return true - ������� ������� �������������, false - ������� �� �������������.
	 */
	static bool GenerateMoments(
		std::string path,
		std::shared_ptr<fe::IBlobProcessor> blob_processor,
		std::shared_ptr<fe::PolynomialManager> poly_manager,
		std::map< std::string, std::vector<fe::ComplexMoments> > & res
		);

	/**�������� ���� �� ���� �������� � ���������� ������������ ��������, 
	 * ������������� � ����� � ���������.
	 * @param base_path - ���� �� ����� � ���������.
	 * @param paths - ����� ��� ������ ��������� �����.
	 * @return true - ���� ������� �������, false - ���� �� �������.
	 */
	static bool GetSamplePaths(
		std::string base_path,
		std::vector<std::string> & paths
		);

	/**���������� ���� ����������� � ��������. �� ����������� ������ �������������� ������ ���� ������.
	 * @param image_path - ���� � ��� ����� � ���������.
	 * @param blob_processor - ���������� ������� �������� ����������� ���� ���������.
	 * @param poly_manager - �������� ���������, ����������� ��� ����������.
	 * @param res - ����� ��� ������ ����������.
	 */
	static void ProcessOneImage(
		std::string image_path,
		std::shared_ptr<fe::IBlobProcessor> blob_processor,
		std::shared_ptr<fe::PolynomialManager> poly_manager,
		fe::ComplexMoments & res
		);

	/**��������� ����������� ������ �� ������ � ���������� � ��������� �������.
	 * ��� ������ ������� ����������, ����� ���������� ��� ������ ������������ � ���� �����.
	 * @param labeled_data_path - ���� �� ����������� ������.
	 * @param ground_data_path - ���� ��� ������ ��������� ������.
	 * @param test_data_path - ���� ��� ������ ��������� ������.
	 * @param percent - ������� ����������� ������, ������� ����� ���������� � ��������� ������.
	 *	��������� ������ ����� ����������� � ����� ��� �������� ������.
	 * @return true - ������ ����������� �������, false - ������ �� �����������.
	 */
	static bool DistributeData(
		std::string labeled_data_path,
		std::string ground_data_path,
		std::string test_data_path,
		double percent
		);

	/**��������� �������.
	 * @param filename - ��� ����� ��� ����������.
	 * @param moments - ������� ��� ����������. ��� ������������� ������.
	 *	���� - �������� ������� (�������� "5")
	 *  �������� - ����� ���������� ������� �������� ����� �������.
	 * @return - true - ������� ���������, false - ������� �� ���������.
	 */
	static bool SaveMoments(
		std::string filename,
		std::map< std::string, std::vector<fe::ComplexMoments> > & moments
		);

	/**������� ������� �� �����.
	 * @param filename - ��� ����� ��� ����������.
	 * @param moment - ����� ��� ������ ��������� ��������. ��� ������������� ������.
	 *	���� - �������� ������� (�������� "5")
	 *  �������� - ����� ���������� ������� �������� ����� �������.
	 * @return true - ������� ������� �������, ������� �� �������.
	 */
	static bool ReadMoments(
		std::string filename,
		std::map< std::string, std::vector<fe::ComplexMoments> > & moments
		);
};

