#pragma once
#include <memory>
#include <vector>

namespace ga
{
	// ������������� ��������� ��� ����� �� ������������� ���������.
	__interface IIndividual
	{
		/**��������� ������� �����.
		 * @return ������������ �����.*/
		std::shared_ptr<IIndividual> Mutation();

		/**��������� ����������� ������� ����� � ������ ������.
		 * @param individual - ����� � ������� ����� ��������� �����������.
		 * @return �������� ����� ����� �����������.*/
		std::shared_ptr<IIndividual> Crossover(std::shared_ptr<IIndividual> individual);

		/**�������� ������������ ����� ������� � ������ ������.
		 * @param individual - ������ �����.
		 * @return ���� ����. ������ �������� - ���������� ����� ��������� ������� ������.
		 *					  ������ �������� - ���������� �����, ��������� ������ ������.
		 */
		std::pair<int, int> Spare(std::shared_ptr<IIndividual> individual);

		/**������� �������.
		 * � �������� ������������ ����� ���������� ��������� �������, 
		 * �� ����� ������� ������� �������� ������������.
		 * @param input - ������� ������
		 * @return �������� ������.
		 */
		std::vector<float> MakeDecision(std::vector<float> & input);

		/**����������� ������� �����.
		 * @return ����� ������� �����.
		 */
		std::shared_ptr<IIndividual> Clone();
	};

	// ��������������� ���� "����� ��������� �� �������".
	typedef std::shared_ptr<IIndividual> pIIndividual;
}