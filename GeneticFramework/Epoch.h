#pragma once
#include "IIndividual.h"
namespace ga
{
	// ����� ��������.
	class Epoch
	{
	public:
		// �����������.
		Epoch();
		// ����������.
		virtual ~Epoch();

		/* ���������. ������������ ����� ������ ���.
		 * ������ ������� � ������ ���� - ���� ��������, ������ - ��������� �� �������.
		 */
		std::vector<std::pair<int, pIIndividual>> population;

		/* ���������� ����� ������� � ������.
		 * ��� ������ ���� ��������� ������ ����������� ��������� �� ������ � ��������� ����.
		 */
		void EpochBattle();
	};

	// ��������������� ���� "����� ��������� �� �����".
	typedef std::shared_ptr<Epoch> pEpoch;
}