#pragma once
#include "Epoch.h"

namespace ga
{
	// ������������ ��������.
	class GeneticAlgorithm
	{
	public:
		// �����������.
		GeneticAlgorithm();
		// ����������.
		virtual ~GeneticAlgorithm();

		// ������� �����.
		pEpoch epoch;
		
		/**�������� ����� � ���� �����. �������� � ���������� ����������� �� 0 �� 100.
		 * @param unchange_perc - ������� ���������, ������� �������� � ��������� ����� ��� ���������.
		 * @param mutation_perc - ������� ����� ���������, ������� ����� ���������� ������������ �����.
		 * @param crossover_perc - ������� ����� ���������, ������� ����� ���������� ����� ����� �����������.
		 * @return - ����������� �����.
		 */
		pEpoch Selection(double unchange_perc, double mutation_perc, double crossover_perc);
	};
}



