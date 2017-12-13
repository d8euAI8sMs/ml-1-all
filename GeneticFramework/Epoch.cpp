#include "Epoch.h"


ga::Epoch::Epoch()
{
}


ga::Epoch::~Epoch()
{
}

void ga::Epoch::EpochBattle()
{
	for (int i = 0; i < population.size(); i++) {
		auto & individual1 = population[i];
		for (int j = 0; j < population.size(); j++) {
			auto & individual2 = population[j];
			auto score = individual1.second->Spare(individual2.second);
			individual1.first += score.first;
			individual2.first += score.second;
		}
	}
}