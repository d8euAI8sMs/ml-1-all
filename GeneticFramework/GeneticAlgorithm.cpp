#include <algorithm>
#include "GeneticAlgorithm.h"

using namespace ga;
using namespace std;

ga::GeneticAlgorithm::GeneticAlgorithm()
{
}


ga::GeneticAlgorithm::~GeneticAlgorithm()
{
}

static pIIndividual Setect(ga::pEpoch epoch, size_t total_points)
{
    float rnd;
    for (;;)
    {
        for each (auto & p in epoch->population)
        {
            rnd = (float)rand() / RAND_MAX;
            if (rnd * total_points < p.first)
            {
                return p.second;
            }
        }
    }
}

ga::pEpoch ga::GeneticAlgorithm::Selection(double unchange_perc, double mutation_perc, double crossover_perc)
{
	// Сортировка особей по набранным очкам.
	sort(epoch->population.begin(), epoch->population.end(),
		[](std::pair<int, pIIndividual> a, std::pair<int, pIIndividual> b)
	{
		return a.first > b.first;
	});

    int unchange = std::ceil(epoch->population.size() * unchange_perc / 100.0);
    int mutation = std::ceil(epoch->population.size() * mutation_perc / 100.0);
    int crossover = std::ceil(epoch->population.size() * crossover_perc / 100.0);

    int total_points = 0;
    for each (auto & p in epoch->population)
    {
        total_points += p.first;
    }

    auto new_epoch = std::make_shared < Epoch > ();

    new_epoch->population.reserve(unchange + mutation + crossover);

    for (size_t i = 0; i < unchange; ++i)
    {
        new_epoch->population.emplace_back(0, epoch->population[i].second->Clone());
    }

    for (size_t i = 0; i < crossover; ++i)
    {
        pIIndividual x, y;
        do
        {
            x = Setect(epoch, total_points); y = Setect(epoch, total_points);
        } while (x != y);
        new_epoch->population.emplace_back(0, x->Crossover(y));
    }

    for (size_t i = 0; i < mutation; ++i)
    {
        pIIndividual x = Setect(epoch, total_points);
        new_epoch->population.emplace_back(0, x->Mutation());
    }

    return (epoch = std::move(new_epoch));
}

