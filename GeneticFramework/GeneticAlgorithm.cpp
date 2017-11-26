#include <algorithm>
#include "GeneticAlgorithm.h"
#include "Util.h"

using namespace ga;
using namespace std;

ga::GeneticAlgorithm::GeneticAlgorithm()
{
}


ga::GeneticAlgorithm::~GeneticAlgorithm()
{
}

static pIIndividual Select(ga::pEpoch epoch, double total_points)
{
    return RandomSelect
    (
        epoch->population.begin(),
        epoch->population.end(),
        [total_points] (const decltype(epoch->population.begin()) & iter) { return (double)iter->first / total_points; }
    )->second;
}

ga::pEpoch ga::GeneticAlgorithm::Selection(double unchange_perc, double mutation_perc, double crossover_perc)
{
	// Сортировка особей по набранным очкам.
	sort(epoch->population.begin(), epoch->population.end(),
		[](std::pair<int, pIIndividual> a, std::pair<int, pIIndividual> b)
	{
		return a.first > b.first;
	});

    double point_dispersion = 0;
    double total_points = 0;
    for each (auto & p in epoch->population)
    {
        point_dispersion += (double)p.first * (double)p.first;
        total_points += p.first;
    }
    point_dispersion /= epoch->population.size();
    point_dispersion -= (total_points / epoch->population.size()) * (total_points / epoch->population.size());

    // if all individuals are 'same', increase mutation rate
    if (point_dispersion < 1e-6) mutation_perc *= 5;

    auto new_epoch = std::make_shared < Epoch > ();

    size_t elite = std::ceil(epoch->population.size() * unchange_perc / 100);

    do
    {
        pIIndividual x = Select(epoch, total_points)
                   , y = nullptr;
        if (elite)
        {
            if (std::find(new_epoch->population.begin(),
                          new_epoch->population.end(),
                          std::make_pair(0, x)) != new_epoch->population.end())
            {
                continue;
            }
            new_epoch->population.emplace_back(0, std::move(x));
            --elite;
        }
        else if (RandomBool(crossover_perc / 100))
        {
            do
            {
                y = Select(epoch, total_points);
            } while (x == y);
            y = x->Crossover(std::move(y));
            if (RandomBool(mutation_perc / 100)) y = y->Mutation();
            new_epoch->population.emplace_back(0, std::move(y));
        }
    } while (new_epoch->population.size() < epoch->population.size());

    return std::move(new_epoch);
}
