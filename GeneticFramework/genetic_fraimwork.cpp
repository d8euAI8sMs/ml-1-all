#include <iostream>
#include <string>
#include "GeneticAlgorithm.h"
#include "XORIndividual.h"

using  namespace std;
using namespace ga;

template < typename T >
std::ostream & operator << (std::ostream & out, std::vector < T > v)
{
    out << "{";
    for (size_t i = 0; i < v.size(); ++i)
    {
        out << v[i];
        if ((i + 1) != v.size())
        {
            out << ", ";
        }
    }
    out << "}";
    return out;
}

void xor_demo()
{
    auto inputs = std::make_shared < std::vector < std::vector < float > > > ();
    auto outputs = std::make_shared < std::vector < std::vector < float > > > ();

    ANN::LoadData("..\\data.dat", *inputs, *outputs);

    size_t initial_size;
    std::cout << "[input] Initial population size: [uint] ";
    std::cin >> initial_size;

    std::string spare_fn_name;
    std::cout << "[input] Spare function name: [bin/log/inv] ";
    std::cin >> spare_fn_name;

    XORIndividual::spare_fn_t spare_fn = XORIndividual::make_simple_spare_fn();
    if ("log" == spare_fn_name)
    {
        spare_fn = XORIndividual::make_logarithmic_spare_fn();
    }
    else if ("inv" == spare_fn_name)
    {
        spare_fn = XORIndividual::make_inv_spare_fn();
    }

    GeneticAlgorithm a;
    a.epoch = std::make_shared < Epoch > ();
    for (size_t i = 0; i < initial_size; ++i)
    {
        a.epoch->population.emplace_back(0, std::make_shared < XORIndividual > (inputs, outputs, spare_fn));
    }

    const double eps = 1e-3;
    double err = 0;
    size_t epoch = 0;
    do
    {
        err = 0;

        a.epoch->EpochBattle();
        a.Selection(2.0 / a.epoch->population.size() * 100, 5, 70);

        ++epoch;

        std::cout << "======== Epoch " << epoch << " ========" << std::endl;

        for (size_t i = 0; i < inputs->size(); ++i)
        {
            auto result = a.epoch->population[0].second->MakeDecision((*inputs)[i]);
            for (size_t j = 0; j < (*outputs)[i].size(); ++j)
            {
                err += ((*outputs)[i][j] - result[j]) * ((*outputs)[i][j] - result[j]);
            }
            std::cout << " Input: " << (*inputs)[i]
                      << " Expected: " << (*outputs)[i]
                      << " Actual: " << result << std::endl;
        }

        err /= inputs->size();
        std::cout << " Error: " << std::sqrt(err) << std::endl;
    } while (err > eps * eps);
}

int main()
{
	cout << "Hello!" << endl;

    xor_demo();

	return 0;
}
