#include "XORIndividual.h"

std::pair<int, int> ga::XORIndividual::Spare(pIIndividual individual)
{
    if (this == individual.get()) return { };

    double self_err = 0, other_err = 0;
    for (size_t i = 0; i < inputs->size(); ++i)
    {
        auto self_r = this->MakeDecision((*inputs)[i]);
        auto other_r = individual->MakeDecision((*inputs)[i]);
        for (size_t j = 0; j < (*outputs)[i].size(); ++j)
        {
            self_err += ((*outputs)[i][j] - self_r[j]) * ((*outputs)[i][j] - self_r[j]);
            other_err += ((*outputs)[i][j] - other_r[j]) * ((*outputs)[i][j] - other_r[j]);
        }
    }
    self_err /= inputs->size(); other_err /= inputs->size();

    return spare_fn(self_err, other_err);
}