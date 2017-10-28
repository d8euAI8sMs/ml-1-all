#pragma once

#include <functional>
#include <utility>
#include <cmath>

#include "ANNIndividual.h"
#include "TTT.h"

namespace ga
{

    class TTTIndividual : public ANNIndividual
    {

    public:

        const size_t    n;

    public:

        TTTIndividual(int n)
            : ANNIndividual({ n * n + 1, n * n * 2, n * n * 2, 1 }, 4)
            , n((size_t)n)
        {
        }

        TTTIndividual(const TTTIndividual & ref)
            : ANNIndividual(ref)
            , n(ref.n)
        {
        }

        std::pair<int, int> Spare(pIIndividual individual) override;

        pIIndividual Clone() override
        {
            return std::make_shared < TTTIndividual > (*this);
        }
    };
}
