#pragma once

#include <functional>
#include <utility>
#include <limits>
#include <cmath>

#include "ANNIndividual.h"
#include "TTT.h"

namespace ga
{

    class TTTIndividual : public ANNIndividual
    {

    public:

              player_fn player;
        const size_t    n;

    public:

        TTTIndividual(int n)
            : ANNIndividual({
                (int) (std::ceil(n / 2.0) * std::ceil(n / 2.0)) + 1,
                (int) (std::ceil(n / 2.0) * std::ceil(n / 2.0)) * 2,
                (int) (std::ceil(n / 2.0) * std::ceil(n / 2.0)) * 2,
                1}, 0.1)
            , n((size_t)n)
        {
            player = std::bind(&TTTIndividual::MakeStepOnBoard,
                               this, std::placeholders::_1, std::placeholders::_2);
        }

        TTTIndividual(const TTTIndividual & ref)
            : ANNIndividual(ref)
            , n(ref.n)
        {
            player = std::bind(&TTTIndividual::MakeStepOnBoard,
                               this, std::placeholders::_1, std::placeholders::_2);
        }

        std::pair<int, int> Spare(pIIndividual individual) override;

        pIIndividual Clone() override
        {
            return std::make_shared < TTTIndividual > (*this);
        }

        bool MakeStepOnBoard(board & b, tic_tac play_as)
        {
            bool can_continue = false;
            tic_tac old;
            float max_w = std::numeric_limits < float > :: lowest();
            size_t max_i = 0;
            for (size_t i = 0; i < n * n; ++i)
            {
                if (ToTicTac(b.cells[i]) != tic_tac::Z)
                {
                    continue;
                }
                old = ToTicTac(b.cells[i]);
                b.cells[i] = ToFloat(play_as);
                auto inv = GetInvariant(b);
                float decision = MakeDecision(inv).front();
                if (max_w < decision)
                {
                    max_w = decision;
                    max_i = i;
                }
                b.cells[i] = ToFloat(old);
                can_continue = true;
            }
            b.cells[max_i] = ToFloat(play_as);
            return can_continue;
        }
    };
}
