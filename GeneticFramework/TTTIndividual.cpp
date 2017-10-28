#include <algorithm>
#include <cmath>

#include "TTTIndividual.h"

std::pair<int, int> ga::TTTIndividual::Spare(pIIndividual individual)
{
    if (this == individual.get()) return { };

    size_t n = std::sqrt(configuration[0] - 1);

    board b(configuration[0], ToFloat(tic_tac::Z));

    bool can_continue;

    tic_tac current_player = tic_tac::O;
    tic_tac winner;

    b[rand() % b.size()] = ToFloat(tic_tac::X);

    do
    {
        b[b.size() - 1] = ToFloat(current_player);
        can_continue = false;
        float max_w = 0;
        size_t max_i = 0;
        tic_tac old;
        for (size_t i = 0; i < n * n; ++i)
        {
            if (ToTicTac(b[i]) != tic_tac::Z)
            {
                continue;
            }
            old = ToTicTac(b[i]);
            b[i] = ToFloat(current_player);
            float decision = std::abs
            (
                (current_player == tic_tac::X)
                    ? MakeDecision(b).front()
                    : individual->MakeDecision(b).front()
            );
            if (max_w < decision)
            {
                max_w = decision;
                max_i = i;
            }
            b[i] = ToFloat(old);
            can_continue = true;
        }
        b[max_i] = ToFloat(current_player);
        current_player = ((current_player == tic_tac::X) ? tic_tac::O : tic_tac::X);
    } while (can_continue && ((winner = GetTTTWinner(b, n)) == tic_tac::Z));

    return (can_continue
                ? std::make_pair(1, 1)
                : ((winner == tic_tac::X)
                    ? std::make_pair(2, 1)
                    : std::make_pair(1, 2)));
}
