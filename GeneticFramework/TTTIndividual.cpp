#include <algorithm>
#include <cmath>

#include "TTTIndividual.h"

std::pair<int, int> ga::TTTIndividual::Spare(pIIndividual individual)
{
    if (this == individual.get()) return { };

    TTTIndividual & other = * (dynamic_cast < TTTIndividual * > (individual.get()));

    board b(this->n);

    b.cells[rand() % b.cells.size()] = ToFloat(tic_tac::X);

    tic_tac winner = PlayTTT(b, tic_tac::O, this->player, other.player);

    return ((winner == tic_tac::Z)
                ? std::make_pair(1, 1)
                : ((winner == tic_tac::X)
                    ? std::make_pair(2, 1)
                    : std::make_pair(1, 2)));
}
