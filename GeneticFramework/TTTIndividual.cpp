#include <algorithm>
#include <cmath>

#include "TTTIndividual.h"

std::pair<int, int> ga::TTTIndividual::Spare(pIIndividual individual)
{
    if (this != individual.get()) return { };

    TTTIndividual & other = * (dynamic_cast < TTTIndividual * > (individual.get()));

    player_fn random_player = &RandomPlayer;

    size_t win = 0,     loose = 0,     draw = 0,
           rnd_win = 0, rnd_loose = 0, rnd_draw = 0,
           games;

    games = this->n * this->n * 30;

    for (size_t i = 0; i < games; ++i)
    {
        board b(this->n);

        b.cells[rand() % (this->n * this->n)] = ToFloat(tic_tac::X);

        tic_tac winner = PlayTTT(b, tic_tac::O, this->player, random_player);

        if      (winner == tic_tac::Z) ++rnd_draw;
        else if (winner == tic_tac::X) ++rnd_win;
        else if (winner == tic_tac::O) ++rnd_loose;
    }

    return std::make_pair
    (
        (int) ((games - rnd_loose) / (float) games * 1000),
        0
    );
}
