#include <iostream>
#include <string>
#include "GeneticAlgorithm.h"
#include "XORIndividual.h"
#include "TTTIndividual.h"
#include "Util.h"

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

void print_ttt_board(ga::board b)
{
    for (size_t i = 0; i < b.n; ++i)
    {
        cout << "| ";
        for (size_t j = 0; j < b.n; ++j)
        {
            tic_tac ij = ToTicTac(b.cells[i * b.n + j]);
            cout <<
            (
                (ij == tic_tac::Z)
                ? " "
                : ((ij == tic_tac::O)
                    ? "O"
                    : "X")
            ) << " | ";
        }
        cout << std::endl;
    }
    cout << std::endl;
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
    std::cout << "[input] Spare function name: [bin/log/inv/sum] ";
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
    else if ("sum" == spare_fn_name)
    {
        spare_fn = XORIndividual::make_composite_spare_fn();
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
    pEpoch new_epoch;
    do
    {
        err = 0;

        a.epoch->EpochBattle();
        new_epoch = a.Selection(2.0 / a.epoch->population.size() * 100, 5, 70);

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

        a.epoch = new_epoch;

        if (RandomBool(0.01))
        {
            a.epoch->population.emplace_back(0, std::make_shared < XORIndividual > (inputs, outputs, spare_fn));
        }
    } while (err > eps * eps);
}

void ttt_demo()
{
    size_t initial_size;
    std::cout << "[input] Initial population size: [uint] ";
    std::cin >> initial_size;
    size_t max_epoch;
    std::cout << "[input] Max epoch: [uint] ";
    std::cin >> max_epoch;
    size_t board_size;
    std::cout << "[input] Board size: [uint] ";
    std::cin >> board_size;

    GeneticAlgorithm a;
    a.epoch = std::make_shared < Epoch > ();
    for (size_t i = 0; i < initial_size; ++i)
    {
        a.epoch->population.emplace_back(0, std::make_shared < TTTIndividual > (board_size));
    }

    size_t epoch = 0;
    pEpoch new_epoch;
    do
    {
        a.epoch->EpochBattle();
        new_epoch = a.Selection(2.0 / a.epoch->population.size() * 100, 5, 70);

        ++epoch;

        std::cout << "======== Epoch " << epoch << " ========" << std::endl;

        for each (auto &p in a.epoch->population)
        {
            std::cout << p.first << " ";
        }
        std::cout << std::endl;

        a.epoch = new_epoch;

        if (RandomBool(0.01))
        {
            a.epoch->population.emplace_back(0, std::make_shared < TTTIndividual > (board_size));
        }
    } while (epoch < max_epoch);

    bool new_game = true;
    while(new_game)
    {
        board b(board_size);
        tic_tac winner;

        print_ttt_board(b);

        player_fn user_player = [] (board & _b, tic_tac _who)
        {
            size_t cell_i, cell_j;
            cin >> cell_i >> cell_j;
            _b.cells[(cell_i - 1) * _b.n + (cell_j - 1)] = ToFloat(_who);
            print_ttt_board(_b);
            return true;
        };

        TTTIndividual & other = * dynamic_cast < TTTIndividual * > (a.epoch->population.front().second.get());

        player_fn other_player = [&other] (board & _b, tic_tac _who)
        {
            if (other.player(_b, _who))
            {
                print_ttt_board(_b);
                return true;
            }
            return false;
        };

        winner = PlayTTT(b, tic_tac::X, user_player, other_player);

        if (winner == tic_tac::X)
        {
            std::cout << "You win!" << std::endl;
        }
        else if (winner == tic_tac::O)
        {
            std::cout << "You loose!" << std::endl;
        }
        else
        {
            std::cout << "Well, let's try again..." << std::endl;
        }
        std::cout << "[input] Try again? [1/0] ";
        std::cin >> new_game;
        std::cout << std::endl;
    }
}

int main()
{
	cout << "Hello!" << endl;

    xor_demo();

	return 0;
}
