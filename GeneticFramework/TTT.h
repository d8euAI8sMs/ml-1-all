#pragma once

#include <vector>

namespace ga
{

    enum class tic_tac : int
    {
        X = -1,
        O = 1,
        Z = 0
    };

    static inline float ToFloat(tic_tac t)
    {
        return static_cast < float > (t);
    }

    static inline tic_tac ToTicTac(float t)
    {
        return static_cast < tic_tac > (static_cast < int > (t));
    }

    struct board
    {
        std::vector < float > cells;
        size_t n;

        board(size_t n)
            : cells(n * n + 1, ToFloat(tic_tac::Z))
            , n(n)
        {
        }
    };

    static inline tic_tac GetTTTWinner(const board & b)
    {
        tic_tac winner;

        // rows

        for (size_t i = 0; i < b.n; ++i)
        {
            winner = ToTicTac(b.cells[i + b.n * 0]);
            for (size_t j = 0; j < b.n; ++j)
            {
                if (winner != ToTicTac(b.cells[i + b.n * j]))
                {
                    winner = tic_tac::Z;
                    break;
                }
            }
            if (winner != tic_tac::Z) return winner;
        }

        // cols

        for (size_t i = 0; i < b.n; ++i)
        {
            winner = ToTicTac(b.cells[i * b.n + 0]);
            for (size_t j = 0; j < b.n; ++j)
            {
                if (winner != ToTicTac(b.cells[i * b.n + j]))
                {
                    winner = tic_tac::Z;
                    break;
                }
            }
            if (winner != tic_tac::Z) return winner;
        }

        // diag

        winner = ToTicTac(b.cells[0]);
        for (size_t i = 0; i < b.n; ++i)
        {
            if (winner != ToTicTac(b.cells[i * b.n + i]))
            {
                winner = tic_tac::Z;
                break;
            }
        }
        if (winner != tic_tac::Z) return winner;

        winner = ToTicTac(b.cells[b.n - 1]);
        for (size_t i = 0; i < b.n; ++i)
        {
            if (winner != ToTicTac(b.cells[i * b.n + (b.n - 1 - i)]))
            {
                winner = tic_tac::Z;
                break;
            }
        }
        if (winner != tic_tac::Z) return winner;

        return tic_tac::Z;
    }
}
