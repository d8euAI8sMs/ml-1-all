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

    using board = std::vector < float > ;

    static inline tic_tac GetTTTWinner(const board & b, size_t n)
    {
        tic_tac winner;

        // rows

        for (size_t i = 0; i < n; ++i)
        {
            winner = ToTicTac(b[i + n * 0]);
            for (size_t j = 0; j < n; ++j)
            {
                if (winner != ToTicTac(b[i + n * j]))
                {
                    winner = tic_tac::Z;
                    break;
                }
            }
            if (winner != tic_tac::Z) return winner;
        }

        // cols

        for (size_t i = 0; i < n; ++i)
        {
            winner = ToTicTac(b[i * n + 0]);
            for (size_t j = 0; j < n; ++j)
            {
                if (winner != ToTicTac(b[i * n + j]))
                {
                    winner = tic_tac::Z;
                    break;
                }
            }
            if (winner != tic_tac::Z) return winner;
        }

        // diag

        winner = ToTicTac(b[0]);
        for (size_t i = 0; i < n; ++i)
        {
            if (winner != ToTicTac(b[i * n + i]))
            {
                winner = tic_tac::Z;
                break;
            }
        }
        if (winner != tic_tac::Z) return winner;

        winner = ToTicTac(b[n - 1]);
        for (size_t i = 0; i < n; ++i)
        {
            if (winner != ToTicTac(b[i * n + (n - 1 - i)]))
            {
                winner = tic_tac::Z;
                break;
            }
        }
        if (winner != tic_tac::Z) return winner;

        return tic_tac::Z;
    }
}
