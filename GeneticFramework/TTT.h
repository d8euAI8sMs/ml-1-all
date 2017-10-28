#pragma once

#include <vector>
#include <functional>

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

    using player_fn = std::function < bool (board &, tic_tac) >;

    /**
     *  Returns: winner
     */
    static inline tic_tac PlayTTT(board & b,
                                  tic_tac start_from,
                                  const player_fn & x,
                                  const player_fn & o)
    {
        tic_tac current_player = start_from;
        tic_tac winner;
        bool can_continue;

        do
        {
            b.cells[b.cells.size() - 1] = ToFloat(start_from);
            if (current_player == tic_tac::X)
            {
                can_continue = x(b, current_player);
                current_player = tic_tac::O;
            }
            else
            {
                can_continue = o(b, current_player);
                current_player = tic_tac::X;
            }
            if (can_continue)
            {
                winner = GetTTTWinner(b);
            }
            else
            {
                winner = tic_tac::Z;
            }
        } while (can_continue && (winner == tic_tac::Z));

        return winner;
    }
}
