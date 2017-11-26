#pragma once

#include <vector>
#include <functional>
#include <algorithm>

namespace ga
{

    enum class tic_tac : int
    {
        X = 3,
        O = 2,
        Z = 1
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

    static inline std::vector < float > GetInvariant(board & b)
    {
        std::vector < float > r, t(b.n * b.n);
        r = b.cells;

        // apply reflections
        for (size_t i = 0; i < b.n; ++i)
        {
            for (size_t j = 0; j < b.n; ++j)
            {
                t[i * b.n + j] += b.cells[i * b.n + (b.n - j - 1)]
                                + b.cells[(b.n - i - 1) * b.n + j];
            }
        }

        r = t;

        // apply rotations
        for (size_t i = 0; i < b.n; ++i)
        {
            for (size_t j = 0; j < b.n; ++j)
            {
                r[i * b.n + j] += t[j * b.n + (b.n - i - 1)]
                                + t[(b.n - i - 1) * b.n + (b.n - j - 1)]
                                + t[(b.n - j - 1) * b.n + i];
                r[i * b.n + j] /= 6;
            }
        }

        size_t m = (size_t) std::ceil(b.n / 2.0);

        t.resize(m * m + 1);

        for (size_t i = 0; i < m; ++i)
        {
            for (size_t j = 0; j < m; ++j)
            {
                t[i * m + j] = r[i * b.n + j];
            }
        }

        t[m * m] = b.cells[b.n * b.n];

        return t;
    }

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

    static inline bool RandomPlayer(board & b, tic_tac who)
    {
        std::vector < size_t > positions(b.n * b.n);
        for (size_t i = 0; i < positions.size(); ++i) positions[i] = i;
        for (;;)
        {
            std::random_shuffle(positions.begin(), positions.end());
            bool can_continue = false;
            for (size_t i = 0; i < b.n * b.n; ++i)
            {
                size_t j = positions[i];
                can_continue |= (ToTicTac(b.cells[j]) == tic_tac::Z);
                if (can_continue)
                {
                    b.cells[j] = ToFloat(who);
                    return true;
                }
            }
            if (!can_continue)
            {
                return false;
            }
        }
    }
}
