#pragma once

#include <cstdlib>

namespace ga
{

    /**
     *  Returns `true` with the given probability.
     */
    static inline bool RandomBool(float probabilityOfTrue)
    {
        return (((float)rand() / RAND_MAX) < probabilityOfTrue);
    }

    /**
     *  summary:
     *
     *      Selects one of the given values according to the
     *      given probabilities.
     *
     *  params:
     *
     *      begin, end
     *
     *          begin- and end-pointing iterators;
     *          `end` might not be dereferenceable;
     *          forward iterators expected
     *
     *      mapper
     *
     *          a functional object, iterator-to-probability
     *          mapper
     *
     *          mapper(const InputIterator &) -> float must be
     *          meaningful for any dereferenceable iterator
     *
     *  returns:
     *
     *      An iterator pointing to the selected value
     */
    template
    <
        typename InputIterator,
        typename ProbabilityMapper
    >
    static inline InputIterator RandomSelect
    (
        InputIterator     begin,
        InputIterator     end,
        ProbabilityMapper mapper
    )
    {
        for (;;)
        {
            InputIterator iter = begin;
            for (; iter != end; ++iter)
            {
                if (RandomBool(mapper(iter)))
                {
                    return std::move(iter);
                }
            }
        }
    }
}
