#pragma once

#include <functional>
#include <utility>

#include "ANNIndividual.h"

#ifndef max
#define max(a,b) (((a) > (b)) ? (a) : (b))
#endif

namespace ga
{

    class XORIndividual : public ANNIndividual
    {
    public:

        using spare_fn_t = std::function < std::pair < int, int > (double, double) > ;

        std::shared_ptr < std::vector < std::vector < float > > > inputs;
        std::shared_ptr < std::vector < std::vector < float > > > outputs;
        spare_fn_t spare_fn;

        static spare_fn_t make_simple_spare_fn()
        {
            return [] (double self_err, double other_err)
            {
                return std::make_pair((self_err < other_err) ? 1 : 0,
                                      (self_err < other_err) ? 0 : 1);
            };
        }

        static spare_fn_t make_inv_spare_fn()
        {
            return [] (double self_err, double other_err)
            {
                return std::make_pair(1 + (int)(1. / (1e-3 + self_err)), 1 + (int)(1. / (1e-3 + other_err)));
            };
        }

        static spare_fn_t make_logarithmic_spare_fn()
        {
            return [] (double self_err, double other_err)
            {
                if (self_err < other_err)
                {
                    return std::make_pair(1 + max(-(int)(100 * (other_err - self_err) * log10(self_err)), 0), 0);
                }
                return std::make_pair(0, 1 + max(-(int)(100 * (self_err - other_err) * log10(other_err)), 0));
            };
        }

        static spare_fn_t make_composite_spare_fn()
        {
            auto f1 = make_simple_spare_fn();
            auto f2 = make_inv_spare_fn();
            auto f3 = make_logarithmic_spare_fn();
            return [f1, f2, f3] (double self_err, double other_err)
            {
                auto r1 = f1(self_err, other_err);
                auto r2 = f2(self_err, other_err);
                auto r3 = f3(self_err, other_err);
                return std::make_pair < int, int > (
                    r1.first * 1000 + r2.first + r3.first,
                    r1.second * 1000 + r2.second + r3.second
                );
            };
        }

        XORIndividual(std::shared_ptr < std::vector < std::vector < float > > > inputs,
                      std::shared_ptr < std::vector < std::vector < float > > > outputs,
                      spare_fn_t spare_fn)
            : ANNIndividual({ 2, 10, 10, 1 }, 50)
            , inputs(inputs)
            , outputs(outputs)
            , spare_fn(spare_fn)
        {
        }

        XORIndividual(const XORIndividual & ref)
            : ANNIndividual(ref)
            , inputs(ref.inputs)
            , outputs(ref.outputs)
            , spare_fn(ref.spare_fn)
        {
        }

        std::pair<int, int> Spare(pIIndividual individual) override;

        pIIndividual Clone() override
        {
            return std::make_shared < XORIndividual > (*this);
        }
    };

}