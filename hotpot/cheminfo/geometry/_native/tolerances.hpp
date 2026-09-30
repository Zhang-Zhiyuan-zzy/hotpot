#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>


namespace hotpot::geometry {


struct NumericTolerances {
    double absolute_length;
    double relative_length;
    double parameter;
    double machine_epsilon_factor;
    double predicate_guard_factor;
    double planarity_factor;
    double winding_residual;
    double intersection_merge_factor;
    double aabb_padding_factor;

    void validate() const {
        const double values[] = {
            absolute_length,
            relative_length,
            parameter,
            machine_epsilon_factor,
            predicate_guard_factor,
            planarity_factor,
            winding_residual,
            intersection_merge_factor,
            aabb_padding_factor,
        };
        for (const double value : values) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument(
                    "geometry tolerances must be finite"
                );
            }
        }
        if (absolute_length <= 0.0) {
            throw std::invalid_argument(
                "absolute_length must be greater than zero"
            );
        }
        if (relative_length < 0.0) {
            throw std::invalid_argument(
                "relative_length must be non-negative"
            );
        }
        if (predicate_guard_factor <= 1.0) {
            throw std::invalid_argument(
                "predicate_guard_factor must be greater than one"
            );
        }
        if (
            parameter <= 0.0
            || parameter >= 1.0 / (2.0 * predicate_guard_factor)
        ) {
            throw std::invalid_argument(
                "parameter lies outside the supported predicate domain"
            );
        }
        if (machine_epsilon_factor < 1.0) {
            throw std::invalid_argument(
                "machine_epsilon_factor must be at least one"
            );
        }
        if (planarity_factor <= 0.0) {
            throw std::invalid_argument(
                "planarity_factor must be greater than zero"
            );
        }
        if (winding_residual <= 0.0 || winding_residual >= 0.5) {
            throw std::invalid_argument(
                "winding_residual must be between zero and one half"
            );
        }
        if (intersection_merge_factor < 1.0) {
            throw std::invalid_argument(
                "intersection_merge_factor must be at least one"
            );
        }
        if (aabb_padding_factor < predicate_guard_factor) {
            throw std::invalid_argument(
                "aabb_padding_factor must cover predicate_guard_factor"
            );
        }
    }

    double effective_relative_length() const noexcept {
        return std::max(
            relative_length,
            machine_epsilon_factor * std::numeric_limits<double>::epsilon()
        );
    }

    double effective_length(double length_scale) const noexcept {
        return absolute_length + effective_relative_length() * length_scale;
    }
};


inline NumericTolerances default_numeric_tolerances() noexcept {
    return {
        1.0e-8,
        1.0e-10,
        1.0e-10,
        64.0,
        4.0,
        1.0,
        1.0e-10,
        4.0,
        4.0,
    };
}


}  // namespace hotpot::geometry
