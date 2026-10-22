#include "core.hpp"

#include <DistGeom/BoundsMatrix.h>
#include <DistGeom/TriangleSmooth.h>
#include <GraphMol/Conformer.h>
#include <GraphMol/DistGeomHelpers/BoundsMatrixBuilder.h>
#include <GraphMol/DistGeomHelpers/Embedder.h>

#include <boost/shared_ptr.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>


namespace hotpot::rdwrappers::core {
namespace {


void apply_bounds(
    DistGeom::BoundsMatrix& bounds,
    const std::int32_t* pairs,
    const double* values,
    std::uint64_t pair_count
) {
    for (std::uint64_t index = 0; index < pair_count; ++index) {
        const auto first = static_cast<unsigned int>(pairs[2 * index]);
        const auto second = static_cast<unsigned int>(pairs[2 * index + 1]);
        bounds.setLowerBound(first, second, values[2 * index]);
        bounds.setUpperBound(first, second, values[2 * index + 1]);
    }
}


void validate_coordination_edges(
    const RDKit::RWMol& molecule,
    const hotpot_rd_constraints_view_v1& constraints
) {
    for (std::uint64_t index = 0;
         index < constraints.coordination_pair_count;
         ++index) {
        const auto donor = static_cast<unsigned int>(
            constraints.coordination_pairs[2 * index]
        );
        const auto metal = static_cast<unsigned int>(
            constraints.coordination_pairs[2 * index + 1]
        );
        const auto* bond = molecule.getBondBetweenAtoms(donor, metal);
        if (bond == nullptr || bond->getBondType() != RDKit::Bond::DATIVE
            || bond->getBeginAtomIdx() != donor
            || bond->getEndAtomIdx() != metal) {
            throw std::invalid_argument(
                "coordination pair is not a donor-to-metal dative bond"
            );
        }
    }
}


boost::shared_ptr<const DistGeom::BoundsMatrix> make_bounds_matrix(
    const RDKit::RWMol& molecule,
    const hotpot_rd_constraints_view_v1& constraints,
    const RDKit::DGeomHelpers::EmbedParameters& parameters
) {
    validate_coordination_edges(molecule, constraints);
    auto bounds = boost::shared_ptr<DistGeom::BoundsMatrix>(
        new DistGeom::BoundsMatrix(molecule.getNumAtoms())
    );
    RDKit::DGeomHelpers::initBoundsMat(bounds);
    RDKit::DGeomHelpers::setTopolBounds(molecule, bounds, parameters);
    apply_bounds(
        *bounds,
        constraints.coordination_pairs,
        constraints.metal_donor_bounds,
        constraints.coordination_pair_count
    );
    apply_bounds(
        *bounds,
        constraints.donor_pair_indices,
        constraints.donor_pair_bounds,
        constraints.donor_pair_count
    );
    if (!bounds->checkValid()) {
        throw std::invalid_argument("coordination bounds matrix is inconsistent");
    }
    if (!DistGeom::triangleSmoothBounds(bounds)) {
        throw std::invalid_argument(
            "coordination bounds matrix failed triangle smoothing"
        );
    }
    return bounds;
}


Result failed_result(std::uint64_t atom_count, std::string diagnostic) {
    Result result;
    result.status = HOTPOT_RD_GENERATION_FAILED;
    result.atom_count = atom_count;
    result.diagnostic = std::move(diagnostic);
    result.has_diagnostic = true;
    return result;
}


}  // namespace


Result generate(
    const Options& options,
    const hotpot_rd_molecule_view_v1& source,
    const hotpot_rd_constraints_view_v1* constraints
) {
    auto molecule = build_molecule(source);

    auto parameters = RDKit::DGeomHelpers::ETKDGv3;
    parameters.numThreads = 1;
    parameters.randomSeed = static_cast<int>(options.seed.value_or(-1));
    parameters.pruneRmsThresh = options.prune_rms_threshold.value_or(-1.0);
    parameters.useRandomCoords = options.use_random_coordinates;
    parameters.clearConfs = true;
    if (constraints != nullptr) {
        try {
            parameters.boundsMat = make_bounds_matrix(
                *molecule,
                *constraints,
                parameters
            );
        } catch (const std::bad_alloc&) {
            throw;
        } catch (const std::exception& error) {
            return failed_result(
                source.atom_count,
                std::string("RDKit coordination-bounds construction failed: ")
                    + error.what()
            );
        }
    }

    RDKit::INT_VECT conformer_ids;
    try {
        RDKit::DGeomHelpers::EmbedMultipleConfs(
            *molecule,
            conformer_ids,
            static_cast<unsigned int>(options.conformer_count),
            parameters
        );
    } catch (const std::bad_alloc&) {
        throw;
    } catch (const std::exception& error) {
        return failed_result(
            source.atom_count,
            std::string("RDKit ETKDGv3 embedding failed: ") + error.what()
        );
    }

    if (conformer_ids.empty()) {
        return failed_result(
            source.atom_count,
            "RDKit ETKDGv3 returned no conformers"
        );
    }

    Result result;
    result.status = conformer_ids.size() == options.conformer_count
        ? HOTPOT_RD_GENERATION_SUCCESS
        : HOTPOT_RD_GENERATION_PARTIAL;
    result.atom_count = source.atom_count;
    result.coordinates.reserve(
        conformer_ids.size() * static_cast<std::size_t>(source.atom_count) * 3
    );
    result.records.reserve(conformer_ids.size());

    for (std::size_t ordinal = 0; ordinal < conformer_ids.size(); ++ordinal) {
        const auto conformer_id = conformer_ids[ordinal];
        const auto& conformer = molecule->getConformer(conformer_id);
        if (conformer.getNumAtoms() != source.atom_count) {
            return failed_result(
                source.atom_count,
                "RDKit ETKDGv3 changed the atom count"
            );
        }
        for (std::uint64_t atom = 0; atom < source.atom_count; ++atom) {
            const auto& position = conformer.getAtomPos(
                static_cast<unsigned int>(atom)
            );
            if (!std::isfinite(position.x) || !std::isfinite(position.y)
                || !std::isfinite(position.z)) {
                return failed_result(
                    source.atom_count,
                    "RDKit ETKDGv3 returned non-finite coordinates"
                );
            }
            result.coordinates.push_back(position.x);
            result.coordinates.push_back(position.y);
            result.coordinates.push_back(position.z);
        }
        result.records.push_back(
            hotpot_rd_conformer_record_v1{
                static_cast<std::uint64_t>(ordinal),
                static_cast<std::int64_t>(conformer_id),
            }
        );
    }
    if (result.status == HOTPOT_RD_GENERATION_PARTIAL) {
        result.diagnostic =
            "RDKit ETKDGv3 returned fewer conformers than requested";
        result.has_diagnostic = true;
    }
    return result;
}


}  // namespace hotpot::rdwrappers::core
