#pragma once

#include "c_api.h"

#include <hotpot/native/molecule_buffer_view.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>


namespace hotpot::rdwrappers {


enum class GenerationStatus : std::uint8_t {
    SUCCESS = HOTPOT_RD_GENERATION_SUCCESS,
    PARTIAL = HOTPOT_RD_GENERATION_PARTIAL,
    UNSUPPORTED = HOTPOT_RD_GENERATION_UNSUPPORTED,
    FAILED = HOTPOT_RD_GENERATION_FAILED,
};


struct ETKDGOptions {
    std::size_t conformer_count = 1;
    std::optional<std::int64_t> seed;
    std::optional<double> prune_rms_threshold;
    bool use_random_coordinates = false;
};


using DistanceBounds = std::array<double, 2>;


struct CoordinationEmbeddingConstraintsView {
    native::ArrayView<native::BondIndex> coordination_pairs;
    native::ArrayView<DistanceBounds> metal_donor_bounds;
    native::ArrayView<native::BondIndex> donor_pair_indices;
    native::ArrayView<DistanceBounds> donor_pair_bounds;
};


struct ConformerRecord {
    std::size_t ordinal;
    std::int64_t engine_conformer_id;
};


struct ConformerGenerationResult {
    GenerationStatus status;
    std::size_t conformer_count;
    std::size_t atom_count;
    std::vector<native::Coordinate> coordinates;
    std::vector<ConformerRecord> records;
    std::string engine;
    std::string engine_version;
    std::optional<std::string> diagnostic;
};


struct RuntimeInfo {
    std::string hotpot_version;
    std::string rdkit_version;
    bool native;
};


namespace detail {


inline std::string copy_string(hotpot_rd_string_view_v1 value) {
    return value.data == nullptr
        ? std::string{}
        : std::string(value.data, static_cast<std::size_t>(value.size));
}


inline std::runtime_error status_error(
    const char* operation,
    hotpot_rd_status status
) {
    const char* message = hotpot_rd_status_message(status);
    return std::runtime_error(
        std::string(operation) + ": "
        + (message == nullptr ? "unknown native error" : message)
    );
}


inline hotpot_rd_options_v1 to_c_options(const ETKDGOptions& options) {
    return hotpot_rd_options_v1{
        sizeof(hotpot_rd_options_v1),
        HOTPOT_RD_C_API_VERSION,
        static_cast<std::uint64_t>(options.conformer_count),
        options.seed.value_or(0),
        options.prune_rms_threshold.value_or(0.0),
        options.seed.has_value() ? 1u : 0u,
        options.prune_rms_threshold.has_value() ? 1u : 0u,
        options.use_random_coordinates ? 1u : 0u,
        0u,
    };
}


inline hotpot_rd_molecule_view_v1 to_c_molecule_view(
    native::MoleculeBufferView molecule
) {
    static_assert(sizeof(native::Coordinate) == 3 * sizeof(double));
    static_assert(sizeof(native::BondIndex) == 2 * sizeof(std::int32_t));
    static_assert(sizeof(native::BondKind) == sizeof(std::uint8_t));
    return hotpot_rd_molecule_view_v1{
        sizeof(hotpot_rd_molecule_view_v1),
        HOTPOT_RD_C_API_VERSION,
        molecule.schema_version,
        0u,
        static_cast<std::uint64_t>(molecule.atom_count),
        static_cast<std::uint64_t>(molecule.bond_count),
        molecule.fields,
        molecule.atomic_numbers.data(),
        molecule.formal_charges.data(),
        molecule.partial_charges.data(),
        reinterpret_cast<const double*>(molecule.coordinates.data()),
        molecule.atom_aromatic.data(),
        reinterpret_cast<const std::int32_t*>(molecule.bond_indices.data()),
        molecule.bond_orders.data(),
        reinterpret_cast<const std::uint8_t*>(molecule.bond_kinds.data()),
        molecule.bond_aromatic.data(),
        molecule.unit_cell.data(),
        nullptr,
        0,
        nullptr,
        0,
        nullptr,
        0,
        nullptr,
        0,
        nullptr,
        0,
    };
}


inline hotpot_rd_constraints_view_v1 to_c_constraints_view(
    CoordinationEmbeddingConstraintsView constraints
) {
    static_assert(sizeof(DistanceBounds) == 2 * sizeof(double));
    return hotpot_rd_constraints_view_v1{
        sizeof(hotpot_rd_constraints_view_v1),
        HOTPOT_RD_C_API_VERSION,
        static_cast<std::uint64_t>(constraints.coordination_pairs.size()),
        reinterpret_cast<const std::int32_t*>(
            constraints.coordination_pairs.data()
        ),
        reinterpret_cast<const double*>(constraints.metal_donor_bounds.data()),
        static_cast<std::uint64_t>(constraints.donor_pair_indices.size()),
        reinterpret_cast<const std::int32_t*>(
            constraints.donor_pair_indices.data()
        ),
        reinterpret_cast<const double*>(constraints.donor_pair_bounds.data()),
    };
}


class ResultHandle final {
public:
    explicit ResultHandle(hotpot_rd_result* result) noexcept : result_(result) {}
    ~ResultHandle() { hotpot_rd_result_destroy(result_); }

    ResultHandle(const ResultHandle&) = delete;
    ResultHandle& operator=(const ResultHandle&) = delete;

    hotpot_rd_result* get() const noexcept { return result_; }

private:
    hotpot_rd_result* result_;
};


inline ConformerGenerationResult copy_result(hotpot_rd_result* result) {
    std::uint64_t conformer_count = 0;
    std::uint64_t atom_count = 0;
    const double* coordinates = hotpot_rd_result_coordinates(
        result,
        &conformer_count,
        &atom_count
    );
    const auto coordinate_count = static_cast<std::size_t>(
        conformer_count * atom_count
    );
    std::vector<native::Coordinate> coordinate_values;
    coordinate_values.reserve(coordinate_count);
    for (std::size_t index = 0; index < coordinate_count; ++index) {
        const auto offset = 3 * index;
        coordinate_values.push_back(
            {coordinates[offset], coordinates[offset + 1], coordinates[offset + 2]}
        );
    }

    std::uint64_t record_count = 0;
    const auto* records = hotpot_rd_result_records(result, &record_count);
    std::vector<ConformerRecord> record_values;
    record_values.reserve(static_cast<std::size_t>(record_count));
    for (std::uint64_t index = 0; index < record_count; ++index) {
        record_values.push_back(
            {
                static_cast<std::size_t>(records[index].ordinal),
                records[index].engine_conformer_id,
            }
        );
    }

    const auto diagnostic = hotpot_rd_result_diagnostic(result);
    return ConformerGenerationResult{
        static_cast<GenerationStatus>(hotpot_rd_result_status(result)),
        static_cast<std::size_t>(conformer_count),
        static_cast<std::size_t>(atom_count),
        std::move(coordinate_values),
        std::move(record_values),
        copy_string(hotpot_rd_result_engine(result)),
        copy_string(hotpot_rd_result_engine_version(result)),
        diagnostic.data == nullptr
            ? std::nullopt
            : std::optional<std::string>(copy_string(diagnostic)),
    };
}


}  // namespace detail


class ETKDGSampler final {
public:
    explicit ETKDGSampler(ETKDGOptions options = {}) : sampler_(nullptr) {
        const auto native_options = detail::to_c_options(options);
        const auto status = hotpot_rd_sampler_create(
            &native_options,
            &sampler_
        );
        if (status != HOTPOT_RD_STATUS_OK) {
            throw detail::status_error("hotpot_rd_sampler_create", status);
        }
    }

    ~ETKDGSampler() { hotpot_rd_sampler_destroy(sampler_); }

    ETKDGSampler(const ETKDGSampler&) = delete;
    ETKDGSampler& operator=(const ETKDGSampler&) = delete;

    ETKDGSampler(ETKDGSampler&& other) noexcept : sampler_(other.sampler_) {
        other.sampler_ = nullptr;
    }

    ETKDGSampler& operator=(ETKDGSampler&& other) noexcept {
        if (this != &other) {
            hotpot_rd_sampler_destroy(sampler_);
            sampler_ = other.sampler_;
            other.sampler_ = nullptr;
        }
        return *this;
    }

    [[nodiscard]] ConformerGenerationResult generate(
        native::MoleculeBufferView molecule,
        std::optional<CoordinationEmbeddingConstraintsView> constraints =
            std::nullopt
    ) const {
        native::validate_molecule_view(molecule);
        const auto native_molecule = detail::to_c_molecule_view(molecule);
        const auto native_constraints = constraints.has_value()
            ? std::optional<hotpot_rd_constraints_view_v1>(
                detail::to_c_constraints_view(*constraints)
            )
            : std::nullopt;
        hotpot_rd_result* raw_result = nullptr;
        const auto status = hotpot_rd_generate(
            sampler_,
            &native_molecule,
            native_constraints.has_value() ? &*native_constraints : nullptr,
            &raw_result
        );
        if (status != HOTPOT_RD_STATUS_OK) {
            throw detail::status_error("hotpot_rd_generate", status);
        }
        if (raw_result == nullptr) {
            throw std::runtime_error(
                "hotpot_rd_generate: native component returned no result"
            );
        }
        detail::ResultHandle result(raw_result);
        return detail::copy_result(result.get());
    }

private:
    hotpot_rd_sampler* sampler_;
};


[[nodiscard]] inline ConformerGenerationResult generate_conformers(
    native::MoleculeBufferView molecule,
    ETKDGOptions options = {},
    std::optional<CoordinationEmbeddingConstraintsView> constraints =
        std::nullopt
) {
    return ETKDGSampler(std::move(options)).generate(molecule, constraints);
}


[[nodiscard]] inline RuntimeInfo runtime_info() {
    if (hotpot_rd_runtime_abi_version() != HOTPOT_RD_C_API_VERSION) {
        throw std::runtime_error("hotpot_rdwrappers C API version mismatch");
    }
    return RuntimeInfo{
        detail::copy_string(hotpot_rd_runtime_hotpot_version()),
        detail::copy_string(hotpot_rd_runtime_rdkit_version()),
        true,
    };
}


}  // namespace hotpot::rdwrappers
