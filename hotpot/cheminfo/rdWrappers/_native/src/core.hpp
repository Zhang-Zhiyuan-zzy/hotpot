#pragma once

#include <hotpot/rdWrappers/c_api.h>

#include <GraphMol/RWMol.h>

#include <cstdint>
#include <compare>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>


namespace hotpot::rdwrappers::core {


inline constexpr const char* kHotpotVersion = "0.5.4.0";
inline constexpr const char* kRDKitVersion = "2026.09.1";
inline constexpr const char* kRDKitCommit =
    "fece8caa860bdf6c9c82bdb9253f22a0cf87150c";
inline constexpr const char* kEngine = "RDKit ETKDGv3";


class UnsupportedInput final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};


class AbiMismatch final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};


struct Options {
    std::uint64_t conformer_count;
    std::optional<std::int64_t> seed;
    std::optional<double> prune_rms_threshold;
    bool use_random_coordinates;
};


struct Result {
    hotpot_rd_generation_status status = HOTPOT_RD_GENERATION_FAILED;
    std::uint64_t atom_count = 0;
    std::vector<double> coordinates;
    std::vector<hotpot_rd_conformer_record_v1> records;
    std::string engine{kEngine};
    std::string engine_version{kRDKitVersion};
    std::string diagnostic;
    bool has_diagnostic = false;
};


Options validate_options(const hotpot_rd_options_v1& options);
void validate_molecule_view(const hotpot_rd_molecule_view_v1& molecule);
void validate_constraints_view(
    const hotpot_rd_constraints_view_v1& constraints,
    std::uint64_t atom_count
);

std::unique_ptr<RDKit::RWMol> build_molecule(
    const hotpot_rd_molecule_view_v1& molecule
);

Result generate(
    const Options& options,
    const hotpot_rd_molecule_view_v1& molecule,
    const hotpot_rd_constraints_view_v1* constraints
);


}  // namespace hotpot::rdwrappers::core
