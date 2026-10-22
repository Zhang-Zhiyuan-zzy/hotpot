#include "core.hpp"

#include <RDGeneral/versions.h>

#include <cstdint>
#include <exception>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>


struct hotpot_rd_sampler {
    hotpot::rdwrappers::core::Options options;
};


struct hotpot_rd_result {
    hotpot::rdwrappers::core::Result value;
};


namespace {


hotpot_rd_string_view_v1 string_view(const std::string& value) noexcept {
    return hotpot_rd_string_view_v1{
        value.data(),
        static_cast<std::uint64_t>(value.size()),
    };
}


hotpot_rd_string_view_v1 literal_view(const char* value) noexcept {
    std::uint64_t size = 0;
    while (value[size] != '\0') ++size;
    return hotpot_rd_string_view_v1{value, size};
}


hotpot_rd_status exception_status() noexcept {
    try {
        throw;
    } catch (const hotpot::rdwrappers::core::AbiMismatch&) {
        return HOTPOT_RD_STATUS_UNSUPPORTED_ABI;
    } catch (const std::invalid_argument&) {
        return HOTPOT_RD_STATUS_INVALID_ARGUMENT;
    } catch (const std::bad_alloc&) {
        return HOTPOT_RD_STATUS_OUT_OF_MEMORY;
    } catch (...) {
        return HOTPOT_RD_STATUS_INTERNAL_ERROR;
    }
}


}  // namespace


extern "C" {


const char* hotpot_rd_status_message(hotpot_rd_status status) {
    switch (status) {
        case HOTPOT_RD_STATUS_OK:
            return "success";
        case HOTPOT_RD_STATUS_INVALID_ARGUMENT:
            return "invalid argument";
        case HOTPOT_RD_STATUS_UNSUPPORTED_ABI:
            return "unsupported ABI";
        case HOTPOT_RD_STATUS_OUT_OF_MEMORY:
            return "out of memory";
        case HOTPOT_RD_STATUS_INTERNAL_ERROR:
            return "internal error";
        default:
            return "unknown status";
    }
}


hotpot_rd_status hotpot_rd_sampler_create(
    const hotpot_rd_options_v1* options,
    hotpot_rd_sampler** out_sampler
) {
    if (out_sampler == nullptr) return HOTPOT_RD_STATUS_INVALID_ARGUMENT;
    *out_sampler = nullptr;
    if (options == nullptr) return HOTPOT_RD_STATUS_INVALID_ARGUMENT;
    try {
        if (std::string_view(RDKit::rdkitVersion)
            != hotpot::rdwrappers::core::kRDKitVersion) {
            return HOTPOT_RD_STATUS_INTERNAL_ERROR;
        }
        auto sampler = std::make_unique<hotpot_rd_sampler>(
            hotpot_rd_sampler{
                hotpot::rdwrappers::core::validate_options(*options)
            }
        );
        *out_sampler = sampler.release();
        return HOTPOT_RD_STATUS_OK;
    } catch (...) {
        return exception_status();
    }
}


void hotpot_rd_sampler_destroy(hotpot_rd_sampler* sampler) {
    delete sampler;
}


hotpot_rd_status hotpot_rd_generate(
    hotpot_rd_sampler* sampler,
    const hotpot_rd_molecule_view_v1* molecule,
    const hotpot_rd_constraints_view_v1* constraints,
    hotpot_rd_result** out_result
) {
    if (out_result == nullptr) return HOTPOT_RD_STATUS_INVALID_ARGUMENT;
    *out_result = nullptr;
    if (sampler == nullptr || molecule == nullptr) {
        return HOTPOT_RD_STATUS_INVALID_ARGUMENT;
    }
    try {
        hotpot::rdwrappers::core::validate_molecule_view(*molecule);
        if (constraints != nullptr) {
            hotpot::rdwrappers::core::validate_constraints_view(
                *constraints,
                molecule->atom_count
            );
        }
        try {
            auto result = std::make_unique<hotpot_rd_result>(
                hotpot_rd_result{
                    hotpot::rdwrappers::core::generate(
                        sampler->options,
                        *molecule,
                        constraints
                    )
                }
            );
            *out_result = result.release();
        } catch (const hotpot::rdwrappers::core::UnsupportedInput& error) {
            hotpot::rdwrappers::core::Result value;
            value.status = HOTPOT_RD_GENERATION_UNSUPPORTED;
            value.atom_count = molecule->atom_count;
            value.diagnostic = error.what();
            value.has_diagnostic = true;
            auto result = std::make_unique<hotpot_rd_result>(
                hotpot_rd_result{std::move(value)}
            );
            *out_result = result.release();
        }
        return HOTPOT_RD_STATUS_OK;
    } catch (...) {
        return exception_status();
    }
}


hotpot_rd_generation_status hotpot_rd_result_status(
    const hotpot_rd_result* result
) {
    return result == nullptr
        ? HOTPOT_RD_GENERATION_FAILED
        : result->value.status;
}


const double* hotpot_rd_result_coordinates(
    const hotpot_rd_result* result,
    std::uint64_t* conformer_count,
    std::uint64_t* atom_count
) {
    if (conformer_count != nullptr) {
        *conformer_count = result == nullptr
            ? 0
            : static_cast<std::uint64_t>(result->value.records.size());
    }
    if (atom_count != nullptr) {
        *atom_count = result == nullptr ? 0 : result->value.atom_count;
    }
    return result == nullptr || result->value.coordinates.empty()
        ? nullptr
        : result->value.coordinates.data();
}


const hotpot_rd_conformer_record_v1* hotpot_rd_result_records(
    const hotpot_rd_result* result,
    std::uint64_t* record_count
) {
    if (record_count != nullptr) {
        *record_count = result == nullptr
            ? 0
            : static_cast<std::uint64_t>(result->value.records.size());
    }
    return result == nullptr || result->value.records.empty()
        ? nullptr
        : result->value.records.data();
}


hotpot_rd_string_view_v1 hotpot_rd_result_engine(
    const hotpot_rd_result* result
) {
    return result == nullptr
        ? hotpot_rd_string_view_v1{nullptr, 0}
        : string_view(result->value.engine);
}


hotpot_rd_string_view_v1 hotpot_rd_result_engine_version(
    const hotpot_rd_result* result
) {
    return result == nullptr
        ? hotpot_rd_string_view_v1{nullptr, 0}
        : string_view(result->value.engine_version);
}


hotpot_rd_string_view_v1 hotpot_rd_result_diagnostic(
    const hotpot_rd_result* result
) {
    return result == nullptr || !result->value.has_diagnostic
        ? hotpot_rd_string_view_v1{nullptr, 0}
        : string_view(result->value.diagnostic);
}


void hotpot_rd_result_destroy(hotpot_rd_result* result) {
    delete result;
}


std::uint32_t hotpot_rd_runtime_abi_version(void) {
    return HOTPOT_RD_C_API_VERSION;
}


hotpot_rd_string_view_v1 hotpot_rd_runtime_hotpot_version(void) {
    return literal_view(hotpot::rdwrappers::core::kHotpotVersion);
}


hotpot_rd_string_view_v1 hotpot_rd_runtime_rdkit_version(void) {
    return literal_view(RDKit::rdkitVersion);
}


hotpot_rd_string_view_v1 hotpot_rd_runtime_rdkit_commit(void) {
    return literal_view(hotpot::rdwrappers::core::kRDKitCommit);
}


}  // extern "C"
