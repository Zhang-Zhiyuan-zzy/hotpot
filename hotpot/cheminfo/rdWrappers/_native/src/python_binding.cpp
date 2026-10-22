#include <hotpot/rdWrappers/c_api.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>


namespace py = pybind11;


namespace {


class NativeError final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};


std::string status_message(
    const char* operation,
    hotpot_rd_status status
) {
    const auto* message = hotpot_rd_status_message(status);
    return std::string(operation) + ": "
        + (message == nullptr ? "unknown native error" : message);
}


class SamplerHandle final {
public:
    explicit SamplerHandle(hotpot_rd_sampler* value) noexcept : value_(value) {}
    ~SamplerHandle() { hotpot_rd_sampler_destroy(value_); }

    SamplerHandle(const SamplerHandle&) = delete;
    SamplerHandle& operator=(const SamplerHandle&) = delete;

    hotpot_rd_sampler* get() const noexcept { return value_; }

private:
    hotpot_rd_sampler* value_;
};


class ResultHandle final {
public:
    explicit ResultHandle(hotpot_rd_result* value) noexcept : value_(value) {}
    ~ResultHandle() { hotpot_rd_result_destroy(value_); }

    ResultHandle(const ResultHandle&) = delete;
    ResultHandle& operator=(const ResultHandle&) = delete;

    hotpot_rd_result* get() const noexcept { return value_; }

private:
    hotpot_rd_result* value_;
};


std::string copy_string(hotpot_rd_string_view_v1 value) {
    return value.data == nullptr
        ? std::string{}
        : std::string(value.data, static_cast<std::size_t>(value.size));
}


template <typename T>
bool has_exact_dtype(const py::array& value) {
    const auto actual = value.dtype();
    const auto expected = py::dtype::of<T>();
    return actual.equal(expected);
}


py::array as_array(py::handle value, const char* name) {
    if (!py::isinstance<py::array>(value)) {
        throw py::type_error(std::string(name) + " must be a NumPy array");
    }
    return py::reinterpret_borrow<py::array>(value);
}


void require_shape(
    const py::array& value,
    const std::vector<py::ssize_t>& shape,
    const char* name
) {
    if (value.ndim() != static_cast<py::ssize_t>(shape.size())) {
        throw py::value_error(std::string(name) + " has an invalid rank");
    }
    for (py::ssize_t axis = 0; axis < value.ndim(); ++axis) {
        const auto expected = shape[static_cast<std::size_t>(axis)];
        if (expected >= 0 && value.shape(axis) != expected) {
            throw py::value_error(std::string(name) + " has an invalid shape");
        }
    }
}


template <typename T>
const T* checked_array(
    py::handle value,
    const std::vector<py::ssize_t>& shape,
    const char* name,
    std::vector<py::array>& keepalive
) {
    auto array = as_array(value, name);
    if (!has_exact_dtype<T>(array)) {
        throw py::type_error(std::string(name) + " has an invalid dtype");
    }
    if ((array.flags() & py::array::c_style) == 0) {
        throw py::value_error(std::string(name) + " must be C-contiguous");
    }
    if (reinterpret_cast<std::uintptr_t>(array.data()) % alignof(T) != 0) {
        throw py::value_error(std::string(name) + " must be aligned");
    }
    require_shape(array, shape, name);
    keepalive.push_back(array);
    return static_cast<const T*>(array.data());
}


template <typename T>
const T* optional_array(
    py::handle owner,
    const char* attribute,
    const std::vector<py::ssize_t>& shape,
    const char* name,
    std::uint64_t field,
    std::uint64_t& field_mask,
    std::vector<py::array>& keepalive
) {
    const auto value = owner.attr(attribute);
    if (value.is_none()) return nullptr;
    field_mask |= field;
    return checked_array<T>(value, shape, name, keepalive);
}


std::uint64_t nonnegative_size(py::handle value, const char* name) {
    const auto result = py::cast<std::int64_t>(value);
    if (result < 0) {
        throw py::value_error(std::string(name) + " must be nonnegative");
    }
    return static_cast<std::uint64_t>(result);
}


struct Utf8Table {
    const std::uint8_t* data = nullptr;
    std::uint64_t data_size = 0;
    const std::uint64_t* offsets = nullptr;
    const std::uint64_t* lengths = nullptr;
    const std::uint8_t* present = nullptr;
    std::uint64_t row_count = 0;

    hotpot_rd_string_view_v1 at(std::uint64_t row, const char* name) const {
        if (row >= row_count) {
            throw py::value_error(std::string(name) + " row is out of range");
        }
        if (present[row] == 0) {
            if (offsets[row] != 0 || lengths[row] != 0) {
                throw py::value_error(
                    std::string(name) + " has data for an absent value"
                );
            }
            return {nullptr, 0};
        }
        if (present[row] != 1 || offsets[row] > data_size
            || lengths[row] > data_size - offsets[row]) {
            throw py::value_error(std::string(name) + " is structurally invalid");
        }
        static constexpr char empty[] = "";
        return {
            lengths[row] == 0
                ? empty
                : reinterpret_cast<const char*>(data + offsets[row]),
            lengths[row],
        };
    }
};


Utf8Table utf8_table(
    py::handle table,
    std::uint64_t row_count,
    const char* name,
    std::vector<py::array>& keepalive
) {
    const auto data_value = table.attr("data");
    const auto data_array = as_array(
        data_value, (std::string(name) + ".data").c_str()
    );
    if (data_array.ndim() != 1) {
        throw py::value_error(std::string(name) + ".data has an invalid shape");
    }
    const auto data_size = static_cast<std::uint64_t>(data_array.shape(0));
    const auto* data = checked_array<std::uint8_t>(
        data_value,
        {static_cast<py::ssize_t>(data_size)},
        (std::string(name) + ".data").c_str(),
        keepalive
    );
    const auto* offsets = checked_array<std::uint64_t>(
        table.attr("offsets"),
        {static_cast<py::ssize_t>(row_count)},
        (std::string(name) + ".offsets").c_str(),
        keepalive
    );
    const auto* lengths = checked_array<std::uint64_t>(
        table.attr("lengths"),
        {static_cast<py::ssize_t>(row_count)},
        (std::string(name) + ".lengths").c_str(),
        keepalive
    );
    const auto* present = checked_array<std::uint8_t>(
        table.attr("present"),
        {static_cast<py::ssize_t>(row_count)},
        (std::string(name) + ".present").c_str(),
        keepalive
    );
    return Utf8Table{data, data_size, offsets, lengths, present, row_count};
}


struct PackedMolecule {
    hotpot_rd_molecule_view_v1 view{};
    std::vector<hotpot_rd_stereo_record_v1> stereo_records;
    std::vector<hotpot_rd_stereo_carrier_v1> stereo_carriers;
    std::vector<hotpot_rd_atom_transport_v1> atom_transport;
    std::vector<hotpot_rd_bond_transport_v1> bond_transport;
    std::vector<py::array> keepalive;
};


void pack_stereo(py::handle source, PackedMolecule& packed) {
    if (source.is_none()) return;
    packed.view.field_mask |= HOTPOT_RD_FIELD_STEREO_RECORDS;
    if (py::cast<std::int32_t>(source.attr("schema_version"))
        != packed.view.schema_version) {
        throw py::value_error("stereo_records schema_version mismatch");
    }

    const auto anchor_kinds_value = source.attr("anchor_kinds");
    const auto anchor_kinds_array = as_array(
        anchor_kinds_value, "stereo_records.anchor_kinds"
    );
    if (anchor_kinds_array.ndim() != 1) {
        throw py::value_error("stereo_records.anchor_kinds has an invalid shape");
    }
    const auto record_count = static_cast<std::uint64_t>(
        anchor_kinds_array.shape(0)
    );
    const auto row_shape = std::vector<py::ssize_t>{
        static_cast<py::ssize_t>(record_count)
    };
    const auto* anchor_kinds = checked_array<std::uint32_t>(
        anchor_kinds_value, row_shape, "stereo_records.anchor_kinds",
        packed.keepalive
    );
    const auto* anchor_offsets = checked_array<std::uint64_t>(
        source.attr("anchor_offsets"), row_shape,
        "stereo_records.anchor_offsets", packed.keepalive
    );
    const auto* anchor_counts = checked_array<std::uint64_t>(
        source.attr("anchor_counts"), row_shape,
        "stereo_records.anchor_counts", packed.keepalive
    );
    const auto* carrier_offsets = checked_array<std::uint64_t>(
        source.attr("carrier_offsets"), row_shape,
        "stereo_records.carrier_offsets", packed.keepalive
    );
    const auto* carrier_counts = checked_array<std::uint64_t>(
        source.attr("carrier_counts"), row_shape,
        "stereo_records.carrier_counts", packed.keepalive
    );
    const auto* group_kinds = checked_array<std::uint32_t>(
        source.attr("group_kinds"), row_shape,
        "stereo_records.group_kinds", packed.keepalive
    );
    const auto* group_ids = checked_array<std::int64_t>(
        source.attr("group_ids"), row_shape,
        "stereo_records.group_ids", packed.keepalive
    );
    const auto* group_present = checked_array<std::uint32_t>(
        source.attr("group_present"), row_shape,
        "stereo_records.group_present", packed.keepalive
    );

    const auto anchor_indices_value = source.attr("anchor_indices");
    const auto anchor_indices_array = as_array(
        anchor_indices_value, "stereo_records.anchor_indices"
    );
    if (anchor_indices_array.ndim() != 1) {
        throw py::value_error("stereo_records.anchor_indices has an invalid shape");
    }
    const auto anchor_index_count = static_cast<std::uint64_t>(
        anchor_indices_array.shape(0)
    );
    packed.view.stereo_anchor_indices = checked_array<std::uint64_t>(
        anchor_indices_value,
        {static_cast<py::ssize_t>(anchor_index_count)},
        "stereo_records.anchor_indices", packed.keepalive
    );
    packed.view.stereo_anchor_index_count = anchor_index_count;

    const auto carrier_kinds_value = source.attr("carrier_kinds");
    const auto carrier_kinds_array = as_array(
        carrier_kinds_value, "stereo_records.carrier_kinds"
    );
    if (carrier_kinds_array.ndim() != 1) {
        throw py::value_error("stereo_records.carrier_kinds has an invalid shape");
    }
    const auto carrier_count = static_cast<std::uint64_t>(
        carrier_kinds_array.shape(0)
    );
    const auto carrier_shape = std::vector<py::ssize_t>{
        static_cast<py::ssize_t>(carrier_count)
    };
    const auto* carrier_kinds = checked_array<std::uint32_t>(
        carrier_kinds_value, carrier_shape, "stereo_records.carrier_kinds",
        packed.keepalive
    );
    const auto* carrier_has_atoms = checked_array<std::uint32_t>(
        source.attr("carrier_has_atoms"), carrier_shape,
        "stereo_records.carrier_has_atoms", packed.keepalive
    );
    const auto* carrier_atoms = checked_array<std::uint64_t>(
        source.attr("carrier_atoms"), carrier_shape,
        "stereo_records.carrier_atoms", packed.keepalive
    );
    packed.stereo_carriers.reserve(static_cast<std::size_t>(carrier_count));
    for (std::uint64_t index = 0; index < carrier_count; ++index) {
        packed.stereo_carriers.push_back(
            {carrier_kinds[index], carrier_has_atoms[index], carrier_atoms[index]}
        );
    }

    const auto group_scopes = utf8_table(
        source.attr("group_scopes"), record_count,
        "stereo_records.group_scopes", packed.keepalive
    );
    const auto descriptors = utf8_table(
        source.attr("descriptors"), record_count,
        "stereo_records.descriptors", packed.keepalive
    );
    const auto source_namespaces = utf8_table(
        source.attr("source_namespaces"), record_count,
        "stereo_records.source_namespaces", packed.keepalive
    );
    const auto source_codes = utf8_table(
        source.attr("source_codes"), record_count,
        "stereo_records.source_codes", packed.keepalive
    );

    packed.stereo_records.reserve(static_cast<std::size_t>(record_count));
    for (std::uint64_t index = 0; index < record_count; ++index) {
        packed.stereo_records.push_back(
            hotpot_rd_stereo_record_v1{
                anchor_kinds[index],
                group_kinds[index],
                anchor_offsets[index],
                anchor_counts[index],
                carrier_offsets[index],
                carrier_counts[index],
                group_ids[index],
                group_present[index],
                0,
                group_scopes.at(index, "stereo_records.group_scopes"),
                descriptors.at(index, "stereo_records.descriptors"),
                source_namespaces.at(index, "stereo_records.source_namespaces"),
                source_codes.at(index, "stereo_records.source_codes"),
            }
        );
    }
    packed.view.stereo_records = packed.stereo_records.data();
    packed.view.stereo_record_count = record_count;
    packed.view.stereo_carriers = packed.stereo_carriers.data();
    packed.view.stereo_carrier_count = carrier_count;
}


hotpot_rd_hydrogen_state_v1 hydrogen_state(
    std::uint64_t index,
    const std::uint32_t* presence_masks,
    const std::int32_t* implicit_hydrogens,
    const std::int32_t* bracket_explicit_hydrogens,
    const std::uint32_t* no_implicit
) {
    return {
        presence_masks[index],
        implicit_hydrogens[index],
        bracket_explicit_hydrogens[index],
        no_implicit[index],
    };
}


void pack_atom_transport(py::handle source, PackedMolecule& packed) {
    if (source.is_none()) return;
    packed.view.field_mask |= HOTPOT_RD_FIELD_ATOM_TRANSPORT;
    if (py::cast<std::int32_t>(source.attr("schema_version"))
        != packed.view.schema_version) {
        throw py::value_error("atom_transport schema_version mismatch");
    }

    const auto atoms_value = source.attr("atoms");
    const auto atoms_array = as_array(atoms_value, "atom_transport.atoms");
    if (atoms_array.ndim() != 1) {
        throw py::value_error("atom_transport.atoms has an invalid shape");
    }
    const auto count = static_cast<std::uint64_t>(atoms_array.shape(0));
    const auto shape = std::vector<py::ssize_t>{static_cast<py::ssize_t>(count)};
    const auto* atoms = checked_array<std::uint64_t>(
        atoms_value, shape, "atom_transport.atoms", packed.keepalive
    );
    const auto* masks = checked_array<std::uint32_t>(
        source.attr("presence_masks"), shape,
        "atom_transport.presence_masks", packed.keepalive
    );
    const auto* isotopes = checked_array<std::int32_t>(
        source.attr("isotopes"), shape,
        "atom_transport.isotopes", packed.keepalive
    );
    const auto* radicals = checked_array<std::int32_t>(
        source.attr("radical_electrons"), shape,
        "atom_transport.radical_electrons", packed.keepalive
    );
    const auto* has_current = checked_array<std::uint32_t>(
        source.attr("has_current_hydrogens"), shape,
        "atom_transport.has_current_hydrogens", packed.keepalive
    );
    const auto* current_masks = checked_array<std::uint32_t>(
        source.attr("current_hydrogen_presence_masks"), shape,
        "atom_transport.current_hydrogen_presence_masks", packed.keepalive
    );
    const auto* current_implicit = checked_array<std::int32_t>(
        source.attr("current_implicit_hydrogens"), shape,
        "atom_transport.current_implicit_hydrogens", packed.keepalive
    );
    const auto* current_explicit = checked_array<std::int32_t>(
        source.attr("current_bracket_explicit_hydrogens"), shape,
        "atom_transport.current_bracket_explicit_hydrogens", packed.keepalive
    );
    const auto* current_no_implicit = checked_array<std::uint32_t>(
        source.attr("current_no_implicit"), shape,
        "atom_transport.current_no_implicit", packed.keepalive
    );
    const auto* has_source = checked_array<std::uint32_t>(
        source.attr("has_source_hydrogens"), shape,
        "atom_transport.has_source_hydrogens", packed.keepalive
    );
    const auto* source_masks = checked_array<std::uint32_t>(
        source.attr("source_hydrogen_presence_masks"), shape,
        "atom_transport.source_hydrogen_presence_masks", packed.keepalive
    );
    const auto* source_implicit = checked_array<std::int32_t>(
        source.attr("source_implicit_hydrogens"), shape,
        "atom_transport.source_implicit_hydrogens", packed.keepalive
    );
    const auto* source_explicit = checked_array<std::int32_t>(
        source.attr("source_bracket_explicit_hydrogens"), shape,
        "atom_transport.source_bracket_explicit_hydrogens", packed.keepalive
    );
    const auto* source_no_implicit = checked_array<std::uint32_t>(
        source.attr("source_no_implicit"), shape,
        "atom_transport.source_no_implicit", packed.keepalive
    );
    const auto source_namespaces = utf8_table(
        source.attr("source_namespaces"), count,
        "atom_transport.source_namespaces", packed.keepalive
    );

    packed.atom_transport.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t index = 0; index < count; ++index) {
        packed.atom_transport.push_back(
            hotpot_rd_atom_transport_v1{
                atoms[index],
                masks[index],
                isotopes[index],
                radicals[index],
                has_current[index],
                has_source[index],
                hydrogen_state(
                    index, current_masks, current_implicit,
                    current_explicit, current_no_implicit
                ),
                hydrogen_state(
                    index, source_masks, source_implicit,
                    source_explicit, source_no_implicit
                ),
                source_namespaces.at(index, "atom_transport.source_namespaces"),
            }
        );
    }
    packed.view.atom_transport = packed.atom_transport.data();
    packed.view.atom_transport_count = count;
}


void pack_bond_transport(py::handle source, PackedMolecule& packed) {
    if (source.is_none()) return;
    packed.view.field_mask |= HOTPOT_RD_FIELD_BOND_TRANSPORT;
    if (py::cast<std::int32_t>(source.attr("schema_version"))
        != packed.view.schema_version) {
        throw py::value_error("bond_transport schema_version mismatch");
    }

    const auto pairs_value = source.attr("atom_indices");
    const auto pairs_array = as_array(
        pairs_value, "bond_transport.atom_indices"
    );
    if (pairs_array.ndim() != 2 || pairs_array.shape(1) != 2) {
        throw py::value_error("bond_transport.atom_indices has an invalid shape");
    }
    const auto count = static_cast<std::uint64_t>(pairs_array.shape(0));
    const auto* pairs = checked_array<std::uint64_t>(
        pairs_value, {static_cast<py::ssize_t>(count), 2},
        "bond_transport.atom_indices", packed.keepalive
    );
    const auto directions = utf8_table(
        source.attr("display_directions"), count,
        "bond_transport.display_directions", packed.keepalive
    );

    packed.bond_transport.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t index = 0; index < count; ++index) {
        packed.bond_transport.push_back(
            hotpot_rd_bond_transport_v1{
                pairs[2 * index],
                pairs[2 * index + 1],
                directions.at(index, "bond_transport.display_directions"),
            }
        );
    }
    packed.view.bond_transport = packed.bond_transport.data();
    packed.view.bond_transport_count = count;
}


PackedMolecule pack_molecule(py::handle source) {
    PackedMolecule packed;
    packed.view.struct_size = sizeof(hotpot_rd_molecule_view_v1);
    packed.view.abi_version = HOTPOT_RD_C_API_VERSION;
    packed.view.schema_version = py::cast<std::int32_t>(
        source.attr("schema_version")
    );
    packed.view.atom_count = nonnegative_size(
        source.attr("atom_count"), "atom_count"
    );
    packed.view.bond_count = nonnegative_size(
        source.attr("bond_count"), "bond_count"
    );
    if (packed.view.atom_count
            > static_cast<std::uint64_t>(std::numeric_limits<py::ssize_t>::max())
        || packed.view.bond_count
            > static_cast<std::uint64_t>(std::numeric_limits<py::ssize_t>::max())) {
        throw py::value_error("molecule dimensions exceed the Python array ABI");
    }
    const auto atoms = static_cast<py::ssize_t>(packed.view.atom_count);
    const auto bonds = static_cast<py::ssize_t>(packed.view.bond_count);

    packed.view.atomic_numbers = optional_array<std::int32_t>(
        source, "atomic_numbers", {atoms}, "atomic_numbers",
        HOTPOT_RD_FIELD_ATOMIC_NUMBERS, packed.view.field_mask, packed.keepalive
    );
    packed.view.formal_charges = optional_array<std::int32_t>(
        source, "formal_charges", {atoms}, "formal_charges",
        HOTPOT_RD_FIELD_FORMAL_CHARGES, packed.view.field_mask, packed.keepalive
    );
    packed.view.partial_charges = optional_array<double>(
        source, "partial_charges", {atoms}, "partial_charges",
        HOTPOT_RD_FIELD_PARTIAL_CHARGES, packed.view.field_mask, packed.keepalive
    );
    packed.view.coordinates = optional_array<double>(
        source, "coordinates", {atoms, 3}, "coordinates",
        HOTPOT_RD_FIELD_COORDINATES, packed.view.field_mask, packed.keepalive
    );
    packed.view.atom_aromatic = optional_array<std::uint8_t>(
        source, "atom_aromatic", {atoms}, "atom_aromatic",
        HOTPOT_RD_FIELD_ATOM_AROMATIC, packed.view.field_mask, packed.keepalive
    );
    packed.view.bond_indices = optional_array<std::int32_t>(
        source, "bond_indices", {bonds, 2}, "bond_indices",
        HOTPOT_RD_FIELD_BOND_INDICES, packed.view.field_mask, packed.keepalive
    );
    packed.view.bond_orders = optional_array<double>(
        source, "bond_orders", {bonds}, "bond_orders",
        HOTPOT_RD_FIELD_BOND_ORDERS, packed.view.field_mask, packed.keepalive
    );
    packed.view.bond_kinds = optional_array<std::uint8_t>(
        source, "bond_kinds", {bonds}, "bond_kinds",
        HOTPOT_RD_FIELD_BOND_KINDS, packed.view.field_mask, packed.keepalive
    );
    packed.view.bond_aromatic = optional_array<std::uint8_t>(
        source, "bond_aromatic", {bonds}, "bond_aromatic",
        HOTPOT_RD_FIELD_BOND_AROMATIC, packed.view.field_mask, packed.keepalive
    );
    packed.view.unit_cell = optional_array<double>(
        source, "unit_cell", {6}, "unit_cell",
        HOTPOT_RD_FIELD_UNIT_CELL, packed.view.field_mask, packed.keepalive
    );

    pack_stereo(source.attr("stereo_records"), packed);
    pack_atom_transport(source.attr("atom_transport"), packed);
    pack_bond_transport(source.attr("bond_transport"), packed);
    return packed;
}


hotpot_rd_options_v1 pack_options(py::handle source) {
    const auto conformer_count = nonnegative_size(
        source.attr("conformer_count"), "conformer_count"
    );
    const auto seed = source.attr("seed");
    const auto prune = source.attr("prune_rms_threshold");
    return hotpot_rd_options_v1{
        sizeof(hotpot_rd_options_v1),
        HOTPOT_RD_C_API_VERSION,
        conformer_count,
        seed.is_none() ? 0 : py::cast<std::int64_t>(seed),
        prune.is_none() ? 0.0 : py::cast<double>(prune),
        seed.is_none() ? 0u : 1u,
        prune.is_none() ? 0u : 1u,
        py::cast<bool>(source.attr("use_random_coordinates")) ? 1u : 0u,
        0u,
    };
}


struct PackedConstraints {
    hotpot_rd_constraints_view_v1 view{};
    std::vector<py::array> keepalive;
};


PackedConstraints pack_constraints(py::handle source) {
    PackedConstraints packed;
    packed.view.struct_size = sizeof(hotpot_rd_constraints_view_v1);
    packed.view.abi_version = HOTPOT_RD_C_API_VERSION;

    const auto coordination_pairs_value = source.attr("coordination_pairs");
    const auto coordination_pairs = as_array(
        coordination_pairs_value, "coordination_pairs"
    );
    if (coordination_pairs.ndim() != 2 || coordination_pairs.shape(1) != 2) {
        throw py::value_error("coordination_pairs must have shape (N, 2)");
    }
    packed.view.coordination_pair_count = static_cast<std::uint64_t>(
        coordination_pairs.shape(0)
    );
    packed.view.coordination_pairs = checked_array<std::int32_t>(
        coordination_pairs_value,
        {coordination_pairs.shape(0), 2},
        "coordination_pairs", packed.keepalive
    );
    packed.view.metal_donor_bounds = checked_array<double>(
        source.attr("metal_donor_bounds"),
        {coordination_pairs.shape(0), 2},
        "metal_donor_bounds", packed.keepalive
    );

    const auto donor_pairs_value = source.attr("donor_pair_indices");
    const auto donor_bounds_value = source.attr("donor_pair_bounds");
    if (donor_pairs_value.is_none() != donor_bounds_value.is_none()) {
        throw py::value_error(
            "donor_pair_indices and donor_pair_bounds must both be present or absent"
        );
    }
    if (!donor_pairs_value.is_none()) {
        const auto donor_pairs = as_array(
            donor_pairs_value, "donor_pair_indices"
        );
        if (donor_pairs.ndim() != 2 || donor_pairs.shape(1) != 2) {
            throw py::value_error("donor_pair_indices must have shape (N, 2)");
        }
        packed.view.donor_pair_count = static_cast<std::uint64_t>(
            donor_pairs.shape(0)
        );
        packed.view.donor_pair_indices = checked_array<std::int32_t>(
            donor_pairs_value, {donor_pairs.shape(0), 2},
            "donor_pair_indices", packed.keepalive
        );
        packed.view.donor_pair_bounds = checked_array<double>(
            donor_bounds_value, {donor_pairs.shape(0), 2},
            "donor_pair_bounds", packed.keepalive
        );
    }
    return packed;
}


py::object copy_result(hotpot_rd_result* result) {
    std::uint64_t conformer_count = 0;
    std::uint64_t atom_count = 0;
    const auto* coordinates = hotpot_rd_result_coordinates(
        result, &conformer_count, &atom_count
    );
    if (conformer_count
            > static_cast<std::uint64_t>(std::numeric_limits<py::ssize_t>::max())
        || atom_count
            > static_cast<std::uint64_t>(std::numeric_limits<py::ssize_t>::max())) {
        throw NativeError("native result dimensions exceed the Python array ABI");
    }
    if (conformer_count != 0
        && atom_count > std::numeric_limits<std::uint64_t>::max()
            / conformer_count / 3) {
        throw NativeError("native result coordinate dimensions overflow");
    }
    const auto value_count = conformer_count * atom_count * 3;
    if (value_count != 0 && coordinates == nullptr) {
        throw NativeError("native result returned no coordinate storage");
    }
    py::array_t<double> coordinate_array(
        {
            static_cast<py::ssize_t>(conformer_count),
            static_cast<py::ssize_t>(atom_count),
            py::ssize_t{3},
        }
    );
    if (value_count != 0) {
        std::memcpy(
            coordinate_array.mutable_data(),
            coordinates,
            static_cast<std::size_t>(value_count) * sizeof(double)
        );
    }

    std::uint64_t record_count = 0;
    const auto* records = hotpot_rd_result_records(result, &record_count);
    if (record_count != conformer_count
        || (record_count != 0 && records == nullptr)) {
        throw NativeError("native result returned inconsistent conformer records");
    }

    const auto contracts = py::module_::import(
        "hotpot.cheminfo.rdWrappers.contracts"
    );
    const auto record_type = contracts.attr("ConformerRecord");
    py::tuple record_values(static_cast<py::ssize_t>(record_count));
    for (std::uint64_t index = 0; index < record_count; ++index) {
        record_values[static_cast<py::ssize_t>(index)] = record_type(
            records[index].ordinal,
            records[index].engine_conformer_id
        );
    }

    const auto diagnostic = hotpot_rd_result_diagnostic(result);
    const auto diagnostic_value = diagnostic.data == nullptr
        ? py::none()
        : py::object(py::str(copy_string(diagnostic)));
    const auto generation_status = contracts.attr("GenerationStatus")(
        static_cast<std::uint32_t>(hotpot_rd_result_status(result))
    );
    return contracts.attr("ConformerBatch")(
        generation_status,
        std::move(coordinate_array),
        std::move(record_values),
        py::str(copy_string(hotpot_rd_result_engine(result))),
        py::str(copy_string(hotpot_rd_result_engine_version(result))),
        diagnostic_value
    );
}


py::object generate(
    py::object molecule,
    py::object options,
    py::object constraints
) {
    auto packed_molecule = pack_molecule(molecule);
    const auto packed_options = pack_options(options);
    auto packed_constraints = constraints.is_none()
        ? std::optional<PackedConstraints>{}
        : std::optional<PackedConstraints>{pack_constraints(constraints)};

    hotpot_rd_sampler* raw_sampler = nullptr;
    auto status = hotpot_rd_sampler_create(&packed_options, &raw_sampler);
    if (status != HOTPOT_RD_STATUS_OK) {
        throw NativeError(status_message("hotpot_rd_sampler_create", status));
    }
    SamplerHandle sampler(raw_sampler);

    hotpot_rd_result* raw_result = nullptr;
    {
        py::gil_scoped_release release;
        status = hotpot_rd_generate(
            sampler.get(),
            &packed_molecule.view,
            packed_constraints.has_value() ? &packed_constraints->view : nullptr,
            &raw_result
        );
    }
    if (status != HOTPOT_RD_STATUS_OK) {
        throw NativeError(status_message("hotpot_rd_generate", status));
    }
    if (raw_result == nullptr) {
        throw NativeError("hotpot_rd_generate returned no result");
    }
    ResultHandle result(raw_result);
    return copy_result(result.get());
}


py::object runtime_info() {
    if (hotpot_rd_runtime_abi_version() != HOTPOT_RD_C_API_VERSION) {
        throw NativeError("hotpot_rdwrappers C API version mismatch");
    }
    const auto contracts = py::module_::import(
        "hotpot.cheminfo.rdWrappers.contracts"
    );
    return contracts.attr("RuntimeInfo")(
        py::str(copy_string(hotpot_rd_runtime_hotpot_version())),
        py::str(copy_string(hotpot_rd_runtime_rdkit_version())),
        true
    );
}


}  // namespace


PYBIND11_MODULE(_rd_native, module) {
    module.doc() = "Native RDKit 2026.09.1 ETKDGv3 binding";
    module.def(
        "generate",
        &generate,
        py::arg("molecule_buffers"),
        py::arg("options"),
        py::arg("constraints") = py::none(),
        "Generate conformers from field-selective contiguous molecule buffers."
    );
    module.def(
        "runtime_info",
        &runtime_info,
        "Return the identities compiled into this native component."
    );
    module.attr("C_API_VERSION") = HOTPOT_RD_C_API_VERSION;
    module.attr("RDKIT_COMMIT") = py::str(copy_string(
        hotpot_rd_runtime_rdkit_commit()
    ));
}
