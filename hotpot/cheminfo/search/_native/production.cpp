#include "production.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>


namespace hotpot::search {

namespace {

using NativeValue = std::variant<
    std::monostate,
    bool,
    std::int64_t,
    double,
    std::string
>;

enum class Truth : std::uint8_t { False, True, Unknown };
enum class AtomOp : std::uint8_t {
    True,
    FieldEq,
    FieldIn,
    PeriodIn,
    DegreeEq,
    ConnectivityEq,
    ValenceEq,
    RingAny,
    RingCountEq,
    RingNone,
    RingSizeEq,
    TotalHEq,
    Recursive,
    And,
    Or,
    Not,
};
enum class BondOp : std::uint8_t { True, KindIs, Aromatic, Or };
enum class LeafKind : std::uint8_t {
    NativeSet,
    HostSet,
    Callable,
    AtomExpression,
    BondExpression,
};

struct SemanticRecord {
    std::int64_t degree = 0;
    std::int64_t connectivity = 0;
    double valence = 0.0;
    std::int64_t ring_count = 0;
    std::vector<std::int64_t> ring_sizes;
};

struct TargetAtom {
    std::unordered_map<std::string, NativeValue> native_fields;
    std::unordered_map<std::string, py::object> host_fields;
    SemanticRecord full;
    SemanticRecord ligand;
    py::object object;
};

struct TargetBond {
    std::unordered_map<std::string, NativeValue> native_fields;
    std::unordered_map<std::string, py::object> host_fields;
    std::string bond_kind;
    bool is_aromatic = false;
    py::object object;
};

struct RecursiveProgram {
    std::shared_ptr<NativeProblem> problem;
    mutable std::mutex cache_mutex;
    mutable std::vector<std::int8_t> cache;

    bool matches(Position target_node) const {
        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            if (cache[target_node] >= 0) {
                return cache[target_node] != 0;
            }
        }
        const bool result = problem->has_anchored_match(target_node);
        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            cache[target_node] = result ? 1 : 0;
        }
        return result;
    }
};

struct AtomExpression {
    AtomOp op = AtomOp::True;
    std::string field;
    NativeValue value;
    std::vector<NativeValue> values;
    std::string semantics;
    std::vector<AtomExpression> children;
    std::shared_ptr<RecursiveProgram> recursive;
};

struct BondExpression {
    BondOp op = BondOp::True;
    std::string kind;
    std::vector<BondExpression> children;
};

struct Leaf {
    LeafKind kind = LeafKind::NativeSet;
    std::string field;
    std::vector<NativeValue> native_values;
    py::object python_value;
    AtomExpression atom_expression;
    BondExpression bond_expression;
};

struct NodeProgram {
    std::vector<std::string> preflight;
    std::vector<Leaf> leaves;
};

struct BondProgram {
    std::vector<std::string> preflight;
    std::vector<Leaf> leaves;
    bool exclude_aromatic = false;
    std::optional<std::string> bond_kind;
};

py::handle get_item(const py::dict& mapping, const char* key) {
    return mapping[py::str(key)];
}

std::vector<std::int64_t> graph_keys(const py::dict& specification) {
    return py::cast<std::vector<std::int64_t>>(get_item(specification, "keys"));
}

std::vector<std::pair<std::int64_t, std::int64_t>> graph_edges(
    const py::dict& specification
) {
    return py::cast<std::vector<std::pair<std::int64_t, std::int64_t>>>(
        get_item(specification, "edges")
    );
}

std::string string_value(const py::handle value) {
    return py::cast<std::string>(value);
}

std::optional<NativeValue> native_value(const py::handle value) {
    if (value.is_none()) {
        return NativeValue{std::monostate{}};
    }
    if (py::isinstance<py::bool_>(value)) {
        return NativeValue{py::cast<bool>(value)};
    }
    if (py::isinstance<py::int_>(value)) {
        const long long integer = PyLong_AsLongLong(value.ptr());
        if (integer == -1 && PyErr_Occurred()) {
            if (PyErr_ExceptionMatches(PyExc_OverflowError)) {
                PyErr_Clear();
                return std::nullopt;
            }
            throw py::error_already_set();
        }
        return NativeValue{static_cast<std::int64_t>(integer)};
    }
    if (py::isinstance<py::float_>(value)) {
        return NativeValue{py::cast<double>(value)};
    }
    if (py::isinstance<py::str>(value)) {
        return NativeValue{string_value(value)};
    }
    return std::nullopt;
}

bool native_equal(const NativeValue& first, const NativeValue& second) noexcept {
    const bool first_is_integer = std::holds_alternative<bool>(first)
        || std::holds_alternative<std::int64_t>(first);
    const bool second_is_integer = std::holds_alternative<bool>(second)
        || std::holds_alternative<std::int64_t>(second);
    const auto integer = [](const NativeValue& value) {
        if (const auto* boolean = std::get_if<bool>(&value)) {
            return std::int64_t{*boolean ? 1 : 0};
        }
        return std::get<std::int64_t>(value);
    };
    if (first_is_integer && second_is_integer) {
        return integer(first) == integer(second);
    }
    const bool first_is_number = first_is_integer
        || std::holds_alternative<double>(first);
    const bool second_is_number = second_is_integer
        || std::holds_alternative<double>(second);
    if (first_is_number && second_is_number) {
        const auto number = [&](const NativeValue& value) {
            if (const auto* boolean = std::get_if<bool>(&value)) {
                return static_cast<long double>(*boolean ? 1 : 0);
            }
            if (const auto* integer_value = std::get_if<std::int64_t>(&value)) {
                return static_cast<long double>(*integer_value);
            }
            return static_cast<long double>(std::get<double>(value));
        };
        return number(first) == number(second);
    }
    return first == second;
}

bool native_contains(
    const std::vector<NativeValue>& values,
    const NativeValue& candidate
) noexcept {
    return std::any_of(
        values.begin(),
        values.end(),
        [&](const NativeValue& value) { return native_equal(value, candidate); }
    );
}

std::int64_t as_integer(const NativeValue& value) {
    if (const auto* boolean = std::get_if<bool>(&value)) {
        return *boolean ? 1 : 0;
    }
    if (const auto* integer = std::get_if<std::int64_t>(&value)) {
        return *integer;
    }
    return static_cast<std::int64_t>(std::get<double>(value));
}

const NativeValue* find_native(
    const std::unordered_map<std::string, NativeValue>& fields,
    const std::string& name
) noexcept {
    const auto found = fields.find(name);
    return found == fields.end() ? nullptr : &found->second;
}

bool python_truth(const py::object& value) {
    const int result = PyObject_IsTrue(value.ptr());
    if (result < 0) {
        throw py::error_already_set();
    }
    return result != 0;
}

bool python_contains(
    const py::object& container,
    const py::object& candidate
) {
    const int result = PySequence_Contains(container.ptr(), candidate.ptr());
    if (result < 0) {
        throw py::error_already_set();
    }
    return result != 0;
}

SemanticRecord parse_semantic_record(const py::dict& record) {
    return {
        py::cast<std::int64_t>(get_item(record, "degree")),
        py::cast<std::int64_t>(get_item(record, "connectivity")),
        py::cast<double>(get_item(record, "valence")),
        py::cast<std::int64_t>(get_item(record, "ring_count")),
        py::cast<std::vector<std::int64_t>>(get_item(record, "ring_sizes")),
    };
}

template <typename Record>
void parse_fields(Record& record, const py::dict& fields) {
    for (const auto& item : fields) {
        const std::string name = string_value(item.first);
        record.host_fields.emplace(
            name,
            py::reinterpret_borrow<py::object>(item.second)
        );
        const std::optional<NativeValue> value = native_value(item.second);
        if (value.has_value()) {
            record.native_fields.emplace(name, *value);
        }
    }
}

std::vector<TargetAtom> parse_target_atoms(const py::dict& target_spec) {
    std::vector<TargetAtom> atoms;
    for (const py::handle item : get_item(target_spec, "atoms")) {
        const py::dict record = py::reinterpret_borrow<py::dict>(item);
        TargetAtom atom;
        parse_fields(
            atom,
            py::reinterpret_borrow<py::dict>(get_item(record, "fields"))
        );
        atom.full = parse_semantic_record(
            py::reinterpret_borrow<py::dict>(get_item(record, "full"))
        );
        atom.ligand = parse_semantic_record(
            py::reinterpret_borrow<py::dict>(get_item(record, "ligand"))
        );
        atom.object = py::reinterpret_borrow<py::object>(get_item(record, "object"));
        atoms.push_back(std::move(atom));
    }
    return atoms;
}

std::vector<TargetBond> parse_target_bonds(const py::dict& target_spec) {
    std::vector<TargetBond> bonds;
    for (const py::handle item : get_item(target_spec, "bonds")) {
        const py::dict record = py::reinterpret_borrow<py::dict>(item);
        TargetBond bond;
        parse_fields(
            bond,
            py::reinterpret_borrow<py::dict>(get_item(record, "fields"))
        );
        bond.bond_kind = string_value(get_item(record, "bond_kind"));
        bond.is_aromatic = py::cast<bool>(get_item(record, "is_aromatic"));
        bond.object = py::reinterpret_borrow<py::object>(get_item(record, "object"));
        bonds.push_back(std::move(bond));
    }
    return bonds;
}

AtomOp atom_op(const std::string& value) {
    static const std::unordered_map<std::string, AtomOp> operations = {
        {"true", AtomOp::True},
        {"field_eq", AtomOp::FieldEq},
        {"field_in", AtomOp::FieldIn},
        {"period_in", AtomOp::PeriodIn},
        {"degree_eq", AtomOp::DegreeEq},
        {"connectivity_eq", AtomOp::ConnectivityEq},
        {"valence_eq", AtomOp::ValenceEq},
        {"ring_any", AtomOp::RingAny},
        {"ring_count_eq", AtomOp::RingCountEq},
        {"ring_none", AtomOp::RingNone},
        {"ring_size_eq", AtomOp::RingSizeEq},
        {"total_h_eq", AtomOp::TotalHEq},
        {"recursive", AtomOp::Recursive},
        {"and", AtomOp::And},
        {"or", AtomOp::Or},
        {"not", AtomOp::Not},
    };
    return operations.at(value);
}

BondOp bond_op(const std::string& value) {
    static const std::unordered_map<std::string, BondOp> operations = {
        {"true", BondOp::True},
        {"kind_is", BondOp::KindIs},
        {"aromatic", BondOp::Aromatic},
        {"or", BondOp::Or},
    };
    return operations.at(value);
}

std::vector<NativeValue> parse_native_values(const py::handle sequence) {
    std::vector<NativeValue> values;
    for (const py::handle item : sequence) {
        values.push_back(native_value(item).value());
    }
    return values;
}

AtomExpression parse_atom_expression(
    const py::dict& specification,
    const py::dict& target_spec,
    const std::size_t target_node_count
) {
    AtomExpression expression;
    expression.op = atom_op(string_value(get_item(specification, "op")));
    const py::handle field = get_item(specification, "field");
    if (!field.is_none()) {
        expression.field = string_value(field);
    }
    const py::handle value = get_item(specification, "value");
    if (expression.op == AtomOp::Recursive) {
        auto recursive = std::make_shared<RecursiveProgram>();
        recursive->problem = std::make_shared<NativeProblem>(
            py::reinterpret_borrow<py::dict>(value),
            target_spec
        );
        recursive->cache.assign(target_node_count, -1);
        expression.recursive = std::move(recursive);
    } else if (
        expression.op == AtomOp::FieldIn
        || expression.op == AtomOp::PeriodIn
    ) {
        expression.values = parse_native_values(value);
    } else if (!value.is_none()) {
        expression.value = native_value(value).value();
    }
    const py::handle semantics = get_item(specification, "semantics");
    if (!semantics.is_none()) {
        expression.semantics = string_value(semantics);
    }
    for (const py::handle child : get_item(specification, "children")) {
        expression.children.push_back(parse_atom_expression(
            py::reinterpret_borrow<py::dict>(child),
            target_spec,
            target_node_count
        ));
    }
    return expression;
}

BondExpression parse_bond_expression(const py::dict& specification) {
    BondExpression expression;
    expression.op = bond_op(string_value(get_item(specification, "op")));
    const py::handle kind = get_item(specification, "kind");
    if (!kind.is_none()) {
        expression.kind = string_value(kind);
    }
    for (const py::handle child : get_item(specification, "children")) {
        expression.children.push_back(parse_bond_expression(
            py::reinterpret_borrow<py::dict>(child)
        ));
    }
    return expression;
}

std::vector<std::string> parse_preflight(const py::dict& program) {
    return py::cast<std::vector<std::string>>(get_item(program, "preflight"));
}

Leaf parse_leaf(
    const py::handle leaf_handle,
    const py::dict& target_spec,
    const std::size_t target_node_count
) {
    const py::tuple specification = py::reinterpret_borrow<py::tuple>(leaf_handle);
    const std::string kind = string_value(specification[0]);
    Leaf leaf;
    if (kind == "native_set") {
        leaf.kind = LeafKind::NativeSet;
        leaf.field = string_value(specification[1]);
        leaf.native_values = parse_native_values(specification[2]);
    } else if (kind == "host_set") {
        leaf.kind = LeafKind::HostSet;
        leaf.field = string_value(specification[1]);
        leaf.python_value = py::reinterpret_borrow<py::object>(specification[2]);
    } else if (kind == "callable") {
        leaf.kind = LeafKind::Callable;
        leaf.python_value = py::reinterpret_borrow<py::object>(specification[1]);
    } else if (kind == "atom_expr") {
        leaf.kind = LeafKind::AtomExpression;
        leaf.atom_expression = parse_atom_expression(
            py::reinterpret_borrow<py::dict>(specification[1]),
            target_spec,
            target_node_count
        );
    } else if (kind == "bond_expr") {
        leaf.kind = LeafKind::BondExpression;
        leaf.bond_expression = parse_bond_expression(
            py::reinterpret_borrow<py::dict>(specification[1])
        );
    } else {
        throw std::invalid_argument("unknown native search predicate leaf");
    }
    return leaf;
}

std::vector<NodeProgram> parse_node_programs(
    const py::dict& query_spec,
    const py::dict& target_spec,
    const std::size_t target_node_count
) {
    std::vector<NodeProgram> programs;
    for (const py::handle item : get_item(query_spec, "nodes")) {
        const py::dict specification = py::reinterpret_borrow<py::dict>(item);
        NodeProgram program;
        program.preflight = parse_preflight(specification);
        for (const py::handle leaf : get_item(specification, "leaves")) {
            program.leaves.push_back(
                parse_leaf(leaf, target_spec, target_node_count)
            );
        }
        programs.push_back(std::move(program));
    }
    return programs;
}

std::vector<BondProgram> parse_bond_programs(
    const py::dict& query_spec,
    const py::dict& target_spec,
    const std::size_t target_node_count
) {
    std::vector<BondProgram> programs;
    for (const py::handle item : get_item(query_spec, "bonds")) {
        const py::dict specification = py::reinterpret_borrow<py::dict>(item);
        BondProgram program;
        program.preflight = parse_preflight(specification);
        program.exclude_aromatic = py::cast<bool>(
            get_item(specification, "exclude_aromatic")
        );
        const py::handle bond_kind = get_item(specification, "bond_kind");
        if (!bond_kind.is_none()) {
            program.bond_kind = string_value(bond_kind);
        }
        for (const py::handle leaf : get_item(specification, "leaves")) {
            program.leaves.push_back(
                parse_leaf(leaf, target_spec, target_node_count)
            );
        }
        programs.push_back(std::move(program));
    }
    return programs;
}

const SemanticRecord& semantic_record(
    const TargetAtom& atom,
    const std::string& semantics
) noexcept {
    return semantics == "ligand_skeleton" || semantics == "ligand"
        ? atom.ligand
        : atom.full;
}

std::int64_t period_number(const std::int64_t atomic_number) noexcept {
    static constexpr std::int64_t upper_bounds[] = {2, 10, 18, 36, 54, 86, 118};
    for (std::int64_t period = 1; period <= 7; ++period) {
        if (atomic_number <= upper_bounds[period - 1]) {
            return period;
        }
    }
    return 0;
}

bool evaluate_atom_expression(
    const AtomExpression& expression,
    const TargetAtom& atom,
    const Position target_node
) {
    const NativeValue* field = nullptr;
    switch (expression.op) {
        case AtomOp::True:
            return true;
        case AtomOp::FieldEq:
            field = find_native(atom.native_fields, expression.field);
            return field != nullptr && native_equal(*field, expression.value);
        case AtomOp::FieldIn:
            field = find_native(atom.native_fields, expression.field);
            return field != nullptr && native_contains(expression.values, *field);
        case AtomOp::PeriodIn:
            field = find_native(atom.native_fields, "atomic_number");
            return field != nullptr && native_contains(
                expression.values,
                NativeValue{period_number(as_integer(*field))}
            );
        case AtomOp::DegreeEq:
            return native_equal(
                NativeValue{semantic_record(atom, expression.semantics).degree},
                expression.value
            );
        case AtomOp::ConnectivityEq:
            return native_equal(
                NativeValue{semantic_record(atom, expression.semantics).connectivity},
                expression.value
            );
        case AtomOp::ValenceEq:
            return native_equal(
                NativeValue{semantic_record(atom, expression.semantics).valence},
                expression.value
            );
        case AtomOp::RingAny:
            return semantic_record(atom, expression.semantics).ring_count != 0;
        case AtomOp::RingCountEq:
            return native_equal(
                NativeValue{semantic_record(atom, expression.semantics).ring_count},
                expression.value
            );
        case AtomOp::RingNone:
            return semantic_record(atom, expression.semantics).ring_count == 0;
        case AtomOp::RingSizeEq: {
            const auto& sizes = semantic_record(atom, expression.semantics).ring_sizes;
            return std::any_of(
                sizes.begin(),
                sizes.end(),
                [&](const std::int64_t size) {
                    return native_equal(NativeValue{size}, expression.value);
                }
            );
        }
        case AtomOp::TotalHEq: {
            const NativeValue* explicit_h = find_native(
                atom.native_fields,
                "explicit_hydrogens"
            );
            const NativeValue* implicit_h = find_native(
                atom.native_fields,
                "implicit_hydrogens"
            );
            return explicit_h != nullptr && implicit_h != nullptr && native_equal(
                NativeValue{as_integer(*explicit_h) + as_integer(*implicit_h)},
                expression.value
            );
        }
        case AtomOp::Recursive:
            return expression.recursive->matches(target_node);
        case AtomOp::And:
            for (const AtomExpression& child : expression.children) {
                if (!evaluate_atom_expression(child, atom, target_node)) {
                    return false;
                }
            }
            return true;
        case AtomOp::Or:
            for (const AtomExpression& child : expression.children) {
                if (evaluate_atom_expression(child, atom, target_node)) {
                    return true;
                }
            }
            return false;
        case AtomOp::Not:
            return !evaluate_atom_expression(
                expression.children.front(),
                atom,
                target_node
            );
    }
    return false;
}

Truth may_evaluate_atom_expression(
    const AtomExpression& expression,
    const TargetAtom& atom
) {
    if (expression.op == AtomOp::Recursive) {
        return Truth::Unknown;
    }
    if (expression.op == AtomOp::And) {
        bool unknown = false;
        for (const AtomExpression& child : expression.children) {
            const Truth result = may_evaluate_atom_expression(child, atom);
            if (result == Truth::False) {
                return Truth::False;
            }
            unknown = unknown || result == Truth::Unknown;
        }
        return unknown ? Truth::Unknown : Truth::True;
    }
    if (expression.op == AtomOp::Or) {
        bool unknown = false;
        for (const AtomExpression& child : expression.children) {
            const Truth result = may_evaluate_atom_expression(child, atom);
            if (result == Truth::True) {
                return Truth::True;
            }
            unknown = unknown || result == Truth::Unknown;
        }
        return unknown ? Truth::Unknown : Truth::False;
    }
    if (expression.op == AtomOp::Not) {
        const Truth result = may_evaluate_atom_expression(
            expression.children.front(),
            atom
        );
        if (result == Truth::Unknown) {
            return Truth::Unknown;
        }
        return result == Truth::True ? Truth::False : Truth::True;
    }
    return evaluate_atom_expression(expression, atom, 0)
        ? Truth::True
        : Truth::False;
}

bool evaluate_bond_expression(
    const BondExpression& expression,
    const TargetBond& bond
) noexcept {
    switch (expression.op) {
        case BondOp::True:
            return true;
        case BondOp::KindIs:
            return bond.bond_kind == expression.kind;
        case BondOp::Aromatic:
            return bond.is_aromatic;
        case BondOp::Or:
            return std::any_of(
                expression.children.begin(),
                expression.children.end(),
                [&](const BondExpression& child) {
                    return evaluate_bond_expression(child, bond);
                }
            );
    }
    return false;
}

bool python_has_attribute(
    const py::object& object,
    const std::string& name
) {
    PyObject* value = PyObject_GetAttrString(object.ptr(), name.c_str());
    if (value != nullptr) {
        Py_DECREF(value);
        return true;
    }
    if (PyErr_ExceptionMatches(PyExc_AttributeError)) {
        PyErr_Clear();
        return false;
    }
    throw py::error_already_set();
}

py::object python_attribute(
    const py::object& object,
    const std::string& name
) {
    PyObject* value = PyObject_GetAttrString(object.ptr(), name.c_str());
    if (value == nullptr) {
        throw py::error_already_set();
    }
    return py::reinterpret_steal<py::object>(value);
}

template <typename Target>
bool accept_preflight_matches(
    const std::vector<std::string>& preflight,
    const Target& target
) {
    for (const std::string& name : preflight) {
        if (target.host_fields.find(name) != target.host_fields.end()) {
            continue;
        }
        py::gil_scoped_acquire acquire;
        if (!python_has_attribute(target.object, name)) {
            return false;
        }
    }
    return true;
}

template <typename Target>
bool may_preflight_match(
    const std::vector<std::string>& preflight,
    const Target& target
) noexcept {
    // Packed fields are known to exist. Unpacked names are deliberately
    // unknown here: checking them would invoke Python during domain pruning.
    (void)preflight;
    (void)target;
    return true;
}

template <typename Target>
bool evaluate_python_leaf(const Leaf& leaf, const Target& target) {
    py::gil_scoped_acquire acquire;
    if (leaf.kind == LeafKind::Callable) {
        return python_truth(leaf.python_value(target.object));
    }
    if (!python_truth(leaf.python_value)) {
        return true;
    }
    return python_contains(
        leaf.python_value,
        python_attribute(target.object, leaf.field)
    );
}

class NativeCompatibility final : public Compatibility {
public:
    NativeCompatibility(
        const py::dict& query_spec,
        const py::dict& target_spec,
        const std::size_t query_node_count,
        const std::size_t query_edge_count,
        const std::size_t target_node_count,
        const std::size_t target_edge_count
    ) : nodes_(parse_node_programs(query_spec, target_spec, target_node_count)),
        bonds_(parse_bond_programs(query_spec, target_spec, target_node_count)),
        atoms_(parse_target_atoms(target_spec)),
        target_bonds_(parse_target_bonds(target_spec)) {
        if (
            nodes_.size() != query_node_count
            || bonds_.size() != query_edge_count
            || atoms_.size() != target_node_count
            || target_bonds_.size() != target_edge_count
        ) {
            throw std::invalid_argument(
                "native search records must align with graph topology"
            );
        }
    }

    bool accept_node(const Position query, const Position target) const override {
        const NodeProgram& program = nodes_[query];
        const TargetAtom& atom = atoms_[target];
        if (!accept_preflight_matches(program.preflight, atom)) {
            return false;
        }
        for (const Leaf& leaf : program.leaves) {
            if (leaf.kind == LeafKind::NativeSet) {
                if (leaf.native_values.empty()) {
                    continue;
                }
                const NativeValue* value = find_native(
                    atom.native_fields,
                    leaf.field
                );
                if (value == nullptr || !native_contains(leaf.native_values, *value)) {
                    return false;
                }
            } else if (
                leaf.kind == LeafKind::HostSet
                || leaf.kind == LeafKind::Callable
            ) {
                if (!evaluate_python_leaf(leaf, atom)) {
                    return false;
                }
            } else if (
                leaf.kind == LeafKind::AtomExpression
                && !evaluate_atom_expression(leaf.atom_expression, atom, target)
            ) {
                return false;
            }
        }
        return true;
    }

    bool accept_edge(const EdgeIndex query, const EdgeIndex target) const override {
        const BondProgram& program = bonds_[query];
        const TargetBond& bond = target_bonds_[target];
        if (program.exclude_aromatic && bond.is_aromatic) {
            return false;
        }
        if (program.bond_kind.has_value() && *program.bond_kind != bond.bond_kind) {
            return false;
        }
        if (!accept_preflight_matches(program.preflight, bond)) {
            return false;
        }
        for (const Leaf& leaf : program.leaves) {
            if (leaf.kind == LeafKind::NativeSet) {
                if (leaf.native_values.empty()) {
                    continue;
                }
                const NativeValue* value = find_native(
                    bond.native_fields,
                    leaf.field
                );
                if (value == nullptr || !native_contains(leaf.native_values, *value)) {
                    return false;
                }
            } else if (
                leaf.kind == LeafKind::HostSet
                || leaf.kind == LeafKind::Callable
            ) {
                if (!evaluate_python_leaf(leaf, bond)) {
                    return false;
                }
            } else if (
                leaf.kind == LeafKind::BondExpression
                && !evaluate_bond_expression(leaf.bond_expression, bond)
            ) {
                return false;
            }
        }
        return true;
    }

    bool may_accept_node(
        const Position query,
        const Position target
    ) const override {
        const NodeProgram& program = nodes_[query];
        const TargetAtom& atom = atoms_[target];
        if (!may_preflight_match(program.preflight, atom)) {
            return false;
        }
        for (const Leaf& leaf : program.leaves) {
            if (leaf.kind == LeafKind::NativeSet) {
                if (leaf.native_values.empty()) {
                    continue;
                }
                const NativeValue* value = find_native(
                    atom.native_fields,
                    leaf.field
                );
                if (value == nullptr || !native_contains(leaf.native_values, *value)) {
                    return false;
                }
            } else if (
                leaf.kind == LeafKind::AtomExpression
                && may_evaluate_atom_expression(leaf.atom_expression, atom)
                    == Truth::False
            ) {
                return false;
            }
        }
        return true;
    }

    bool may_accept_edge(
        const EdgeIndex query,
        const EdgeIndex target
    ) const override {
        const BondProgram& program = bonds_[query];
        const TargetBond& bond = target_bonds_[target];
        if (program.exclude_aromatic && bond.is_aromatic) {
            return false;
        }
        if (program.bond_kind.has_value() && *program.bond_kind != bond.bond_kind) {
            return false;
        }
        if (!may_preflight_match(program.preflight, bond)) {
            return false;
        }
        for (const Leaf& leaf : program.leaves) {
            if (leaf.kind == LeafKind::NativeSet) {
                if (leaf.native_values.empty()) {
                    continue;
                }
                const NativeValue* value = find_native(
                    bond.native_fields,
                    leaf.field
                );
                if (value == nullptr || !native_contains(leaf.native_values, *value)) {
                    return false;
                }
            } else if (
                leaf.kind == LeafKind::BondExpression
                && !evaluate_bond_expression(leaf.bond_expression, bond)
            ) {
                return false;
            }
        }
        return true;
    }

private:
    std::vector<NodeProgram> nodes_;
    std::vector<BondProgram> bonds_;
    std::vector<TargetAtom> atoms_;
    std::vector<TargetBond> target_bonds_;
};

}  // namespace

NativeProblem::NativeProblem(
    const py::dict& query_spec,
    const py::dict& target_spec
) : query_(graph_keys(query_spec), graph_edges(query_spec)),
    target_(graph_keys(target_spec), graph_edges(target_spec)),
    plan_(build_query_plan(query_)),
    compatibility_(std::make_unique<NativeCompatibility>(
        query_spec,
        target_spec,
        query_.node_count(),
        query_.edge_count(),
        target_.node_count(),
        target_.edge_count()
    )) {
    const std::size_t word_count = (target_.node_count() + 63) / 64;
    domains_.assign(
        query_.node_count(),
        std::vector<std::uint64_t>(word_count, 0)
    );
    for (Position query_node = 0; query_node < query_.node_count(); ++query_node) {
        for (Position target_node = 0; target_node < target_.node_count(); ++target_node) {
            if (
                query_.degrees()[query_node] <= target_.degrees()[target_node]
                && compatibility_->may_accept_node(query_node, target_node)
            ) {
                domains_[query_node][target_node / 64] |= (
                    std::uint64_t{1} << (target_node % 64)
                );
            }
        }
    }
}

NativeProblem::~NativeProblem() = default;

bool NativeProblem::has_anchored_match(const Position target_node) const {
    const auto found = std::lower_bound(
        query_.keys().begin(),
        query_.keys().end(),
        std::int64_t{0}
    );
    if (found == query_.keys().end() || *found != 0) {
        throw std::invalid_argument(
            "recursive native query must contain anchor key 0"
        );
    }
    const Position query_node = static_cast<Position>(found - query_.keys().begin());
    MappingCursor cursor(
        shared_from_this(),
        std::make_pair(query_node, target_node)
    );
    return cursor.next_mapping().has_value();
}

}  // namespace hotpot::search
