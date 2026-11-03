#include "production.hpp"
#include "../../../../rules/chemistry/substructure/_native/predicates.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>


namespace hotpot::search {

namespace {

enum class LeafKind : std::uint8_t {
    NativeSet,
    HostSet,
    Callable,
    AtomExpression,
    BondExpression,
};

using HostFields = std::unordered_map<std::string, py::object>;

}  // namespace

struct NativeTarget::Data {
    std::vector<TargetAtom> atoms;
    std::vector<TargetBond> bonds;
    // Host slots remain aligned with the pure records and retain original handles.
    std::vector<HostFields> atom_host_fields;
    std::vector<HostFields> bond_host_fields;
    std::vector<py::object> atom_objects;
    std::vector<py::object> bond_objects;
};

namespace {

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
    SemanticRecord result;
    if (record.contains("degree")) {
        result.degree = py::cast<std::int64_t>(get_item(record, "degree"));
    }
    if (record.contains("connectivity")) {
        result.connectivity = py::cast<std::int64_t>(
            get_item(record, "connectivity")
        );
    }
    if (record.contains("valence")) {
        result.valence = py::cast<double>(get_item(record, "valence"));
    }
    if (record.contains("ring_count")) {
        result.ring_count = py::cast<std::int64_t>(
            get_item(record, "ring_count")
        );
    }
    if (record.contains("ring_sizes")) {
        result.ring_sizes = py::cast<std::vector<std::int64_t>>(
            get_item(record, "ring_sizes")
        );
    }
    return result;
}

template <typename Record>
void parse_fields(Record& record, HostFields& host_fields, const py::dict& fields) {
    for (const auto& item : fields) {
        const std::string name = string_value(item.first);
        const std::optional<NativeValue> value = native_value(item.second);
        if (value.has_value()) {
            record.native_fields.emplace(name, *value);
        } else {
            host_fields.emplace(
                name,
                py::reinterpret_borrow<py::object>(item.second)
            );
        }
    }
}

std::vector<TargetAtom> parse_target_atoms(
    const py::dict& target_spec, NativeTarget::Data& data
) {
    std::vector<TargetAtom> atoms;
    for (const py::handle item : get_item(target_spec, "atoms")) {
        const py::dict record = py::reinterpret_borrow<py::dict>(item);
        TargetAtom atom;
        data.atom_host_fields.emplace_back();
        parse_fields(
            atom,
            data.atom_host_fields.back(),
            py::reinterpret_borrow<py::dict>(get_item(record, "fields"))
        );
        atom.full = parse_semantic_record(
            py::reinterpret_borrow<py::dict>(get_item(record, "full"))
        );
        atom.ligand = parse_semantic_record(
            py::reinterpret_borrow<py::dict>(get_item(record, "ligand"))
        );
        data.atom_objects.push_back(
            py::reinterpret_borrow<py::object>(get_item(record, "object"))
        );
        atoms.push_back(std::move(atom));
    }
    return atoms;
}

std::vector<TargetBond> parse_target_bonds(
    const py::dict& target_spec, NativeTarget::Data& data
) {
    std::vector<TargetBond> bonds;
    for (const py::handle item : get_item(target_spec, "bonds")) {
        const py::dict record = py::reinterpret_borrow<py::dict>(item);
        TargetBond bond;
        data.bond_host_fields.emplace_back();
        parse_fields(
            bond,
            data.bond_host_fields.back(),
            py::reinterpret_borrow<py::dict>(get_item(record, "fields"))
        );
        bond.bond_kind = string_value(get_item(record, "bond_kind"));
        bond.is_aromatic = py::cast<bool>(get_item(record, "is_aromatic"));
        data.bond_objects.push_back(
            py::reinterpret_borrow<py::object>(get_item(record, "object"))
        );
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
    const std::shared_ptr<NativeTarget>& target,
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
        auto problem = std::make_shared<NativeProblem>(
            py::reinterpret_borrow<py::dict>(value),
            target
        );
        recursive->has_anchored_match = [problem = std::move(problem)](
            const std::uint32_t target_node
        ) { return problem->has_anchored_match(target_node); };
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
            target,
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
    const std::shared_ptr<NativeTarget>& target,
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
            target,
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
    const std::shared_ptr<NativeTarget>& target,
    const std::size_t target_node_count
) {
    std::vector<NodeProgram> programs;
    for (const py::handle item : get_item(query_spec, "nodes")) {
        const py::dict specification = py::reinterpret_borrow<py::dict>(item);
        NodeProgram program;
        program.preflight = parse_preflight(specification);
        for (const py::handle leaf : get_item(specification, "leaves")) {
            program.leaves.push_back(
                parse_leaf(leaf, target, target_node_count)
            );
        }
        programs.push_back(std::move(program));
    }
    return programs;
}

std::vector<BondProgram> parse_bond_programs(
    const py::dict& query_spec,
    const std::shared_ptr<NativeTarget>& target,
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
                parse_leaf(leaf, target, target_node_count)
            );
        }
        programs.push_back(std::move(program));
    }
    return programs;
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
    const Target& target,
    const HostFields& host_fields,
    const py::object& object
) {
    for (const std::string& name : preflight) {
        if (
            target.native_fields.find(name) != target.native_fields.end()
            || host_fields.find(name) != host_fields.end()
        ) {
            continue;
        }
        py::gil_scoped_acquire acquire;
        if (!python_has_attribute(object, name)) {
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

bool evaluate_python_leaf(const Leaf& leaf, const py::object& object) {
    py::gil_scoped_acquire acquire;
    if (leaf.kind == LeafKind::Callable) {
        return python_truth(leaf.python_value(object));
    }
    if (!python_truth(leaf.python_value)) {
        return true;
    }
    return python_contains(
        leaf.python_value,
        python_attribute(object, leaf.field)
    );
}

class NativeCompatibility final : public Compatibility {
public:
    NativeCompatibility(
        const py::dict& query_spec,
        std::shared_ptr<NativeTarget> target,
        const std::size_t query_node_count,
        const std::size_t query_edge_count
    ) : target_(std::move(target)),
        nodes_(parse_node_programs(
            query_spec,
            target_,
            target_->graph().node_count()
        )),
        bonds_(parse_bond_programs(
            query_spec,
            target_,
            target_->graph().node_count()
        )) {
        if (
            nodes_.size() != query_node_count
            || bonds_.size() != query_edge_count
        ) {
            throw std::invalid_argument(
                "native search records must align with graph topology"
            );
        }
    }

    bool accept_node(const Position query, const Position target) const override {
        const NodeProgram& program = nodes_[query];
        const TargetAtom& atom = target_->data().atoms[target];
        if (!accept_preflight_matches(
                program.preflight, atom,
                target_->data().atom_host_fields[target],
                target_->data().atom_objects[target]
            )) {
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
                if (!evaluate_python_leaf(leaf, target_->data().atom_objects[target])) {
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
        const TargetBond& bond = target_->data().bonds[target];
        if (program.exclude_aromatic && bond.is_aromatic) {
            return false;
        }
        if (program.bond_kind.has_value() && *program.bond_kind != bond.bond_kind) {
            return false;
        }
        if (!accept_preflight_matches(
                program.preflight, bond,
                target_->data().bond_host_fields[target],
                target_->data().bond_objects[target]
            )) {
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
                if (!evaluate_python_leaf(leaf, target_->data().bond_objects[target])) {
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
        const TargetAtom& atom = target_->data().atoms[target];
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
        const TargetBond& bond = target_->data().bonds[target];
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
    std::shared_ptr<NativeTarget> target_;
    std::vector<NodeProgram> nodes_;
    std::vector<BondProgram> bonds_;
};

}  // namespace

NativeTarget::NativeTarget(const py::dict& target_spec) :
    graph_(graph_keys(target_spec), graph_edges(target_spec)),
    data_(std::make_unique<Data>()) {
    data_->atoms = parse_target_atoms(target_spec, *data_);
    data_->bonds = parse_target_bonds(target_spec, *data_);
    if (
        data_->atoms.size() != graph_.node_count()
        || data_->bonds.size() != graph_.edge_count()
    ) {
        throw std::invalid_argument(
            "native search records must align with graph topology"
        );
    }
}

NativeTarget::~NativeTarget() = default;

NativeProblem::NativeProblem(
    const py::dict& query_spec,
    std::shared_ptr<NativeTarget> target
) : query_(graph_keys(query_spec), graph_edges(query_spec)),
    target_(std::move(target)),
    plan_(build_query_plan(query_)),
    compatibility_(std::make_unique<NativeCompatibility>(
        query_spec,
        target_,
        query_.node_count(),
        query_.edge_count()
    )) {
    domains_ = build_candidate_domains(query_, target_->graph(), *compatibility_);
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
