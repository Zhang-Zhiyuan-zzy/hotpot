#include "predicates.hpp"

#include <algorithm>

namespace hotpot::search {

bool RecursiveProgram::matches(const std::uint32_t target_node) const {
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        if (cache[target_node] >= 0) {
            return cache[target_node] != 0;
        }
    }
    const bool result = has_anchored_match(target_node);
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        cache[target_node] = result ? 1 : 0;
    }
    return result;
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
    const std::uint32_t target_node
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

}  // namespace hotpot::search
