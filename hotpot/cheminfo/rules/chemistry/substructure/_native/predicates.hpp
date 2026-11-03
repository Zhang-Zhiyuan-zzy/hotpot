#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace hotpot::search {

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

struct SemanticRecord {
    std::int64_t degree = 0;
    std::int64_t connectivity = 0;
    double valence = 0.0;
    std::int64_t ring_count = 0;
    std::vector<std::int64_t> ring_sizes;
};

struct TargetAtom {
    std::unordered_map<std::string, NativeValue> native_fields;
    SemanticRecord full;
    SemanticRecord ligand;
};

struct TargetBond {
    std::unordered_map<std::string, NativeValue> native_fields;
    std::string bond_kind;
    bool is_aromatic = false;
};

struct RecursiveProgram {
    std::function<bool(std::uint32_t)> has_anchored_match;
    mutable std::mutex cache_mutex;
    mutable std::vector<std::int8_t> cache;

    bool matches(std::uint32_t target_node) const;
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

bool native_equal(const NativeValue& first, const NativeValue& second) noexcept;
bool native_contains(
    const std::vector<NativeValue>& values, const NativeValue& candidate
) noexcept;
const NativeValue* find_native(
    const std::unordered_map<std::string, NativeValue>& fields,
    const std::string& name
) noexcept;
bool evaluate_atom_expression(
    const AtomExpression& expression, const TargetAtom& atom,
    std::uint32_t target_node
);
Truth may_evaluate_atom_expression(
    const AtomExpression& expression, const TargetAtom& atom
);
bool evaluate_bond_expression(
    const BondExpression& expression, const TargetBond& bond
) noexcept;

}  // namespace hotpot::search
