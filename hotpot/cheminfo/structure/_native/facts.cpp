#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <structmember.h>

#include <array>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <memory>
#include <limits>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <vector>

// Seed probing, clean insertion and body transfer adapt CPython Objects/setobject.c
// (Python Software Foundation; see CPYTHON_LICENSE). Built against each target
// CPython's headers; this module does not support free-threaded or limited ABI.
#ifdef Py_GIL_DISABLED
#error "Hotpot factual seed undo requires a GIL-enabled CPython build"
#endif

namespace py = pybind11;

namespace {

enum class Name { Parent, Owner, Uid, UidOwner, Atoms, Topology, Table, Size, Columns, Sources, Presence, Status, Seeds, Lock, TransactionMol, Producer, Cells, Failed, Source, Rows, Uids, Fields, Values, Present, Producers, SourceOwner, Acquire, Release, Dependencies, Count };

// Only immutable declarations and Python code/type bindings live beyond a call.
struct Field {
    std::string name;
    std::uint32_t presence;
    std::vector<int> affected;
    int materialized = -1;
    std::uint64_t affected_mask = 0, materialized_mask = 0;
};

struct Configuration {
    py::object indices, writable, readonly, source_type, update_type, check_source;
    py::object transactions, failures, symbol_number, number_type, asarray, atom_type, eager, sparse;
    py::object foreign_error, detached_error, seed_dummy;
    std::array<py::object, static_cast<std::size_t>(Name::Count)> names;
    py::object table_type;
    std::array<py::object, static_cast<std::size_t>(Name::Count)> table_descriptors;
    std::array<Py_ssize_t, static_cast<std::size_t>(Name::Count)> table_offsets;
    std::vector<Field> fields;
    std::vector<int> coordinate_fields;
    int element_field = -1;
    int groups = 0;
    int explicit_source = 1;
    int current_status = 0, needs_recalc_status = 1;
    std::unordered_set<int> producers;
};

Configuration *config = nullptr;

[[noreturn]] void python_error() { throw py::error_already_set(); }

PyObject *name(Name value) { return config->names[static_cast<std::size_t>(value)].ptr(); }

py::object get(py::handle object, Name key) {
    Py_ssize_t offset = config->table_offsets[static_cast<std::size_t>(key)];
    if (offset >= 0) {
        if (!PyObject_TypeCheck(object.ptr(), reinterpret_cast<PyTypeObject *>(config->table_type.ptr())))
            throw py::type_error("Atom storage requires the declared fact-table type");
        PyObject *value = *reinterpret_cast<PyObject **>(reinterpret_cast<char *>(object.ptr()) + offset);
        if (!value) { PyErr_SetObject(PyExc_AttributeError, name(key)); python_error(); }
        return py::reinterpret_borrow<py::object>(value);
    }
    PyObject *value = PyObject_GetAttr(object.ptr(), name(key));
    if (!value) python_error();
    return py::reinterpret_steal<py::object>(value);
}

py::object dictionary(py::handle object) {
    PyObject *value = PyObject_GenericGetDict(object.ptr(), nullptr);
    if (!value) python_error();
    return py::reinterpret_steal<py::object>(value);
}

py::handle item(py::handle dict, Name key) {
    PyObject *value = PyDict_GetItemWithError(dict.ptr(), name(key));
    if (!value) {
        if (!PyErr_Occurred()) PyErr_SetObject(PyExc_AttributeError, name(key));
        python_error();
    }
    return value;
}

bool equal(py::handle first, py::handle second) {
    if (first.is(second)) return true;
    int result = PyObject_RichCompareBool(first.ptr(), second.ptr(), Py_EQ);
    if (result < 0) python_error();
    return result != 0;
}

Py_ssize_t integer(py::handle value) {
    py::object indexed = py::reinterpret_steal<py::object>(PyNumber_Index(value.ptr()));
    if (!indexed) python_error();
    Py_ssize_t result = PyLong_AsSsize_t(indexed.ptr());
    if (result == -1 && PyErr_Occurred()) python_error();
    return result;
}

struct OwnerLock {
    py::object lock;
    explicit OwnerLock(py::handle mol) : lock(get(mol, Name::Lock)) {
        PyObject *result = PyObject_CallMethodNoArgs(lock.ptr(), name(Name::Acquire));
        if (!result) python_error();
        Py_DECREF(result);
    }
    ~OwnerLock() {
        PyObject *type, *value, *trace;
        PyErr_Fetch(&type, &value, &trace);
        PyObject *result = PyObject_CallMethodNoArgs(lock.ptr(), name(Name::Release));
        Py_XDECREF(result);
        if (!result) PyErr_Clear();
        PyErr_Restore(type, value, trace);
    }
};

py::object current_transaction(py::handle mol) {
    PyObject *value = nullptr;
    if (PyContextVar_Get(config->transactions.ptr(), nullptr, &value) < 0) python_error();
    py::object stack = py::reinterpret_steal<py::object>(value);
    for (Py_ssize_t index = PyTuple_GET_SIZE(stack.ptr()); index-- > 0;) {
        py::handle tx = PyTuple_GET_ITEM(stack.ptr(), index);
        if (get(tx, Name::TransactionMol).is(mol)) return py::reinterpret_borrow<py::object>(tx);
    }
    return py::none();
}

void poison(py::handle transaction) {
    if (!transaction.is_none()) {
        if (PyObject_SetAttr(transaction.ptr(), name(Name::Failed), Py_True) < 0) python_error();
    }
}

// Fault state is a ContextVar value installed only by the private test context.
// A checkpoint sets a builtin exception directly; it never calls a Python hook.
struct Failpoint {
    py::object state;
    std::string boundary;
    Py_ssize_t remaining = 0;
    PyObject *exception = nullptr;
    Failpoint() {
        PyObject *value = nullptr;
        if (PyContextVar_Get(config->failures.ptr(), nullptr, &value) < 0) python_error();
        state = py::reinterpret_steal<py::object>(value);
        if (!state.is_none()) {
            boundary = py::cast<std::string>(state[py::int_(0)]);
            remaining = integer(state[py::int_(1)]);
            exception = state[py::int_(2)].ptr();
        }
    }
    void after(const char *name) {
        if (exception && boundary == name && --remaining == 0) {
            PyErr_SetString(exception, "injected after native fact store");
            python_error();
        }
    }
};

template <class T, std::size_t Inline> struct SmallArray {
    std::array<T, Inline> local;
    std::vector<T> dynamic;
    std::size_t size = 0;
    void resize(std::size_t count) {
        size = count;
        if (count > Inline) dynamic.resize(count);
    }
    void push_back(T value) {
        if (size < Inline) local[size++] = std::move(value);
        else {
            if (size == Inline) {
                dynamic.reserve(Inline * 2);
                for (auto &entry : local) dynamic.push_back(std::move(entry));
            }
            dynamic.push_back(std::move(value));
            size = dynamic.size();
        }
    }
    T &operator[](std::size_t index) { return size <= Inline ? local[index] : dynamic[index]; }
    const T &operator[](std::size_t index) const { return size <= Inline ? local[index] : dynamic[index]; }
};

struct Cell {
    char *value, *source;
    std::uint64_t before, after;
    std::uint8_t old_source, new_source;
};
struct Presence { char *target; std::uint32_t before, after; };
struct Status { char *target; std::uint8_t before, after; };
bool active_key(PyObject *key) { return key && key != config->seed_dummy.ptr(); }

// Preserve headers, weakrefs and the identity of the actual Python set owner.
// Inline contents must return to that owner's smalltable before older actions.
void swap_seed_bodies(PySetObject *first, PySetObject *second) noexcept {
    bool first_inline = first->table == first->smalltable;
    bool second_inline = second->table == second->smalltable;
    std::swap(first->fill, second->fill);
    std::swap(first->used, second->used);
    std::swap(first->mask, second->mask);
    std::swap(first->hash, second->hash);
    std::swap(first->finger, second->finger);
    std::swap(first->table, second->table);
    if (first_inline || second_inline) {
        setentry temporary[PySet_MINSIZE];
        std::memcpy(temporary, first->smalltable, sizeof(temporary));
        std::memcpy(first->smalltable, second->smalltable, sizeof(temporary));
        std::memcpy(second->smalltable, temporary, sizeof(temporary));
    }
    if (second_inline) first->table = first->smalltable;
    if (first_inline) second->table = second->smalltable;
}

struct SeedAction {
    PySetObject *set = nullptr;
    py::object candidate, before_owner, after_owner;
    Py_ssize_t index = 0, old_fill = 0, old_used = 0, new_fill = 0, new_used = 0;
    setentry before{}, after{};
    const char *boundary = nullptr;
    bool applied = false;

    void write(setentry value, Py_ssize_t fill, Py_ssize_t used) noexcept {
        // Only exact builtin integer keys are admitted by preparation.
        auto &entry = set->table[index];
        if (active_key(value.key)) Py_INCREF(value.key);
        PyObject *old = entry.key;
        entry = value;
        set->fill = fill;
        set->used = used;
        if (active_key(old)) Py_DECREF(old);
    }
    void install() noexcept {
        if (candidate) swap_seed_bodies(set, reinterpret_cast<PySetObject *>(candidate.ptr()));
        else write(after, new_fill, new_used);
        applied = true;
    }
    void restore() noexcept {
        if (!applied) return;
        if (candidate) swap_seed_bodies(set, reinterpret_cast<PySetObject *>(candidate.ptr()));
        else write(before, old_fill, old_used);
        applied = false;
    }
};

// A short scalar path allocates no overlay; larger batches use expected O(k)
// operation-local metadata, never a copy of the entire current seed set.
struct SeedOverlay {
    std::array<std::pair<Py_ssize_t, setentry>, 8> local{};
    std::size_t size = 0;
    std::unordered_map<Py_ssize_t, setentry> dynamic;
    setentry get(PySetObject *set, Py_ssize_t index) const {
        if (size <= local.size()) {
            for (std::size_t i = 0; i < size; ++i)
                if (local[i].first == index) return local[i].second;
        } else {
            auto found = dynamic.find(index);
            if (found != dynamic.end()) return found->second;
        }
        return set->table[index];
    }
    void put(Py_ssize_t index, setentry value) {
        if (size <= local.size()) {
            for (std::size_t i = 0; i < size; ++i) {
                if (local[i].first == index) { local[i].second = value; return; }
            }
            if (size < local.size()) { local[size++] = {index, value}; return; }
            dynamic.reserve(local.size() * 2);
            for (const auto &entry : local) dynamic.emplace(entry);
        }
        dynamic[index] = value;
        size = dynamic.size();
    }
    void clear() { size = 0; dynamic.clear(); }
};

// Same unsigned probe sequence and last-dummy choice as CPython 3.10–3.14.
// This closed lookup accepts builtin UID integers only, before any stores.
struct SeedPreparation {
    SmallArray<SeedAction, 8> &actions;
    PySetObject *active, *generation;
    Py_ssize_t fill, used;
    SeedOverlay overlay;

    SeedPreparation(SmallArray<SeedAction, 8> &actions, py::handle set)
        : actions(actions) {
        if (!PySet_CheckExact(set.ptr())) throw py::type_error("Active dependency seeds require builtin sets");
        active = generation = reinterpret_cast<PySetObject *>(set.ptr());
        fill = active->fill; used = active->used;
    }

    std::pair<Py_ssize_t, bool> lookup(PyObject *uid, Py_hash_t hash) const {
        std::size_t mask = generation->mask, perturb = static_cast<std::size_t>(hash);
        std::size_t index = perturb & mask;
        Py_ssize_t dummy = -1;
        for (;;) {
            int probes = index + 9 <= mask ? 9 : 0;
            for (int offset = 0; offset <= probes; ++offset) {
                Py_ssize_t bucket = static_cast<Py_ssize_t>(index + offset);
                auto entry = overlay.get(generation, bucket);
                if (!entry.key && entry.hash == 0) return {dummy < 0 ? bucket : dummy, false};
                if (entry.hash == -1) dummy = bucket;
                else if (entry.hash == hash) {
                    if (!PyLong_CheckExact(entry.key)) throw py::type_error("Dependency seed keys require builtin integer UIDs");
                    if (entry.key == uid || equal(py::handle(entry.key), py::handle(uid))) return {bucket, true};
                }
            }
            perturb >>= 5;
            index = (index * 5 + 1 + perturb) & mask;
        }
    }

    static void insert_clean(PySetObject *set, setentry value) noexcept {
        std::size_t mask = set->mask, perturb = static_cast<std::size_t>(value.hash);
        std::size_t index = perturb & mask;
        for (;;) {
            int probes = index + 9 <= mask ? 9 : 0;
            for (int offset = 0; offset <= probes; ++offset) {
                auto &entry = set->table[index + offset];
                if (!entry.key) { entry = value; Py_INCREF(value.key); return; }
            }
            perturb >>= 5;
            index = (index * 5 + 1 + perturb) & mask;
        }
    }

    py::object resized() const {
        py::object candidate = py::reinterpret_steal<py::object>(PySet_New(nullptr));
        if (!candidate) python_error();
        auto *set = reinterpret_cast<PySetObject *>(candidate.ptr());
        std::size_t minimum = static_cast<std::size_t>(used) * (used > 50000 ? 2 : 4);
        std::size_t count = PySet_MINSIZE;
        while (count <= minimum) count <<= 1;
        if (count > PySet_MINSIZE) {
            auto *buckets = PyMem_New(setentry, count);
            if (!buckets) { PyErr_NoMemory(); python_error(); }
            std::memset(buckets, 0, sizeof(setentry) * count);
            set->table = buckets;
            set->mask = static_cast<Py_ssize_t>(count - 1);
        }
        // Copy only at a genuine resize; preserve the old physical buckets by
        // the later body swap, instead of reconstructing them during rollback.
        for (Py_ssize_t index = 0; index <= generation->mask; ++index) {
            auto entry = overlay.get(generation, index);
            if (active_key(entry.key)) {
                if (!PyLong_CheckExact(entry.key)) throw py::type_error("Dependency seed keys require builtin integer UIDs");
                insert_clean(set, entry);
                ++set->fill; ++set->used;
            }
        }
        return candidate;
    }

    void change(py::handle uid, bool present) {
        if (!PyLong_CheckExact(uid.ptr())) throw py::type_error("Dependency seeds require builtin integer UIDs");
        Py_hash_t hash = PyObject_Hash(uid.ptr());
        if (hash == -1) python_error();
        auto found = lookup(uid.ptr(), hash);
        if (found.second == present) return;
        auto before = overlay.get(generation, found.first);
        SeedAction action;
        action.set = active; action.index = found.first;
        action.old_fill = fill; action.old_used = used; action.before = before;
        action.after = present ? setentry{uid.ptr(), hash} : setentry{config->seed_dummy.ptr(), -1};
        action.boundary = present ? "seed_add" : "seed_remove";
        if (active_key(before.key)) action.before_owner = py::reinterpret_borrow<py::object>(before.key);
        if (present) action.after_owner = py::reinterpret_borrow<py::object>(uid);
        used += present ? 1 : -1;
        if (present && !before.key) ++fill;
        action.new_fill = fill; action.new_used = used;
        overlay.put(found.first, action.after);
        if (present && !before.key && static_cast<std::size_t>(fill) * 5 >= static_cast<std::size_t>(generation->mask) * 3) {
            action.candidate = resized();
            generation = reinterpret_cast<PySetObject *>(action.candidate.ptr());
            fill = generation->fill; used = generation->used;
            overlay.clear();
        }
        actions.push_back(std::move(action));
    }

    void clear() {
        SeedAction action;
        action.set = active; action.boundary = "seed_clear";
        action.candidate = py::reinterpret_steal<py::object>(PySet_New(nullptr));
        if (!action.candidate) python_error();
        actions.push_back(std::move(action));
    }
};

struct Undo {
    // These are the original physical owners, including after table remaps.
    std::array<py::object, 6> owners;
    SmallArray<Cell, 19> cells;
    SmallArray<Presence, 1> presence;
    SmallArray<Status, 4> statuses;
    SmallArray<SeedAction, 8> seeds;
    bool restored = false;

    void restore() noexcept {
        if (restored) return;
        for (std::size_t i = seeds.size; i-- > 0;) seeds[i].restore();
        for (std::size_t i = 0; i < cells.size; ++i) {
            auto &cell = cells[i];
            std::memcpy(cell.value, &cell.before, sizeof(cell.before));
            std::memcpy(cell.source, &cell.old_source, sizeof(cell.old_source));
        }
        for (std::size_t i = 0; i < presence.size; ++i) {
            auto &entry = presence[i];
            std::memcpy(entry.target, &entry.before, sizeof(entry.before));
        }
        for (std::size_t i = 0; i < statuses.size; ++i) {
            auto &entry = statuses[i];
            std::memcpy(entry.target, &entry.before, sizeof(entry.before));
        }
        restored = true;
    }

    void install(Failpoint &fault) {
        for (std::size_t i = 0; i < cells.size; ++i) {
            auto &cell = cells[i];
            std::memcpy(cell.value, &cell.after, sizeof(cell.after));
            fault.after("value");
            std::memcpy(cell.source, &cell.new_source, sizeof(cell.new_source));
            fault.after("source");
        }
        for (std::size_t i = 0; i < presence.size; ++i) {
            auto &entry = presence[i];
            std::memcpy(entry.target, &entry.after, sizeof(entry.after));
            fault.after("presence");
        }
        for (std::size_t i = 0; i < statuses.size; ++i) {
            auto &entry = statuses[i];
            std::memcpy(entry.target, &entry.after, sizeof(entry.after));
            fault.after("status");
        }
        for (std::size_t i = 0; i < seeds.size; ++i) {
            auto &entry = seeds[i];
            entry.install();
            fault.after(entry.boundary);
        }
    }
};

py::array array(py::handle value, char kind, Py_ssize_t itemsize, int dimensions, bool writable) {
    if (!py::isinstance<py::array>(value)) throw py::type_error("Atom storage must be NumPy arrays");
    py::array result = py::reinterpret_borrow<py::array>(value);
    auto dtype = result.dtype();
    if (result.ndim() != dimensions || dtype.kind() != kind || result.itemsize() != itemsize
        || (dtype.byteorder() != '=' && dtype.byteorder() != '|'
            && dtype.byteorder() != ([] { std::uint16_t n = 1; return *reinterpret_cast<char *>(&n) ? '<' : '>'; })()))
        throw py::value_error("Atom storage has incompatible dtype or dimensions");
    if (writable && !result.writeable()) throw py::value_error("Atom update targets read-only storage");
    return result;
}

char *address(const py::array &value, Py_ssize_t row, Py_ssize_t column = 0) {
    return static_cast<char *>(const_cast<void *>(value.data())) + row * value.strides(0)
           + (value.ndim() == 2 ? column * value.strides(1) : 0);
}

struct Column {
    int index;
    py::object values;
    bool present;
    int producer;
    std::uint64_t immediate_bits = 0;
    bool immediate = false;
};

struct Selection {
    std::vector<Py_ssize_t> rows;
    py::tuple uids;
    py::object owner;
};

void check_member(py::handle mol, py::handle atoms, py::handle topology, py::handle owner,
                   Py_ssize_t row, py::handle uid, PyObject *expected = nullptr) {
    if (row < 0 || row >= PyList_GET_SIZE(atoms.ptr()))
        throw py::value_error("Atom update member is no longer current");
    py::handle atom = PyList_GET_ITEM(atoms.ptr(), row);
    if (expected && atom.ptr() != expected) throw py::value_error("Atom update member is no longer current");
    py::object ad = dictionary(atom);
    py::object key = py::reinterpret_steal<py::object>(PyLong_FromVoidPtr(atom.ptr()));
    PyObject *entry = PyDict_GetItemWithError(topology.ptr(), key.ptr());
    if (!entry && PyErr_Occurred()) python_error();
    if (!item(ad, Name::Parent).is(mol) || !item(ad, Name::UidOwner).is(owner)
        || !equal(item(ad, Name::Uid), uid)
        || !entry || integer(py::handle(PyTuple_GET_ITEM(entry, 0))) != row)
        throw py::value_error("Atom update member or UID is no longer current");
}

void check_rows(py::handle mol, const Selection &selection, py::handle expected = {}) {
    py::object md = dictionary(mol);
    if (!item(md, Name::Owner).is(selection.owner))
        throw std::runtime_error("Molecular source owner changed; recalculate before committing");
    py::handle atoms = item(md, Name::Atoms), topology = item(md, Name::Topology);
    for (std::size_t i = 0; i < selection.rows.size(); ++i)
        check_member(mol, atoms, topology, selection.owner, selection.rows[i], PyTuple_GET_ITEM(selection.uids.ptr(), i),
                     expected ? PyTuple_GET_ITEM(expected.ptr(), i) : nullptr);
}

template <class Rows, class Columns>
void prepare_undo(Undo &undo, py::handle table, const Rows &rows,
                  const Columns &columns, py::handle uids, bool dependencies) {
    undo.owners[0] = get(table, Name::Columns);
    undo.owners[1] = get(table, Name::Sources);
    undo.owners[2] = get(table, Name::Presence);
    auto sources = array(undo.owners[1], 'u', 1, 2, true);
    auto presence = array(undo.owners[2], 'u', 4, 1, true);
    Py_ssize_t logical = integer(get(table, Name::Size));
    if (sources.shape(0) < logical || sources.shape(1) != static_cast<Py_ssize_t>(config->fields.size())
        || presence.shape(0) < logical)
        throw py::value_error("Atom metadata storage is not aligned");
    for (auto row : rows)
        if (row < 0 || row >= logical) throw py::index_error("Atom row is outside the current table");
    undo.presence.resize(rows.size());
    for (std::size_t r = 0; r < rows.size(); ++r) {
        auto &p = undo.presence[r];
        p.target = address(presence, rows[r]);
        std::memcpy(&p.before, p.target, sizeof(p.before));
        p.after = p.before;
    }
    undo.cells.resize(rows.size() * columns.size());
    std::uint64_t affected = 0, materialized = 0;
    for (std::size_t f = 0; f < columns.size(); ++f) {
        const auto &column = columns[f];
        if (column.index < 0 || column.index >= static_cast<int>(config->fields.size()))
            throw py::value_error("Atom update requires canonical fields");
        const auto &field = config->fields[column.index];
        auto values = array(PyTuple_GET_ITEM(undo.owners[0].ptr(), column.index), 'f', 8, 1, true);
        if (values.shape(0) < logical) throw py::value_error("Atom value storage is not aligned");
        if (!column.immediate && PyBytes_GET_SIZE(column.values.ptr()) != static_cast<Py_ssize_t>(rows.size() * sizeof(double)))
            throw py::value_error("Atom update bytes must align with selected rows");
        const char *input = column.immediate ? reinterpret_cast<const char *>(&column.immediate_bits)
                                            : PyBytes_AS_STRING(column.values.ptr());
        for (std::size_t r = 0; r < rows.size(); ++r) {
            auto &cell = undo.cells[f * rows.size() + r];
            cell.value = address(values, rows[r]);
            cell.source = address(sources, rows[r], column.index);
            std::memcpy(&cell.before, cell.value, sizeof(cell.before));
            std::memcpy(&cell.after, input + r * sizeof(double), sizeof(cell.after));
            std::memcpy(&cell.old_source, cell.source, sizeof(cell.old_source));
            cell.new_source = static_cast<std::uint8_t>(column.producer);
            auto &p = undo.presence[r];
            p.after = column.present ? p.after | field.presence : p.after & ~field.presence;
        }
        affected |= field.affected_mask;
        materialized |= field.materialized_mask;
    }
    if (!dependencies || !(affected | materialized)) return;
    undo.owners[3] = get(table, Name::Status);
    undo.owners[4] = get(table, Name::Seeds);
    undo.owners[5] = py::reinterpret_borrow<py::object>(uids);
    auto statuses = array(undo.owners[3], 'u', 1, 2, true);
    if (statuses.shape(0) < logical || statuses.shape(1) != config->groups)
        throw py::value_error("Atom dependency storage is not aligned");
    std::vector<Status> status_changes;
    for (int group = 0; group < config->groups; ++group) {
        std::uint64_t bit = std::uint64_t{1} << group;
        if (!((affected | materialized) & bit)) continue;
        bool makes_current = materialized & bit;
        SeedPreparation seeds(undo.seeds, PyTuple_GET_ITEM(undo.owners[4].ptr(), group));
        for (std::size_t r = 0; r < rows.size(); ++r) {
            py::handle uid = PyTuple_Check(uids.ptr()) ? PyTuple_GET_ITEM(uids.ptr(), r) : uids.ptr();
            seeds.change(uid, !makes_current);
            if (makes_current) {
                char *target = address(statuses, rows[r], group);
                auto before = static_cast<std::uint8_t>(*target);
                if (before != config->current_status)
                    status_changes.push_back({target, before, static_cast<std::uint8_t>(config->current_status)});
            }
        }
    }
    undo.statuses.resize(status_changes.size());
    for (std::size_t i = 0; i < status_changes.size(); ++i) undo.statuses[i] = status_changes[i];
}

void install_undo(Undo &local, py::handle transaction) {
    Undo *undo = &local;
    py::object capsule;
    Failpoint fault;
    try {
        if (!transaction.is_none()) {
            auto owned = std::make_unique<Undo>();
            capsule = py::capsule(owned.get(), "hotpot.atom_fact_undo", [](PyObject *object) {
                delete static_cast<Undo *>(PyCapsule_GetPointer(object, "hotpot.atom_fact_undo"));
            });
            *owned = std::move(local);
            undo = owned.release();
            py::object cells = get(transaction, Name::Cells);
            if (PyList_Append(cells.ptr(), capsule.ptr()) < 0) python_error();
            fault.after("enrollment");
        }
        undo->install(fault);
    } catch (...) {
        auto error = std::current_exception();
        undo->restore();
        poison(transaction);
        std::rethrow_exception(error);
    }
}

template <class Rows, class Columns>
void install(py::handle table, const Rows &rows,
             const Columns &columns, py::handle uids, py::handle transaction,
             bool dependencies) {
    try {
        Undo undo;
        prepare_undo(undo, table, rows, columns, uids, dependencies);
        install_undo(undo, transaction);
    } catch (...) { poison(transaction); throw; }
}

double scalar_value(py::handle input) {
    if (PyFloat_CheckExact(input.ptr())) return PyFloat_AS_DOUBLE(input.ptr());
    if (PyLong_CheckExact(input.ptr())) {
        double value = PyLong_AsDouble(input.ptr());
        if (value == -1 && PyErr_Occurred()) python_error();
        return value;
    }
    int numeric = PyObject_IsInstance(input.ptr(), config->number_type.ptr());
    if (numeric < 0) python_error();
    if (PyFloat_Check(input.ptr()) || PyLong_Check(input.ptr()) || numeric) {
        py::object value = py::reinterpret_steal<py::object>(PyNumber_Float(input.ptr()));
        if (!value) python_error();
        return PyFloat_AS_DOUBLE(value.ptr());
    }
    py::array_t<double, py::array::forcecast> candidate(py::reinterpret_borrow<py::object>(input));
    if (candidate.ndim() != 0 && !(candidate.ndim() == 1 && candidate.size() == 1))
        throw py::value_error("A single-row Atom update requires a scalar or one value");
    double value;
    std::memcpy(&value, candidate.data(), sizeof(value));
    return value;
}

py::bytes scalar_bytes(py::handle input) {
    double value = scalar_value(input);
    return py::bytes(reinterpret_cast<const char *>(&value), sizeof(value));
}

py::bytes column_bytes(py::handle input, std::size_t count, const std::string &field) {
    py::array_t<double, py::array::forcecast> candidate(py::reinterpret_borrow<py::object>(input));
    if (candidate.ndim() != 0 && !(candidate.ndim() == 1 && candidate.size() == static_cast<Py_ssize_t>(count)))
        throw py::value_error("Atom field '" + field + "' requires a scalar or aligned column");
    py::bytes result = py::reinterpret_steal<py::bytes>(PyBytes_FromStringAndSize(nullptr, count * sizeof(double)));
    if (!result) python_error();
    char *out = PyBytes_AS_STRING(result.ptr());
    for (std::size_t i = 0; i < count; ++i)
        std::memcpy(out + i * sizeof(double), candidate.ndim() == 0 ? static_cast<const void *>(candidate.data())
                                                                 : static_cast<const void *>(address(candidate, i)), sizeof(double));
    return result;
}

py::object make_update(py::handle source, py::tuple rows, py::tuple uids,
                       const std::vector<Column> &columns) {
    auto *type = reinterpret_cast<PyTypeObject *>(config->update_type.ptr());
    py::object result = py::reinterpret_steal<py::object>(type->tp_alloc(type, 0));
    if (!result) python_error();
    py::tuple fields(columns.size()), values(columns.size()), presence(columns.size()), codes(columns.size());
    for (std::size_t i = 0; i < columns.size(); ++i) {
        fields[i] = py::str(config->fields[columns[i].index].name);
        values[i] = columns[i].values;
        presence[i] = py::bool_(columns[i].present);
        codes[i] = py::int_(columns[i].producer);
    }
    const char *names[] = {"source", "rows", "atom_uids", "fields", "value_bytes", "presence", "sources"};
    py::handle entries[] = {source, rows, uids, fields, values, presence, codes};
    for (int i = 0; i < 7; ++i) {
        py::str name(names[i]);
        if (PyObject_GenericSetAttr(result.ptr(), name.ptr(), entries[i].ptr()) < 0) python_error();
    }
    return result;
}

py::object make_source(py::handle owner) {
    auto *type = reinterpret_cast<PyTypeObject *>(config->source_type.ptr());
    py::object result = py::reinterpret_steal<py::object>(type->tp_alloc(type, 0));
    if (!result) python_error();
    py::tuple dependencies;
    if (PyObject_GenericSetAttr(result.ptr(), name(Name::SourceOwner), owner.ptr()) < 0
        || PyObject_GenericSetAttr(result.ptr(), name(Name::Dependencies), dependencies.ptr()) < 0)
        python_error();
    return result;
}

std::vector<Py_ssize_t> row_values(py::handle rows) {
    std::vector<Py_ssize_t> result;
    for (auto item : py::reinterpret_borrow<py::iterable>(rows)) result.push_back(integer(item));
    return result;
}

int field_index(py::handle field) {
    PyObject *index = PyDict_GetItemWithError(config->indices.ptr(), field.ptr());
    if (!index) {
        if (PyErr_Occurred()) python_error();
        PyErr_SetObject(PyExc_KeyError, field.ptr());
        python_error();
    }
    return static_cast<int>(integer(index));
}

bool same_column(py::bytes first, py::bytes second) {
    if (PyBytes_GET_SIZE(first.ptr()) != PyBytes_GET_SIZE(second.ptr())) return false;
    Py_ssize_t size = PyBytes_GET_SIZE(first.ptr());
    const char *a = PyBytes_AS_STRING(first.ptr()), *b = PyBytes_AS_STRING(second.ptr());
    if (std::memcmp(a, b, size) == 0) return true;
    for (Py_ssize_t i = 0; i < size; i += sizeof(double)) {
        double x, y;
        std::memcpy(&x, a + i, sizeof(x)); std::memcpy(&y, b + i, sizeof(y));
        if (!(x == y || (std::isnan(x) && std::isnan(y)))) return false;
    }
    return true;
}

std::vector<Column> normalize_columns(py::dict supplied, std::size_t count,
                                      bool public_fields, int producer, bool present) {
    for (auto pair : supplied) {
        if (public_fields) {
            int writable = PySet_Contains(config->writable.ptr(), pair.first.ptr());
            if (writable < 0) python_error();
            if (!writable) {
                int readonly = PySet_Contains(config->readonly.ptr(), pair.first.ptr());
                if (readonly < 0) python_error();
                if (readonly) throw py::attribute_error("Atom field is not writable through updata_atoms");
                throw py::type_error("Unknown Atom update field");
            }
        } else if (!PyDict_Contains(config->indices.ptr(), pair.first.ptr())) {
            throw py::type_error("Internal Atom updates require canonical fields");
        }
    }
    std::vector<py::object> encoded(config->fields.size());
    auto add = [&](int index, py::bytes value) {
        if (encoded[index] && !same_column(py::reinterpret_borrow<py::bytes>(encoded[index]), value))
            throw py::value_error("Conflicting Atom update aliases for '" + config->fields[index].name + "'");
        encoded[index] = std::move(value);
    };
    for (auto pair : supplied) {
        std::string name = py::cast<std::string>(pair.first);
        if (public_fields && name == "symbol") {
            py::array symbols = py::cast<py::array>(config->asarray(pair.second));
            int index = config->element_field;
            if (symbols.ndim() == 0) {
                py::object value = config->symbol_number(symbols.attr("item")());
                add(index, column_bytes(value, count, config->fields[index].name));
            } else if (symbols.ndim() == 1 && symbols.size() == static_cast<Py_ssize_t>(count)) {
                py::bytes bytes = py::reinterpret_steal<py::bytes>(PyBytes_FromStringAndSize(nullptr, count * sizeof(double)));
                if (!bytes) python_error();
                for (std::size_t row = 0; row < count; ++row) {
                    double value = scalar_value(config->symbol_number(symbols[py::int_(row)]));
                    std::memcpy(PyBytes_AS_STRING(bytes.ptr()) + row * sizeof(double), &value, sizeof(value));
                }
                add(index, std::move(bytes));
            } else throw py::value_error("Atom symbols require a scalar or aligned column");
        } else if (public_fields && name == "coordinates") {
            py::array_t<double, py::array::forcecast> coordinates(py::reinterpret_borrow<py::object>(pair.second));
            Py_ssize_t width = static_cast<Py_ssize_t>(config->coordinate_fields.size());
            bool broadcast = coordinates.ndim() == 1 && coordinates.shape(0) == width;
            if (!broadcast && !(coordinates.ndim() == 2 && coordinates.shape(0) == static_cast<Py_ssize_t>(count)
                                && coordinates.shape(1) == width))
                throw py::value_error("Atom coordinates require shape (3,) or (N, 3)");
            for (Py_ssize_t axis = 0; axis < width; ++axis) {
                py::bytes bytes = py::reinterpret_steal<py::bytes>(PyBytes_FromStringAndSize(nullptr, count * sizeof(double)));
                if (!bytes) python_error();
                for (std::size_t row = 0; row < count; ++row)
                    std::memcpy(PyBytes_AS_STRING(bytes.ptr()) + row * sizeof(double),
                                broadcast ? address(coordinates, axis) : address(coordinates, row, axis), sizeof(double));
                add(config->coordinate_fields[axis], std::move(bytes));
            }
        } else {
            int index = field_index(pair.first);
            add(index, column_bytes(pair.second, count, name));
        }
    }
    std::vector<Column> result;
    for (std::size_t index = 0; index < encoded.size(); ++index)
        if (encoded[index]) result.push_back({static_cast<int>(index), py::reinterpret_borrow<py::bytes>(encoded[index]), present, producer});
    return result;
}

py::tuple integer_tuple(py::iterable input) {
    py::list values;
    for (py::handle value : input) {
        py::object index = py::reinterpret_steal<py::object>(PyNumber_Index(value.ptr()));
        if (!index) python_error();
        if (PyObject_RichCompareBool(index.ptr(), py::int_(0).ptr(), Py_LT) != 0)
            throw py::value_error("Atom updates require nonnegative rows and UIDs");
        values.append(index);
    }
    py::tuple result(values);
    py::set unique(result);
    if (py::len(unique) != py::len(result)) throw py::value_error("Atom updates require unique rows and UIDs");
    return result;
}

py::object normalize_facts(py::handle source, py::iterable raw_rows, py::iterable raw_uids,
                           py::dict columns, bool public_fields, int producer, bool present) {
    py::tuple rows = integer_tuple(raw_rows), uids = integer_tuple(raw_uids);
    if (rows.size() != uids.size()) throw py::value_error("Atom updates require aligned rows and UIDs");
    auto prepared = normalize_columns(columns, rows.size(), public_fields, producer, present);
    return make_update(source, rows, uids, prepared);
}

py::object normalize_scalar(py::handle source, py::handle row, py::handle uid,
                            py::handle field, py::handle value, int producer, bool present) {
    int index = field_index(field);
    std::vector<Column> columns{{index, scalar_bytes(value), present, producer}};
    return make_update(source, py::make_tuple(row), py::make_tuple(uid), columns);
}

void commit_scalar(py::handle atom, int index, py::handle value, int producer, bool present) {
    if (index < 0 || index >= static_cast<int>(config->fields.size())) throw py::value_error("Unknown canonical Atom field");
    py::object ad = dictionary(atom);
    py::object mol = py::reinterpret_borrow<py::object>(item(ad, Name::Parent));
    OwnerLock lock(mol);
    py::object transaction = current_transaction(mol);
    try {
        py::object md = dictionary(mol);
        py::object token = py::reinterpret_borrow<py::object>(item(md, Name::Owner));
        py::object uid = py::reinterpret_borrow<py::object>(item(ad, Name::Uid));
        py::object key = py::reinterpret_steal<py::object>(PyLong_FromVoidPtr(atom.ptr()));
        PyObject *entry = PyDict_GetItemWithError(item(md, Name::Topology).ptr(), key.ptr());
        if (!entry) { if (!PyErr_Occurred()) PyErr_SetObject(PyExc_KeyError, key.ptr()); python_error(); }
        Py_ssize_t row = integer(PyTuple_GET_ITEM(entry, 0));
        bool callback_free = PyFloat_CheckExact(value.ptr()) || PyLong_CheckExact(value.ptr());
        const double number = scalar_value(value);
        if (!callback_free) md = dictionary(mol);
        if (!item(md, Name::Owner).is(token))
            throw std::runtime_error("Molecular source owner changed; recalculate before committing");
        if (callback_free) {
            py::handle atoms = item(md, Name::Atoms);
            if (row < 0 || row >= PyList_GET_SIZE(atoms.ptr()) || PyList_GET_ITEM(atoms.ptr(), row) != atom.ptr()
                || !item(ad, Name::Parent).is(mol) || !item(ad, Name::UidOwner).is(token)
                || !equal(item(ad, Name::Uid), uid))
                throw py::value_error("Atom update member or UID is no longer current");
            // entry is the sole-index binding just read under this lock; exact
            // builtin conversion cannot call Python or alter that binding.
        } else check_member(mol, item(md, Name::Atoms), item(md, Name::Topology), token, row, uid, atom.ptr());
        if (producer == config->explicit_source && !transaction.is_none()) producer = py::cast<int>(get(transaction, Name::Producer));
        std::array<Column, 1> columns{{{index, py::object(), present, producer, 0, true}}};
        std::memcpy(&columns[0].immediate_bits, &number, sizeof(number));
        const std::array<Py_ssize_t, 1> rows{{row}};
        install(item(md, Name::Table), rows, columns, uid, transaction, true);
    } catch (...) { poison(transaction); throw; }
}

py::object commit_columns(py::handle mol, py::object requested, py::dict supplied) {
    OwnerLock lock(mol);
    py::object transaction = current_transaction(mol);
    if (transaction.is_none()) throw std::runtime_error("Atom column commits require an outer factual transaction");
    try {
        py::object md = dictionary(mol);
        py::tuple atoms = requested.is_none() ? py::tuple(py::reinterpret_borrow<py::object>(item(md, Name::Atoms)))
                                             : py::cast<py::tuple>(requested);
        py::tuple rows(atoms.size()), uids(atoms.size());
        std::unordered_set<PyObject *> seen;
        py::object token = py::reinterpret_borrow<py::object>(item(md, Name::Owner));
        for (std::size_t i = 0; i < atoms.size(); ++i) {
            py::handle atom = atoms[i];
            py::object ad = dictionary(atom);
            if (requested.is_none()) {
                rows[i] = py::int_(i); uids[i] = item(ad, Name::Uid);
                continue;
            }
            if (!item(ad, Name::Parent).is(mol)) {
                py::object error = config->foreign_error(mol, atom);
                PyErr_SetObject(config->foreign_error.ptr(), error.ptr()); python_error();
            }
            py::object key = py::reinterpret_steal<py::object>(PyLong_FromVoidPtr(atom.ptr()));
            PyObject *entry = PyDict_GetItemWithError(item(md, Name::Topology).ptr(), key.ptr());
            if (!entry) {
                if (PyErr_Occurred()) python_error();
                py::object error = config->detached_error(atom, mol);
                PyErr_SetObject(config->detached_error.ptr(), error.ptr()); python_error();
            }
            Py_ssize_t row = integer(PyTuple_GET_ITEM(entry, 0));
            if (row < 0 || row >= PyList_GET_SIZE(item(md, Name::Atoms).ptr())
                || PyList_GET_ITEM(item(md, Name::Atoms).ptr(), row) != atom.ptr()
                || !item(ad, Name::UidOwner).is(token) || !seen.insert(atom.ptr()).second)
                throw py::value_error("Atom selection requires unique current members");
            rows[i] = py::int_(row); uids[i] = item(ad, Name::Uid);
        }
        auto columns = normalize_columns(supplied, atoms.size(), true, config->explicit_source, true);
        Selection selection{row_values(rows), uids, token};
        py::object source = make_source(token);
        py::object update = make_update(source, rows, uids, columns);
        check_rows(mol, selection, atoms);
        int producer = py::cast<int>(get(transaction, Name::Producer));
        for (auto &column : columns) column.producer = producer;
        install(get(mol, Name::Table), selection.rows, columns, selection.uids, transaction, true);
        return update;
    } catch (...) { poison(transaction); throw; }
}

void commit_fields(py::handle atom, py::dict supplied, int producer, bool present) {
    py::object ad = dictionary(atom);
    py::object mol = py::reinterpret_borrow<py::object>(item(ad, Name::Parent));
    OwnerLock lock(mol);
    py::object transaction = current_transaction(mol);
    try {
        py::object md = dictionary(mol);
        py::object token = py::reinterpret_borrow<py::object>(item(md, Name::Owner));
        py::object uid = py::reinterpret_borrow<py::object>(item(ad, Name::Uid));
        py::object key = py::reinterpret_steal<py::object>(PyLong_FromVoidPtr(atom.ptr()));
        PyObject *entry = PyDict_GetItemWithError(item(md, Name::Topology).ptr(), key.ptr());
        if (!entry) { if (!PyErr_Occurred()) PyErr_SetObject(PyExc_KeyError, key.ptr()); python_error(); }
        Py_ssize_t row = integer(PyTuple_GET_ITEM(entry, 0));
        py::object source = make_source(token);
        auto columns = normalize_columns(supplied, 1, false, producer, present);
        py::tuple rows = py::make_tuple(row), uids = py::make_tuple(uid);
        py::object update = make_update(source, rows, uids, columns);
        Selection selection{{row}, uids, token};
        check_rows(mol, selection, py::make_tuple(atom));
        if (producer == config->explicit_source && !transaction.is_none()) {
            int inherited = py::cast<int>(get(transaction, Name::Producer));
            for (auto &column : columns) column.producer = inherited;
        }
        install(get(mol, Name::Table), selection.rows, columns, uids, transaction, true);
    } catch (...) { poison(transaction); throw; }
}

py::capsule capture_status(py::handle table, py::iterable rows, py::iterable groups) {
    auto undo = std::make_unique<Undo>();
    undo->owners[3] = get(table, Name::Status);
    auto status = array(undo->owners[3], 'u', 1, 2, true);
    auto selected_rows = row_values(rows), selected_groups = row_values(groups);
    undo->statuses.resize(selected_rows.size() * selected_groups.size());
    std::size_t offset = 0;
    for (auto row : selected_rows) for (auto group : selected_groups) {
        char *target = address(status, row, group);
        auto before = static_cast<std::uint8_t>(*target);
        undo->statuses[offset++] = {target, before, before};
    }
    py::capsule result(undo.get(), "hotpot.atom_fact_undo", [](PyObject *object) {
        delete static_cast<Undo *>(PyCapsule_GetPointer(object, "hotpot.atom_fact_undo"));
    });
    undo.release(); return result;
}

void update_dependency_seeds(py::handle mol, py::tuple groups, py::tuple uids, bool present) {
    OwnerLock lock(mol);
    py::object transaction = current_transaction(mol);
    try {
        Undo undo;
        undo.owners[4] = get(get(mol, Name::Table), Name::Seeds);
        undo.owners[5] = uids;
        for (auto selected : groups) {
            Py_ssize_t group = integer(selected);
            if (group < 0 || group >= config->groups) throw py::index_error("Unknown dependency group");
            SeedPreparation seeds(undo.seeds, PyTuple_GET_ITEM(undo.owners[4].ptr(), group));
            for (auto uid : uids) seeds.change(uid, present);
        }
        install_undo(undo, transaction);
    } catch (...) { poison(transaction); throw; }
}

void clear_dependency_seeds(py::handle mol, int group, py::iterable rows) {
    OwnerLock lock(mol);
    py::object transaction = current_transaction(mol);
    try {
        Undo undo;
        py::object table = get(mol, Name::Table);
        undo.owners[3] = get(table, Name::Status);
        undo.owners[4] = get(table, Name::Seeds);
        auto status = array(undo.owners[3], 'u', 1, 2, true);
        if (group < 0 || group >= config->groups) throw py::index_error("Unknown dependency group");
        auto selected = row_values(rows);
        auto logical = integer(get(table, Name::Size));
        if (status.shape(0) < logical || status.shape(1) != config->groups)
            throw py::value_error("Atom dependency storage is not aligned");
        for (auto row : selected) {
            if (row < 0 || row >= logical) throw py::index_error("Atom row is outside the current table");
            char *target = address(status, row, group);
            auto before = static_cast<std::uint8_t>(*target);
            if (before == config->current_status) undo.statuses.push_back({target, before, static_cast<std::uint8_t>(config->needs_recalc_status)});
        }
        SeedPreparation seeds(undo.seeds, PyTuple_GET_ITEM(undo.owners[4].ptr(), group));
        seeds.clear();
        install_undo(undo, transaction);
    } catch (...) { poison(transaction); throw; }
}

PyObject *atom_setattr(PyObject *self, PyObject *const *args) {
    try {
        PyObject *index = PyDict_GetItemWithError(config->indices.ptr(), args[0]);
        if (!index && PyErr_Occurred()) python_error();
        if (index) {
            int field = static_cast<int>(integer(index));
            if (field == config->element_field) config->eager(py::handle(self), py::handle(args[0]), py::handle(args[1]));
            else commit_scalar(py::handle(self), field, py::handle(args[1]), config->explicit_source, true);
        } else if (PyUnicode_Check(args[0])
                   && (PyUnicode_CompareWithASCIIString(args[0], "isotope") == 0
                       || PyUnicode_CompareWithASCIIString(args[0], "frac_valence") == 0)) {
            py::object ad = dictionary(py::handle(self));
            PyObject *mol = PyDict_GetItemString(ad.ptr(), "_mol");
            if (mol) {
                py::dict kwargs; kwargs[py::handle(args[0])] = py::handle(args[1]);
                config->sparse(py::handle(mol), py::handle(self), **kwargs);
            } else if (PyObject_GenericSetAttr(self, args[0], args[1]) < 0) python_error();
        } else if (PyObject_GenericSetAttr(self, args[0], args[1]) < 0) python_error();
        Py_RETURN_NONE;
    } catch (py::error_already_set &error) { error.restore(); }
    catch (const py::builtin_exception &error) { error.set_error(); }
    catch (const std::bad_alloc &) { PyErr_NoMemory(); }
    catch (const std::exception &error) { PyErr_SetString(PyExc_RuntimeError, error.what()); }
    return nullptr;
}

void install_dispatch(py::object atom, py::object eager, py::object sparse,
                       py::object foreign, py::object detached) {
    config->atom_type = atom; config->eager = eager; config->sparse = sparse;
    config->foreign_error = foreign; config->detached_error = detached;
    static PyMethodDef method{"__setattr__", reinterpret_cast<PyCFunction>(+[](PyObject *, PyObject *const *args,
                                                                            Py_ssize_t count, PyObject *keywords) -> PyObject * {
        if ((!keywords || !PyTuple_GET_SIZE(keywords)) && count == 3)
            return atom_setattr(args[0], args + 1);
        PyObject *resolved[3] = {nullptr, nullptr, nullptr};
        if (count > 3) { PyErr_SetString(PyExc_TypeError, "Atom.__setattr__ requires self, key and value"); return nullptr; }
        for (Py_ssize_t i = 0; i < count; ++i) resolved[i] = args[i];
        if (keywords) for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(keywords); ++i) {
            PyObject *key = PyTuple_GET_ITEM(keywords, i);
            int position = PyUnicode_CompareWithASCIIString(key, "self") == 0 ? 0
                         : PyUnicode_CompareWithASCIIString(key, "key") == 0 ? 1
                         : PyUnicode_CompareWithASCIIString(key, "value") == 0 ? 2 : -1;
            if (position < 0 || resolved[position]) {
                PyErr_SetString(PyExc_TypeError, "Unexpected or repeated Atom assignment keyword"); return nullptr;
            }
            resolved[position] = args[count + i];
        }
        if (!resolved[0] || !resolved[1] || !resolved[2]) {
            PyErr_SetString(PyExc_TypeError, "Atom.__setattr__ requires self, key and value"); return nullptr;
        }
        return atom_setattr(resolved[0], resolved + 1);
    }), METH_FASTCALL | METH_KEYWORDS,
    "__setattr__(self, key, value)\n--\n\nSchema-addressed factual Atom assignment."};
    py::object function = py::reinterpret_steal<py::object>(PyCFunction_NewEx(&method, nullptr, nullptr));
    if (!function) python_error();
    py::object descriptor = py::reinterpret_steal<py::object>(PyInstanceMethod_New(function.ptr()));
    if (!descriptor) python_error();
    if (PyObject_SetAttrString(atom.ptr(), "__setattr__", descriptor.ptr()) < 0) python_error();
}

std::vector<Column> prepared_columns(py::handle update, int inherited) {
    py::object fields = get(update, Name::Fields), values = get(update, Name::Values);
    py::object presence = get(update, Name::Present), producers = get(update, Name::Producers);
    std::vector<Column> result;
    for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(fields.ptr()); ++i) {
        py::handle field = PyTuple_GET_ITEM(fields.ptr(), i);
        PyObject *index = PyDict_GetItemWithError(config->indices.ptr(), field.ptr());
        if (!index) { if (PyErr_Occurred()) python_error(); throw py::value_error("Unknown canonical Atom field"); }
        int code = static_cast<int>(integer(PyTuple_GET_ITEM(producers.ptr(), i)));
        result.push_back({static_cast<int>(integer(index)), py::reinterpret_borrow<py::bytes>(PyTuple_GET_ITEM(values.ptr(), i)),
                          PyTuple_GET_ITEM(presence.ptr(), i) == Py_True,
                          code == config->explicit_source ? inherited : code});
    }
    return result;
}

void commit_facts(py::handle mol, py::handle update) {
    OwnerLock lock(mol);
    py::object transaction = current_transaction(mol);
    try {
        py::object source = get(update, Name::Source);
        config->check_source(mol, source);
        Selection selection{row_values(get(update, Name::Rows)), py::cast<py::tuple>(get(update, Name::Uids)), get(source, Name::SourceOwner)};
        check_rows(mol, selection);
        int producer = transaction.is_none() ? config->explicit_source : py::cast<int>(get(transaction, Name::Producer));
        auto columns = prepared_columns(update, producer);
        install(get(mol, Name::Table), selection.rows, columns, selection.uids, transaction, true);
    } catch (...) { poison(transaction); throw; }
}

void write_prepared(py::handle table, py::handle selected, py::iterable prepared, py::handle transaction) {
    std::vector<Py_ssize_t> rows;
    if (PyIndex_Check(selected.ptr()) && !py::isinstance<py::array>(selected)) rows.push_back(integer(selected));
    else if (PySlice_Check(selected.ptr())) {
        Py_ssize_t start, stop, step, count;
        if (PySlice_GetIndicesEx(selected.ptr(), integer(get(table, Name::Size)), &start, &stop, &step, &count) < 0) python_error();
        for (Py_ssize_t i = 0; i < count; ++i) rows.push_back(start + i * step);
    } else rows = row_values(selected);
    std::vector<Column> columns;
    for (py::handle entry : prepared) {
        auto values = py::reinterpret_borrow<py::sequence>(entry);
        int index = static_cast<int>(integer(values[0]));
        columns.push_back({index, rows.size() == 1 ? scalar_bytes(values[1])
                                                 : column_bytes(values[1], rows.size(), config->fields.at(index).name),
                           py::cast<bool>(values[2]), py::cast<int>(values[3])});
    }
    install(table, rows, columns, py::none(), transaction, false);
}

void configure(py::dict values) {
    if (config) return;
    auto next = std::make_unique<Configuration>();
    // 3.13+ no longer export the dummy through public headers. Obtain this
    // immutable runtime sentinel from a disposable builtin set, never from an
    // active owner, and retain only the sentinel's lifetime/code identity.
    py::set probe;
    py::int_ probe_uid(0);
    if (PySet_Add(probe.ptr(), probe_uid.ptr()) < 0
        || PySet_Discard(probe.ptr(), probe_uid.ptr()) < 0) python_error();
    auto *probe_set = reinterpret_cast<PySetObject *>(probe.ptr());
    for (Py_ssize_t i = 0; i <= probe_set->mask; ++i) {
        if (probe_set->table[i].hash == -1 && probe_set->table[i].key) {
            next->seed_dummy = py::reinterpret_borrow<py::object>(probe_set->table[i].key);
            break;
        }
    }
    const char *attribute_names[] = {"_mol", "_owner_token", "_member_uid", "_uid_owner", "_atoms", "_topology_index", "_atom_table", "_size", "_columns", "_sources", "_presence", "_group_status", "_dirty_seeds", "_mutation_lock", "mol", "producer", "_cells", "_failed", "source", "rows", "atom_uids", "fields", "value_bytes", "presence", "sources", "owner", "acquire", "release", "dependencies"};
    for (std::size_t i = 0; i < next->names.size(); ++i) {
        next->names[i] = py::reinterpret_steal<py::object>(PyUnicode_InternFromString(attribute_names[i]));
        if (!next->names[i]) python_error();
        if (PyObject_Hash(next->names[i].ptr()) == -1) python_error();
    }
    next->table_type = values["table_type"];
    next->table_offsets.fill(-1);
    for (Name key : {Name::Size, Name::Columns, Name::Sources, Name::Presence, Name::Status, Name::Seeds}) {
        std::size_t index = static_cast<std::size_t>(key);
        PyObject *descriptor = PyObject_GetAttr(next->table_type.ptr(), next->names[index].ptr());
        if (!descriptor) python_error();
        next->table_descriptors[index] = py::reinterpret_steal<py::object>(descriptor);
        if (!Py_IS_TYPE(descriptor, &PyMemberDescr_Type)) throw py::type_error("Fact-table storage requires declared member slots");
        PyMemberDef *member = reinterpret_cast<PyMemberDescrObject *>(descriptor)->d_member;
        if (member->type != T_OBJECT_EX || member->offset < 0
            || member->offset + static_cast<Py_ssize_t>(sizeof(PyObject *)) > reinterpret_cast<PyTypeObject *>(next->table_type.ptr())->tp_basicsize)
            throw py::type_error("Incompatible fact-table member layout");
        next->table_offsets[index] = member->offset;
    }
    next->indices = py::dict(values["indices"]);
    py::dict aliases = values["aliases"];
    for (auto name : py::reinterpret_borrow<py::iterable>(aliases[py::str("coordinates")]))
        next->coordinate_fields.push_back(py::cast<int>(next->indices[name]));
    next->element_field = py::cast<int>(next->indices[aliases[py::str("symbol")][py::int_(0)]]);
    next->writable = values["writable"]; next->readonly = values["readonly"];
    next->source_type = values["source_type"]; next->update_type = values["update_type"];
    next->transactions = values["transactions"]; next->failures = values["failures"];
    next->check_source = values["check_source"]; next->symbol_number = values["symbol_number"];
    next->number_type = values["number_type"]; next->asarray = values["asarray"];
    next->current_status = py::cast<int>(values["current_status"]);
    next->needs_recalc_status = py::cast<int>(values["needs_recalc_status"]);
    next->groups = py::len(values["groups"]);
    if (next->groups > 64) throw std::runtime_error("Declared fact groups exceed the finite native mask");
    next->explicit_source = py::cast<int>(values["producers"][py::str("explicit")]);
    for (auto pair : py::reinterpret_borrow<py::dict>(values["producers"])) next->producers.insert(py::cast<int>(pair.second));
    py::sequence names = values["fields"], bits = values["presence"];
    py::dict effects = values["effects"];
    for (py::size_t i = 0; i < py::len(names); ++i) {
        Field field;
        field.name = py::cast<std::string>(names[i]); field.presence = py::cast<std::uint32_t>(bits[i]);
        py::sequence effect = effects[names[i]];
        field.affected = py::cast<std::vector<int>>(effect[0]);
        if (!effect[1].is_none()) field.materialized = py::cast<int>(effect[1]);
        for (int group : field.affected) field.affected_mask |= std::uint64_t{1} << group;
        if (field.materialized >= 0) field.materialized_mask = std::uint64_t{1} << field.materialized;
        next->fields.push_back(std::move(field));
    }
    config = next.release();
}

} // namespace

PYBIND11_MODULE(_fact_native, module) {
    module.def("_configure", &configure);
    module.def("_write_prepared", &write_prepared, py::arg("table"), py::arg("selection"), py::arg("prepared"), py::arg("transaction") = py::none());
    module.def("_restore_write", [](py::capsule state) {
        auto *undo = static_cast<Undo *>(PyCapsule_GetPointer(state.ptr(), "hotpot.atom_fact_undo"));
        if (!undo) python_error();
        undo->restore();
    });
    module.def("_commit_atom_facts", &commit_facts);
    module.def("_scalar_value", &scalar_value);
    module.def("_commit_atom_scalar", &commit_scalar, py::arg("atom"), py::arg("field_index"), py::arg("value"),
               py::kw_only(), py::arg("producer"), py::arg("present") = true);
    module.def("_commit_atom_columns", &commit_columns);
    module.def("_commit_atom_fields", &commit_fields, py::arg("atom"), py::arg("columns"),
               py::kw_only(), py::arg("producer"), py::arg("present") = true);
    module.def("_install_atom_dispatch", &install_dispatch);
    module.def("_capture_status", &capture_status);
    module.def("_update_dependency_seeds", &update_dependency_seeds);
    module.def("_clear_dependency_seeds", &clear_dependency_seeds);
    module.def("_normalize_atom_scalar", &normalize_scalar, py::arg("source"), py::arg("row"), py::arg("atom_uid"),
               py::arg("field"), py::arg("value"), py::arg("producer"), py::arg("present"));
    module.def("_normalize_atom_facts", [](py::handle source, py::iterable rows, py::iterable uids, py::dict columns) {
        return normalize_facts(source, rows, uids, columns, true, config->explicit_source, true);
    });
    module.def("_normalize_atom_fields", [](py::handle source, py::iterable rows, py::iterable uids, py::dict columns,
                                           int producer, bool present) {
        return normalize_facts(source, rows, uids, columns, false, producer, present);
    });
}
