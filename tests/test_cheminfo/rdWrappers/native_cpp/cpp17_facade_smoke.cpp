#include <hotpot/rdWrappers/rdwrapper.hpp>

#include <cassert>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <thread>


int main() {
    namespace native = hotpot::native;
    namespace rdw = hotpot::rdwrappers;

    native::OwnedMoleculeData molecule;
    molecule.atom_count = 5;
    molecule.bond_count = 4;
    molecule.fields =
        native::molecule_field_mask(native::MoleculeField::ATOMIC_NUMBERS)
        | native::molecule_field_mask(native::MoleculeField::FORMAL_CHARGES)
        | native::molecule_field_mask(native::MoleculeField::ATOM_AROMATIC)
        | native::molecule_field_mask(native::MoleculeField::BOND_INDICES)
        | native::molecule_field_mask(native::MoleculeField::BOND_KINDS)
        | native::molecule_field_mask(native::MoleculeField::BOND_AROMATIC);
    molecule.atomic_numbers = {6, 1, 1, 1, 1};
    molecule.formal_charges = {0, 0, 0, 0, 0};
    molecule.atom_aromatic = {0, 0, 0, 0, 0};
    molecule.bond_indices = {{0, 1}, {0, 2}, {0, 3}, {0, 4}};
    molecule.bond_kinds = {
        native::BondKind::SINGLE,
        native::BondKind::SINGLE,
        native::BondKind::SINGLE,
        native::BondKind::SINGLE,
    };
    molecule.bond_aromatic = {0, 0, 0, 0};

    rdw::ETKDGOptions options;
    options.conformer_count = 2;
    options.seed = 0x5eed;
    const rdw::ETKDGSampler sampler(options);
    const auto result = sampler.generate(molecule.view());
    assert(result.status == rdw::GenerationStatus::SUCCESS);
    assert(result.conformer_count == 2);
    assert(result.atom_count == 5);
    assert(result.coordinates.size() == 10);
    assert(result.records.size() == 2);
    assert(result.records[0].ordinal == 0);
    for (const auto& coordinate : result.coordinates) {
        for (double value : coordinate) assert(std::isfinite(value));
    }

    std::array<double, 4> concurrent_first_x{};
    std::array<std::thread, 4> workers;
    for (std::size_t index = 0; index < workers.size(); ++index) {
        workers[index] = std::thread([&, index] {
            const auto concurrent = sampler.generate(molecule.view());
            assert(concurrent.status == rdw::GenerationStatus::SUCCESS);
            concurrent_first_x[index] = concurrent.coordinates[0][0];
        });
    }
    for (auto& worker : workers) worker.join();
    for (double value : concurrent_first_x) {
        assert(value == result.coordinates[0][0]);
    }
    const auto runtime = rdw::runtime_info();
    assert(runtime.native);
    assert(runtime.hotpot_version == "0.5.4.5");
    assert(runtime.rdkit_version == "2026.09.1");
    return 0;
}
