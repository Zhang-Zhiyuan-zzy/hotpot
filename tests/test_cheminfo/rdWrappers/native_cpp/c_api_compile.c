#include <hotpot/rdWrappers/c_api.h>


int consume_contract(void) {
    hotpot_rd_options_v1 options = {0};
    hotpot_rd_molecule_view_v1 molecule = {0};
    options.struct_size = (uint32_t)sizeof(options);
    options.abi_version = HOTPOT_RD_C_API_VERSION;
    molecule.struct_size = (uint32_t)sizeof(molecule);
    molecule.abi_version = HOTPOT_RD_C_API_VERSION;
    return options.abi_version == molecule.abi_version ? 0 : 1;
}
