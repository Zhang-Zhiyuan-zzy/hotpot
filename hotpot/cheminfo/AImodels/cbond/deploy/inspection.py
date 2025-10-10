import logging
from typing import Iterable

import torch
import torch.nn as nn


def find_fx_scripted_modules(model: nn.Module, verbose: bool = True):
    scripted = []
    for name, mod in model.named_modules():
        # Check known TorchScript module types
        is_script = isinstance(mod, (torch.jit.ScriptModule, torch.jit.RecursiveScriptModule))
        # Fallback: TorchScript modules often have a _c (C++) object with code()
        if not is_script:
            has_c = getattr(mod, "_c", None) is not None
            has_code = hasattr(getattr(mod, "_c", None), "code")
            is_script = bool(has_c and has_code)
        if is_script:
            scripted.append((name, mod))

    if verbose:
        if scripted:
            print("Scripted submodules found:")
            for name, mod in scripted:
                t = type(mod).__name__
                try:
                    kind = "scripted" if mod._c._is_script_module() else "unknown"
                except Exception:
                    kind = "scripted"
                print(f" - {name or '<root>'}: {t}")
        else:
            print("No scripted submodules detected.")
    return scripted

def check_data_types(model: nn.Module, dummy_inputs: Iterable[torch.Tensor]):
    def register_complex_dtype_hooks(m: nn.Module):
        _handles = []

        def hook_fn(module, inputs, output):
            def check_tensor(x, where):
                if isinstance(x, torch.Tensor):
                    if x.is_complex():
                        print(f"[COMPLEX] {where} in {module.__class__.__name__} "
                              f"dtype={x.dtype}, shape={tuple(x.shape)}")
                elif isinstance(x, (list, tuple)):
                    for i, xi in enumerate(x):
                        check_tensor(xi, f"{where}[{i}]")
                elif isinstance(x, dict):
                    for k, v in x.items():
                        check_tensor(v, f"{where}['{k}']")

            check_tensor(inputs, "inputs")
            check_tensor(output, "output")

        for m in m.modules():
            # Skip the root module if you want only submodules; keep if you want all
            _handles.append(m.register_forward_hook(hook_fn))
        return _handles

    # Usage
    handles = register_complex_dtype_hooks(model)
    _ = model(*dummy_inputs)
    for h in handles: h.remove()

def _create_gm_nodes_table(gm: torch.fx.GraphModule):
    from tabulate import tabulate
    node_specs = [[n.op, n.name, n.target, n.args, n.kwargs] for n in gm.graph.nodes]
    return tabulate(node_specs, headers=["opcode", "name", "target", "args", "kwargs"])

def _dummy_compiler(gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]):
    logging.info(f'\n\n[magenta]====================== In GraphModule ==========================[/]')
    logging.info(f'[blue]====================== FX Graph Table ==========================[/]')
    logging.info(_create_gm_nodes_table(gm))
    logging.info(f'[blue]===================== GraphModule Code =========================[/]')
    logging.info(gm.code.strip())
    logging.info(f'[blue]===================== End of GraphModule =======================[/]')
    logging.info(f'[magenta]===================== End of GraphModule =======================[/]\n\n')
    return gm.forward

def _show_fx_graph(model: nn.Module, dummy_inputs: list[torch.Tensor]):
    fx_model = torch.compile(model, backend=_dummy_compiler)
    fx_model(*dummy_inputs)
    return fx_model