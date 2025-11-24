"""
@File Name:        check.py
@Project:          
@Author:           Zhiyuan Zhang
@Created On:       2025/11/20 20:10
@Project:          Hotpot
"""
import torch.nn as nn
from hotpot.utils import fmt_print


NAME_LEN = 100

def _print_table_header(headers, col_widths):
    """Helper to print a neat header."""
    header_str = "".join([f"{h:<{w}}" for h, w in zip(headers, col_widths)])
    separator_str = "-" * len(header_str)

    fmt_print.bold_magenta(separator_str)
    fmt_print.bold_magenta(header_str)
    fmt_print.bold_magenta(separator_str)


def _format_row(columns, col_widths):
    """Helper to format a single row."""
    return "".join([f"{str(c):<{w}}" for c, w in zip(columns, col_widths)])


def check_trainable_parameters(model: nn.Module):
    """
    1) Checks and prints only the parameters that are currently trainable.
    Formatted as a table.
    """
    trainable_params = 0
    all_param = 0

    # Define column widths: Name, Shape, Param Count
    col_widths = [NAME_LEN, 25, 25]
    headers = ["Layer Name", "Shape", "Params"]

    fmt_print.bold_magenta("\n=== CHECKING TRAINABLE PARAMETERS ===")
    _print_table_header(headers, col_widths)

    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

            # Truncate long names if necessary to keep table neat
            display_name = (name[:NAME_LEN - 2] + '..') if len(name) > NAME_LEN else name
            shape_str = str(list(param.shape))

            row_str = _format_row([display_name, shape_str, param.numel()], col_widths)
            fmt_print.bold_magenta(row_str)

    percentage = 100 * trainable_params / all_param if all_param > 0 else 0

    fmt_print.bold_magenta("-" * sum(col_widths))
    fmt_print.bold_magenta(f"Total Trainable: {trainable_params}")
    fmt_print.bold_magenta(f"Total Params:    {all_param}")
    fmt_print.bold_magenta(f"Trainable %:     {percentage:.4f}%")


def check_all_parameters_status(model: nn.Module):
    """
    2) Prints the shape and trainable status for ALL parameters.
    Formatted as a table.
    """
    # Define column widths: Name, Shape, Status
    col_widths = [NAME_LEN, 25, 25]
    headers = ["Layer Name", "Shape", "Status"]

    fmt_print.bold_magenta("\n=== ALL PARAMETERS STATUS ===")
    _print_table_header(headers, col_widths)

    for name, param in model.named_parameters():
        status = "TRAINABLE" if param.requires_grad else "FROZEN"

        display_name = (name[:NAME_LEN - 2] + '..') if len(name) > NAME_LEN else name
        shape_str = str(list(param.shape))

        row_str = _format_row([display_name, shape_str, status], col_widths)
        fmt_print.bold_magenta(row_str)


def check_gradient_values(model: nn.Module):
    """
    3) Checks which parameters actually have a populated .grad attribute.
    Formatted as a table.
    """
    # Define column widths: Name, Grad Status, Mean Value
    col_widths = [NAME_LEN, 25, 25]
    headers = ["Layer Name", "Grad Status", "Mean Abs Grad"]

    fmt_print.bold_magenta("\n=== CHECKING GRADIENT VALUES (Post-Backward) ===")
    _print_table_header(headers, col_widths)

    has_grad_count = 0
    no_grad_count = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            display_name = (name[:NAME_LEN - 2] + '..') if len(name) > NAME_LEN else name

            if param.grad is not None:
                grad_mean = param.grad.abs().mean().item()
                row_str = _format_row([display_name, "DETECTED", f"{grad_mean:.8f}"], col_widths)
                fmt_print.bold_magenta(row_str)
                has_grad_count += 1
            else:
                row_str = _format_row([display_name, "MISSING", "None"], col_widths)
                fmt_print.bold_magenta(row_str)
                no_grad_count += 1

    fmt_print.bold_magenta("-" * sum(col_widths))
    fmt_print.bold_magenta(f"Parameters with Gradients:    {has_grad_count}")
    fmt_print.bold_magenta(f"Trainable but NO Gradients:   {no_grad_count}")