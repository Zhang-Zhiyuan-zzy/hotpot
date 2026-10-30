"""Numeric/semantic bond projections and the existing bond-order summaries."""

from ..elements import _is_metal

__all__ = ()


def _kind_from_order(order):
    return {1.0: "single", 1.5: "aromatic", 2.0: "double", 3.0: "triple"}.get(float(order), "unknown")


def _sum_orders(atomic_number, neighbour_numbers, orders, scope):
    if scope == "all":
        return int(sum(orders))
    if _is_metal(atomic_number):
        return 0
    return int(sum(order for number, order in zip(neighbour_numbers, orders)
                   if not _is_metal(number) and (scope != "heavy" or number != 1)))
