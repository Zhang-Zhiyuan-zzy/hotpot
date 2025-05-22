# -*- coding: utf-8 -*-
"""
===========================================================
 Project   : hotpot
 File      : custom_json
 Created   : 2025/5/17 20:52
 Author    : zhang
 Python    : 
-----------------------------------------------------------
 Description
 ----------------------------------------------------------
 
===========================================================
"""
import json


def dumps_with_partial_indent(obj, *,
                               indent=4,          # size of one indent block
                               pretty_levels=2,   # how many levels get it
                               **json_kwargs
    ):    # any normal json.dumps kw’s
    """
    json.dumps()  that is pretty-printed only for the first *pretty_levels*
    of dict nesting; deeper levels are emitted in the normal compact form.
    """
    spacer = " " * indent             # cache the basic indent string

    def _render(value, depth):
        # If we are still inside the “pretty” zone and value is a dict, lay it out
        if depth < pretty_levels and isinstance(value, dict):
            # Collect "<indent><key>: <value>" lines
            pieces = []
            for k, v in value.items():
                rendered_v = _render(v, depth + 1)
                pieces.append(
                    f"{spacer * (depth + 1)}{json.dumps(k)}: {rendered_v}"
                )
            inner = ",\n".join(pieces)
            return "{\n" + inner + "\n" + spacer * depth + "}"
        # otherwise fall back to normal compact json
        return json.dumps(value, separators=(",", ":"), **json_kwargs)

    return _render(obj, 0)
