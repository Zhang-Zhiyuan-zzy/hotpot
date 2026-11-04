"""Lazy CIF syntax only; no external geometry, group or chemistry objects."""


def _syntax():
    try:
        from gemmi import cif
    except ImportError as exc:
        raise ValueError('CRYSTAL-UNSUPPORTED-PROJECTION: CIF syntax requires the optional '
                         'crystal dependency gemmi>=0.7.5,<0.8') from exc
    return cif


def _tag(value):
    return value.lower().replace('.', '_').replace('[', '').replace(']', '')


def _read_cif(text, block=None):
    cif = _syntax()
    document = cif.read_string(text)
    blocks = tuple(document)
    if block is None:
        if len(blocks) != 1:
            raise ValueError('CRYSTAL-UNSUPPORTED-PROJECTION: select one exact CIF block')
        selected = blocks[0]
    else:
        selected = next((value for value in blocks if value.name == block), None)
        if selected is None:
            raise ValueError('CRYSTAL-UNSUPPORTED-PROJECTION: requested CIF block is absent: ' + str(block))
    values = {}
    def put(tag, entries):
        tag = _tag(tag)
        if tag in values:
            raise ValueError('CRYSTAL-CONTEXT-INTEGRITY: repeated normalized CIF tag ' + tag)
        values[tag] = tuple(value if value in ('?', '.') else cif.as_string(value) for value in entries)
    for item in selected:
        if item.pair is not None:
            put(item.pair[0], (item.pair[1],))
        elif item.loop is not None:
            for index, tag in enumerate(item.loop.tags):
                put(tag, item.loop.values[index::len(item.loop.tags)])
    return selected.name, values


def _write_cif(name, pairs, loops):
    cif = _syntax()
    document = cif.Document()
    block = document.add_new_block(name)
    def quote(value):
        return value if value in ('?', '.') else cif.quote(value)
    for tag, value in pairs:
        block.set_pair(tag, quote(value))
    for tags, rows in loops:
        loop = block.init_loop('', tags)
        for row in rows:
            loop.add_row([quote(value) for value in row])
    return document.as_string()
