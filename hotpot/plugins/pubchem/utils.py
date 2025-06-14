# -*- coding: utf-8 -*-
"""
===========================================================
 Project   : hotpot
 File      : utils
 Created   : 2025/6/12 22:23
 Author    : zhang
 Python    : 
-----------------------------------------------------------
 Description
 ----------------------------------------------------------
 
===========================================================
"""
import asyncio

import pubchempy as pcp


def get_compounds_cid(identifier: str, id_type: str = 'name'):
    try:
        compounds = pcp.get_compounds(identifier, id_type)[0]
        return compounds.cid
    except IndexError:
        print(UserWarning(f'Failed to get compounds cid for {identifier}'))
        return None

async def async_get_compounds_cid(identifier: str, id_type: str = 'name'):
    def sync_func():
        try:
            compounds = pcp.get_compounds(identifier, id_type)[0]
            return compounds.cid
        except IndexError:
            print(UserWarning(f'Failed to get compounds cid for {identifier}'))
            return None

    return await asyncio.to_thread(sync_func)
