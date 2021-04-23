# -*- coding: utf-8 -*-
"""
_____________________________________________________________________________
Created By  : Vu Hoang Viet - VietVH9
Created Date: Mon March 22 16:34:00 VNT 2021
Project : AkaOCR core
_____________________________________________________________________________

This file contain split and merge lmdb code
_____________________________________________________________________________
"""


import os
import lmdb
def write_cache(env, cache):
    """Write data to the database

    Parameters
    ----------
    env : Environment
        the lmdb Environment, which is the database.
    cache : dict
        the data to be written to the database
    """
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)
            
def merge_lmdb(input_path, output_path):
    
    entries_num = 0
    for path in os.listdir(input_path): 
        if 'ipynb_checkpoints' in path:
            continue
        env = lmdb.open(os.path.join(input_path,path),
                        max_readers=32,
                        readonly=True,
                        lock=False,
                        readahead=False,
                        meminit=False)
        entries_num += env.stat()['entries']
        env.close()
        
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    cache = {}
    ind = 0
    for path in os.listdir(input_path):  
        if 'ipynb_checkpoints' in path:
            continue 
        input_env = lmdb.open(os.path.join(input_path,path),
                                max_readers=32,
                                readonly=True,
                                lock=False,
                                readahead=False,
                                meminit=False)
        with input_env.begin(write = False) as txn:
            for _,v in txn.cursor():
                tail = str(ind).zfill(len(str(entries_num)))
                key = 'image-%s'%tail
                cache[key.encode("ascii")] = v
                ind+=1
        input_env.close()
    output_env = lmdb.open(output_path, map_size=int(1e12))
    write_cache(output_env, cache)
            
def split_lmdb(input_path, output_path, output_size):
    input_env = lmdb.open(input_path,
                          max_readers=32,
                          readonly=True,
                          lock=False,
                          readahead=False,
                          meminit=False)
    entries_num = input_env.stat()['entries']
    lmdb_name = os.path.basename(input_path)
    out_num = 0
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    with input_env.begin(write = False) as txn:
        for i, (_,v) in enumerate(txn.cursor()):
            if i % output_size == 0:
                out_name = lmdb_name+"_%s"%str(out_num).zfill(4)
                if out_num == 10:
                    break
                out_num += 1
                out_path = os.path.join(output_path, out_name)
                output_env = lmdb.open(out_path, map_size=int(1e12))
                cache = {}
            tail = str(i%output_size).zfill(len(str(output_size)))
            key = 'image-%s'%tail
            cache[key.encode("ascii")] = v
            if i% output_size == output_size - 1:
                write_cache(output_env, cache)
                output_env.close()
                print(out_num)