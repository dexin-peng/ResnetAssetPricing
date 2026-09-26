"""Inspect, rebuild, and verify individual numerical exhibits."""
import argparse
import json
from pathlib import Path
import tempfile
import time
import numpy as np
from .artifact_registry import ARTIFACTS
from .artifact_cache import load_cache,encode
from .preview import available,table,figure,rebuild
from .paths import RESULTS_DIR

def _compare(left,right,path='result'):
    """Compare data structure, significance and estimates, allowing float roundoff."""
    if left is None and isinstance(right,(int,float)) and np.isnan(right): return
    if right is None and isinstance(left,(int,float)) and np.isnan(left): return
    if isinstance(left,dict):
        if not isinstance(right,dict) or left.keys()!=right.keys(): raise AssertionError(f'{path}: different fields')
        for key in left: _compare(left[key],right[key],f'{path}.{key}')
    elif isinstance(left,list):
        if not isinstance(right,list) or len(left)!=len(right): raise AssertionError(f'{path}: different lengths')
        for i,(a,b) in enumerate(zip(left,right)): _compare(a,b,f'{path}[{i}]')
    elif isinstance(left,(int,float)) and isinstance(right,(int,float)):
        if not np.isclose(left,right,rtol=1e-9,atol=1e-10,equal_nan=True): raise AssertionError(f'{path}: {left} != {right}')
    elif left!=right:
        raise AssertionError(f'{path}: {left!r} != {right!r}')

def verify(names):
    with tempfile.TemporaryDirectory(prefix='resassetpricing-verify-') as directory:
        for name in names:
            start=time.perf_counter()
            if ARTIFACTS[name][4]=='static':
                assert not table(name).empty
            else:
                reference=load_cache(name)
                result=rebuild(name,cache_root=Path(directory))
                _compare(encode(reference),encode(result),name)
            print(f'PASS {name} ({time.perf_counter()-start:.1f} s)',flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list',action='store_true')
    parser.add_argument('--asset',nargs='+',choices=sorted(ARTIFACTS))
    parser.add_argument('--all',action='store_true')
    parser.add_argument('--rebuild',action='store_true',help='Recompute estimates from the documented input summaries.')
    parser.add_argument('--verify',action='store_true',help='Recompute from local inputs and compare with local reference caches without changing them.')
    args=parser.parse_args()
    if args.list: print(available().to_string(index=False));return
    names=list(ARTIFACTS) if args.all else args.asset
    if not names: parser.error('Choose --list, --asset NAME, or --all')
    if args.verify and args.rebuild: parser.error('--verify and --rebuild are separate operations')
    if args.verify: verify(names);return
    for name in names:
        if args.rebuild: rebuild(name)
        if ARTIFACTS[name][4]=='figure': figure(name);print(RESULTS_DIR/'figures'/f'{name}.png')
        else:
            frame=table(name);print(frame.to_string())
            path=RESULTS_DIR/f'{name}.csv'
            raw=table(name,raw=True)
            raw.to_csv(path,index=True)
            print(f'Underlying table: {path}')
if __name__=='__main__': main()
