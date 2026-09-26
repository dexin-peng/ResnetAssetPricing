"""Portable numerical results; JSON contains data, never executable objects."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
from .paths import RESULTS_DIR

@dataclass(frozen=True)
class Estimate:
    value: float
    stars: str = ""

def encode(value):
    if isinstance(value, pd.DataFrame):
        return {"type": "frame", "columns": encode(list(value.columns)), "index": encode(list(value.index)),
                "index_names": list(value.index.names), "column_names": list(value.columns.names),
                "data": encode(value.to_numpy().tolist())}
    if isinstance(value, pd.Series):
        return {"type": "series", "name": encode(value.name), "index": encode(list(value.index)), "data": encode(value.tolist())}
    if isinstance(value, np.ndarray):
        return {"type": "array", "data": encode(value.tolist())}
    if isinstance(value, dict):
        return {"type": "mapping", "items": [[encode(k), encode(v)] for k,v in value.items()]}
    if isinstance(value, tuple):
        return {"type": "tuple", "items": [encode(v) for v in value]}
    if isinstance(value, list): return [encode(v) for v in value]
    if hasattr(value, "value") and hasattr(value, "stars"):
        return {"type": "estimate", "value": encode(value.value), "stars": value.stars}
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return {"type": "date", "value": pd.Timestamp(value).isoformat()}
    if isinstance(value, np.generic): return encode(value.item())
    if value is pd.NA: return None
    if isinstance(value,float) and not math.isfinite(value): return None
    return value

def decode(value):
    if isinstance(value, list): return [decode(v) for v in value]
    if not isinstance(value, dict): return value
    kind=value["type"]
    if kind=="mapping": return {decode(k):decode(v) for k,v in value["items"]}
    if kind=="tuple": return tuple(decode(v) for v in value["items"])
    if kind=="array": return np.array(decode(value["data"]))
    if kind=="date": return pd.Timestamp(value["value"])
    if kind=="estimate": return Estimate(float('nan') if value['value'] is None else value['value'],value['stars'])
    if kind=="series": return pd.Series(decode(value["data"]),index=decode(value["index"]),name=decode(value["name"]))
    if kind=="frame":
        index=decode(value["index"]);columns=decode(value["columns"])
        index=pd.MultiIndex.from_tuples(index,names=value["index_names"]) if index and isinstance(index[0],tuple) else pd.Index(index,name=value["index_names"][0])
        columns=pd.MultiIndex.from_tuples(columns,names=value["column_names"]) if columns and isinstance(columns[0],tuple) else pd.Index(columns,name=value["column_names"][0])
        return pd.DataFrame(decode(value["data"]),index=index,columns=columns)
    raise ValueError(f"Unknown result type: {kind}")

def cache_path(asset,cache_root=None):
    from .artifact_registry import ARTIFACTS
    if asset not in ARTIFACTS: raise ValueError(f"Unknown exhibit: {asset}")
    return Path(cache_root or RESULTS_DIR)/f"{asset}.json"

def write_cache(asset,payload,*,cache_root=None):
    if payload is None: raise ValueError(f"No data for {asset}")
    path=cache_path(asset,cache_root);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(encode(payload),ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
    return path

def load_cache(asset,cache_root=None):
    path=cache_path(asset,cache_root)
    if not path.exists(): raise FileNotFoundError(f"Missing {asset}. Run python -m utils.paper_source --asset {asset} --rebuild")
    return decode(json.loads(path.read_text()))

load_render_cache=load_cache
