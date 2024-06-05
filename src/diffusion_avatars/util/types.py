from dataclasses import dataclass, fields
from typing import Dict


def to_shallow_dict(dclass: dataclass) -> Dict:
    return {field.name: getattr(dclass, field.name) for field in fields(dclass)}


def default_dict_to_dict(d):
    # Taken from: https://stackoverflow.com/questions/20428636/how-to-convert-defaultdict-to-dict
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = default_dict_to_dict(v)
    return dict(d)
