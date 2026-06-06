from typing import Any


def resolve(name) -> Any:
    """Config string -> callable. Bare name = a helper above; dotted = an import path."""
    if "." not in name:
        return globals()[name]
    head, *rest = name.split(".")
    obj = __import__(head)
    for attr in rest:
        obj = getattr(obj, attr)
    return obj
