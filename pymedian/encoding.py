from __future__ import annotations

from typing import Sequence

from .types import DetailsFormat


def to_one_based(indices: Sequence[int]) -> list[int]:
    return [value + 1 for value in indices]


def to_binary(indices: Sequence[int], n: int) -> list[int]:
    if n < 0:
        raise ValueError("n must be non-negative.")

    binary = [0] * n

    for idx in indices:
        if idx < 0 or idx >= n:
            raise IndexError(f"Index out of bounds for n={n}: {idx}")

        binary[idx] = 1

    return binary


def encode_facilities(
    indices        : Sequence[int],
    n              : int          ,
    details_format : DetailsFormat
) -> list[int]:
    if details_format == "indices":
        return list(indices)
    if details_format == "binary" :
        return to_binary(indices, n)

    raise ValueError("details_format must be 'binary' or 'indices'.")
