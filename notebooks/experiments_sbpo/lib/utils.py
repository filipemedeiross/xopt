import os
import math


def parse_optional_int_env(name: str) -> int | None:
    raw_value = os.getenv(name)

    if raw_value is None:
        return None

    raw_value = raw_value.strip()
    if not raw_value:
        return None

    return int(raw_value)


def finite_or_none(
    value: float | int | None
) -> float | None:
    if value is None:
        return None

    value = float(value)

    if not math.isfinite(value):
        return None

    return value


def normalize_number(
    value  : float | int | None,
    digits : int = 4
):
    value = finite_or_none(value)

    if value is None:
        return None

    if abs(value - round(value)) < 10 ** (-digits):
        return int(round(value))

    return round(value, digits)
