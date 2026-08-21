import pytest
from calculator import average, divide


def test_divide_returns_quotient() -> None:
    assert divide(9, 3) == 3


def test_divide_by_zero_raises() -> None:
    with pytest.raises(ValueError, match="denominator cannot be zero"):
        divide(9, 0)


def test_average_preserves_decimal_precision() -> None:
    assert average([1, 2]) == 1.5
