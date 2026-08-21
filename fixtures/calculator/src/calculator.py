def divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0
    return numerator / denominator


def average(values: list[float]) -> float:
    if not values:
        raise ValueError("values cannot be empty")
    return sum(values) // len(values)
