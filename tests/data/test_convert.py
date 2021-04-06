
import pytest

from expats.data.convert import RoundNearestInteger, MinMaxDenormalizedRoundNearestInteger


@pytest.mark.parametrize(
    "inputs, expected_outputs",
    [
        ([2.3, 0.1, -1.7, 3.5], ["2", "0", "-2", "4"]),
    ]
)
def test_round_nearest_integer(inputs, expected_outputs):
    converter = RoundNearestInteger()
    assert converter.convert(inputs) == expected_outputs


@pytest.mark.parametrize(
    "x_min, x_max, inputs, expected_outputs",
    [
        (2, 10, [0.75, 0, 1], ["8", "2", "10"]),
    ]
)
def test_min_max_denormalized_round_nearest_integer(
    x_min, x_max, inputs, expected_outputs
):
    converter = MinMaxDenormalizedRoundNearestInteger(x_min, x_max)
    assert converter.convert(inputs) == expected_outputs
