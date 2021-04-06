
import pytest

from expats.metric.metric import ClassificationMetric, RegressionMetric


@pytest.mark.parametrize(
    "name, inputs, expected",
    [
        ("Accuracy", [("0", "0"), ("1", "2"), ("2", "1"), ("0", "0"), ("1", "0"), ("2", "1")], 2 / 6),
        ("MacroF1", [("0", "0"), ("1", "2"), ("2", "1"), ("0", "0"), ("1", "0"), ("2", "1")], 0.26666666),
        ("MicroF1", [("0", "0"), ("1", "2"), ("2", "1"), ("0", "0"), ("1", "0"), ("2", "1")], 0.33333333),
    ]
)
def test_classification_metric(name, inputs, expected):
    metric = ClassificationMetric.create_from_factory(name, None)
    actual = metric.calculate(inputs)
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize(
    "name, inputs, expected",
    [
        ("PearsonCorrelation", [(1, 10), (2, 9), (3, 2.5), (4, 6), (5, 4)], -0.74261065),
    ]
)
def test_regression_metric(name, inputs, expected):
    metric = RegressionMetric.create_from_factory(name, None)
    actual = metric.calculate(inputs)
    assert actual == pytest.approx(expected)
