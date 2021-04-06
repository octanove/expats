
import pytest

from expats.common.instantiate import ConfigFactoried


class MockBase(ConfigFactoried):
    pass


@MockBase.register
class MockImpl(MockBase):
    pass


def test_create():
    mock_impl_instance = MockBase.create_from_factory("MockImpl", {})
    assert type(mock_impl_instance) == MockImpl

    with pytest.raises(KeyError):
        MockBase.create_from_factory("NonExistImpl", {})
