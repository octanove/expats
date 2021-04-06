
import pytest

from expats.common.config_util import dump_to_file, load_from_file, merge_with_dotlist


@pytest.mark.parametrize(
    "orig_dic, dotlist, expected_dic",
    [
        ({"a": 1, "b": 3}, None, {"a": 1, "b": 3}),
        ({"a": 1, "b": 3}, ["b=10"], {"a": 1, "b": 10}),
        ({"a": 1, "b": 3}, ["c=10"], {"a": 1, "b": 3, "c": 10}),
    ]
)
def test_save_and_load_and_merge(tmp_path, orig_dic, dotlist, expected_dic):
    path = str(tmp_path / "test_save_and_load")
    dump_to_file(orig_dic, path)
    actual_dic = load_from_file(path)
    if dotlist:
        actual_dic = merge_with_dotlist(actual_dic, dotlist)
    assert actual_dic == expected_dic
