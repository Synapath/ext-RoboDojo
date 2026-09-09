import importlib.util
from pathlib import Path
import copy
import pytest

spec = importlib.util.spec_from_file_location("intervention", Path(__file__).parents[1]/"utils/charger_intervention.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def apply(layout, raw, **kw):
    return m.shifted_socket_layout(layout, raw, task=kw.get("task", "plug_in_charger"), checkpoint=kw.get("checkpoint", "engineering-test"), diagnostics=kw.get("diagnostics", True))


def test_translation_preserves_every_other_field_and_source():
    src = {"Rigid": {"socket": [{"default_pos": [1.,2.,3.], "default_ori": [1,0,0,0], "physics": {"mass":2}}], "charger": [{"default_pos": [4,5,6]}]}, "other": [1,2]}
    before = copy.deepcopy(src)
    shifted, receipt = apply(src, '[0.003,-0.004]')
    assert shifted["Rigid"]["socket"][0]["default_pos"] == [1.003,1.996,3.]
    shifted["Rigid"]["socket"][0]["default_pos"] = before["Rigid"]["socket"][0]["default_pos"]
    assert shifted == before == src
    assert receipt["candidate_layout_sha256"] != receipt["source_layout_sha256"]
    zero, receipt = apply(src, '[0,0]')
    assert zero == src and zero is not src
    assert receipt["candidate_layout_sha256"] == receipt["source_layout_sha256"]


@pytest.mark.parametrize("raw", ['[0.026,0]', '[0.02,0.02]', '[NaN,0]', '[true,0]', '[0,0,0]', 'null'])
def test_bad_vectors(raw):
    with pytest.raises(ValueError): apply({"Rigid":{"socket":[{"default_pos":[0,0,0]}]}}, raw)


@pytest.mark.parametrize("kw", [{"task":"other"}, {"checkpoint":"model"}, {"diagnostics":False}])
def test_scope(kw):
    with pytest.raises(ValueError): apply({}, '[0,0]', **kw)


def test_nonunique_socket():
    with pytest.raises(ValueError): apply({"Rigid":{"socket":[]}}, '[0,0]')
