"""Subsetting must not relocate, renumber, or change a UE's traffic template."""
import json
import pytest

from phy.rt_sys.scenario import ROOT, build_case


def subset_config(tmp_path, subset):
    config = json.loads((ROOT / 'configs/rt_sys/default.json').read_text())
    config['ue_subset'] = subset
    config['ray_solver']['max_depth'] = 2
    path = tmp_path / 'six.json'
    path.write_text(json.dumps(config))
    return path


def test_subset_preserves_source_trajectories_ids_and_business(tmp_path):
    ids = ['ue_0', 'ue_1', 'ue_2', 'ue_4', 'ue_6', 'ue_8']
    path = subset_config(tmp_path, {'source_num_ues': 12, 'ue_ids': ids})
    original, old_config, _ = build_case('munich', 3, 12, 1, 4000, device='cpu')
    selected, new_config, _ = build_case('munich', 3, 6, 1, 4000, device='cpu', config_path=path)
    assert selected['ues'] == [u for u in original['ues'] if u['ue_id'] in ids]
    assert new_config['ue']['ues'] == [u for u in old_config['ue']['ues'] if u['ue_id'] in ids]
    assert len(selected['ues']) == len(new_config['ue']['ues']) == 6
    assert selected['ray_solver']['max_depth'] == 2
    assert original['ray_solver']['max_depth'] == 3
    assert selected['cells'] == original['cells']


@pytest.mark.parametrize('subset', [
    {'source_num_ues': 12, 'ue_ids': ['ue_0'] * 6},
    {'source_num_ues': 12, 'ue_ids': ['ue_0', 'ue_1', 'ue_2', 'ue_4', 'ue_6', 'ue_12']},
    {'source_num_ues': 12, 'ue_ids': ['ue_0', 'ue_1']},
    {'source_num_ues': 5, 'ue_ids': ['ue_0', 'ue_1', 'ue_2', 'ue_3', 'ue_4', 'ue_5']},
    {'source_num_ues': 12.0, 'ue_ids': []},
    ['ue_0'],
])
def test_invalid_subset_is_rejected(tmp_path, subset):
    path = subset_config(tmp_path, subset)
    with pytest.raises(ValueError, match='ue_subset'):
        build_case('munich', 3, 6, config_path=path)


def test_plain_six_ue_generation_is_not_mistaken_for_original_subset(tmp_path):
    path = subset_config(tmp_path, {'source_num_ues': 12, 'ue_ids': ['ue_0', 'ue_1', 'ue_2', 'ue_4', 'ue_6', 'ue_8']})
    subset, _, _ = build_case('munich', 3, 6, config_path=path)
    fresh, _, _ = build_case('munich', 3, 6)
    assert subset['ues'][0]['trajectory']['positions_m'][0] != fresh['ues'][0]['trajectory']['positions_m'][0]


def test_explicit_replacement_position_changes_only_selected_ue_geometry(tmp_path):
    path = subset_config(tmp_path, {'source_num_ues': 12, 'ue_ids': ['ue_0', 'ue_1', 'ue_2', 'ue_4', 'ue_6', 'ue_8']})
    before, old, _ = build_case('munich', 3, 6, 1, 4000, config_path=path)
    config = json.loads(path.read_text())
    config['ue_route_overrides'] = {'ue_0': {'route_index': 1, 'start_fraction': .4}}
    path.write_text(json.dumps(config))
    after, new, _ = build_case('munich', 3, 6, 1, 4000, config_path=path)
    assert after['ues'][0]['trajectory']['positions_m'][0] == [35.8, 20.0, 1.5]
    assert after['ues'][1:] == before['ues'][1:]
    for a,b in zip(old['ue']['ues'], new['ue']['ues']):
        assert {k:v for k,v in a.items() if k not in ('x_m','y_m','z_m')} == {k:v for k,v in b.items() if k not in ('x_m','y_m','z_m')}
    start,end=after['ues'][0]['trajectory']['positions_m'][::len(after['ues'][0]['trajectory']['positions_m'])-1]
    assert end[0]-start[0] == pytest.approx(3.)


@pytest.mark.parametrize('overrides', [
    {'ue_99': {'route_index': 1, 'start_fraction': .4}},
    {'ue_0': {'route_index': 99, 'start_fraction': .4}},
    {'ue_0': {'route_index': 1, 'start_fraction': float('nan')}},
    {'ue_0': {'route_index': 1, 'start_fraction': -1}},
])
def test_invalid_road_override_is_rejected(tmp_path, overrides):
    path = subset_config(tmp_path, {'source_num_ues': 12, 'ue_ids': ['ue_0', 'ue_1', 'ue_2', 'ue_4', 'ue_6', 'ue_8']})
    config=json.loads(path.read_text());config['ue_route_overrides']=overrides
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='ue_route_overrides'):
        build_case('munich', 3, 6, config_path=path)
