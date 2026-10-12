"""Portable configuration loading for the sensitivity experiment bundles."""
from copy import deepcopy
from pathlib import Path
from core.config_loader import load_all_configs, load_yaml_file


def filter_ric_config_xapps(ric_config, *, enabled_xapp_names):
    result=deepcopy(ric_config)
    available={item['name'] for item in result.get('xapps',[])}
    unknown=set(enabled_xapp_names)-available
    if unknown:raise ValueError(f'Unknown xApps: {sorted(unknown)}')
    result['xapps']=[item for item in result.get('xapps',[]) if item['name'] in enabled_xapp_names]
    return result


def load_xapp_experiment_configs(project_root, **filenames):
    root=Path(project_root).resolve()/'configs'
    configs=load_all_configs(str(root)); sources={}
    for argument,filename in filenames.items():
        if not argument.endswith('_filename'):raise ValueError(f'Unknown option: {argument}')
        section=argument.removesuffix('_filename')
        if section not in {'sim','cell','ue','ric','beamforming','scenario','phy'}:
            raise ValueError(f'Unknown configuration section: {section}')
        path=(root/filename).resolve()
        if not path.is_relative_to(root):raise ValueError('Configuration must stay under configs/')
        configs[section]=load_yaml_file(str(path)); sources[section]=str(path)
    return configs,sources
