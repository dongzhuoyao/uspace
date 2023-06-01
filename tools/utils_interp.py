from configs.configs_utils_common import construct_solver_desc
from tools.utils_attr import get_attr_name_from_attr_id


def cal_delta_change(
    _real_data, _data_recovered, config, target_txt="delta_change.txt"
):
    _diff = (_data_recovered - _real_data).abs().mean()
    delta_change = _diff / _real_data.abs().mean()
    _attr_name = get_attr_name_from_attr_id(
        config.dissection.ith_attr, config.dataset.name
    )
    _data_mean = _data_recovered.abs().mean()
    _solver_desc = construct_solver_desc(**config.dissection.solver_kwargs)
    print(f"{_data_mean},{_attr_name},{_solver_desc}\n")
    with open(target_txt, "a") as f:
        f.write(f"{_data_mean:5f},{_attr_name},{_solver_desc}\n")
