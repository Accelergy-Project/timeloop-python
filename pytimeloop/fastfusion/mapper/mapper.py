from collections import defaultdict
from collections.abc import Iterable
from functools import partial

from ruamel.yaml import YAML
yaml = YAML(typ='safe')

from bindings.looptree import (
    LooptreeWorkload,
    LooptreeWorkloadDependencyAnalyzer
)

from pytimeloop.fastfusion.pareto import OpData, Pareto

from .level_mapper.exhaustive import ExhaustiveLevelMapper
from .level_mapper.top_level import TopLevelMapper
from .stepped_model import Stats, SteppedModel, SteppedModelState


def mapper(config, spec, workload, name_of_einsum_to_eval, tmp_path):
    einsum_name_to_id = workload.einsum_name_to_id()
    id_of_einsum_to_eval = einsum_name_to_id[name_of_einsum_to_eval]

    bindings, max_spatial, max_capacity = get_hardware_levels(spec.architecture)

    ranks = workload.einsum_ospace_dimensions(id_of_einsum_to_eval)
    tensors = (
        workload.tensors_read_by_einsum(id_of_einsum_to_eval)
        |
        workload.tensors_written_by_einsum(id_of_einsum_to_eval)
    )

    adj_list = get_neighbors(workload)

    fusable_tensors = tensors & get_intermediate_tensors(workload)

    # Shape is given as *inclusive* (min, max) by workload
    einsum_shape = {
        rank_id: workload.get_rank_shape(rank_id)[1]+1 for rank_id in ranks
    }

    analyzer = LooptreeWorkloadDependencyAnalyzer(workload)

    model = SteppedModel(config, spec, bindings, workload, analyzer)
    model.call_accelergy(tmp_path)
    state = SteppedModelState()
    model.initialize(state, 0, id_of_einsum_to_eval, list(tensors))

    def step_back_model():
        model.step_back()

    def final_model(level, state, temporal_loops, spatial_loops, retained_tensors):
        model.add_compute(state,
                          level,
                          name_of_einsum_to_eval,
                          temporal_loops,
                          spatial_loops)
        return model.run(state)

    def partial_model(level, state, temporal_loops, spatial_loops, retained_tensors):
        model.add_storage(state,
                          level,
                          temporal_loops,
                          spatial_loops,
                          retained_tensors)
        return Stats()



    cur_mapper = None
    for hw_level in reversed(range(1, len(bindings)-1)):
        if cur_mapper is None:
            cur_mapper = ExhaustiveLevelMapper(hw_level,
                                               ranks,
                                               tensors,
                                               max_spatial=max_spatial[hw_level],
                                               max_capacity=max_capacity,
                                               can_bypass=False,
                                               lower_mapper=None,
                                               partial_model=partial(final_model,
                                                                     level=hw_level),
                                               step_back_model=step_back_model)
        else:
            cur_mapper = ExhaustiveLevelMapper(hw_level,
                                               ranks,
                                               tensors,
                                               max_spatial=max_spatial[hw_level],
                                               max_capacity=max_capacity,
                                               can_bypass=True,
                                               lower_mapper=cur_mapper,
                                               partial_model=partial(partial_model,
                                                                     level=hw_level),
                                               step_back_model=step_back_model)

    cur_mapper = TopLevelMapper(hw_level,
                                ranks,
                                tensors,
                                fusable_tensors,
                                id_of_einsum_to_eval,
                                adj_list[id_of_einsum_to_eval],
                                lower_mapper=cur_mapper,
                                partial_model=partial_model,
                                step_back_model=step_back_model,
                                max_spatial=max_spatial,
                                max_capacity=max_capacity)

    cur_mapper.run(einsum_shape)

    result = cur_mapper.get_result()
    op_data = OpData(frozenset({id_of_einsum_to_eval}), frozenset(tensors))
    pareto = Pareto({op_data: result})

    return pareto


def get_hardware_levels(arch):
    bindings = {}
    fanout = {}
    max_capacity = {}
    for node in arch['nodes']:
        bindings_id = len(bindings)
        bindings[bindings_id] = node['name']
        fanout[bindings_id] = (node.spatial.meshX, node.spatial.meshY)
        attribute = node.attribute
        if 'width' in attribute and 'height' in attribute:
            max_capacity[bindings_id] = \
                attribute.width * attribute.height / attribute.datawidth
    return bindings, fanout, max_capacity


def get_neighbors(workload):
    adj_list = defaultdict(lambda: list())
    for einsum_u_id in workload.einsum_id_to_name():
        for einsum_v_id in workload.einsum_id_to_name():
            u_written_tensor = workload.tensor_written_by_einsum(einsum_u_id)
            v_read_tensors = workload.tensors_read_by_einsum(einsum_v_id)
            if u_written_tensor is not None and u_written_tensor in v_read_tensors:
                adj_list[einsum_u_id].append(einsum_v_id)
                adj_list[einsum_v_id].append(einsum_u_id)
                continue
            u_read_tensors = workload.tensors_read_by_einsum(einsum_u_id)
            v_written_tensor = workload.tensor_written_by_einsum(einsum_v_id)
            if v_written_tensor is not None and v_written_tensor in u_read_tensors:
                adj_list[einsum_u_id].append(einsum_v_id)
                adj_list[einsum_v_id].append(einsum_u_id)
    return adj_list


def get_intermediate_tensors(workload: LooptreeWorkload):
    tensor_id_to_name = workload.data_space_id_to_name()
    result = set()
    for einsum in workload.einsum_id_to_name():
        written_tensors = workload.tensors_written_by_einsum(einsum)
        for tensor in written_tensors:
            reader_einsums = workload.reader_einsums(tensor)
            for reader in reader_einsums:
                if reader in workload.einsum_id_to_name():
                    result.add(tensor_id_to_name[tensor])
                    break

    return result
