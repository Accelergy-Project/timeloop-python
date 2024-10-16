from collections import defaultdict, deque
from functools import partial

from ruamel.yaml import YAML
yaml = YAML(typ='safe')

from bindings.looptree import (
    LooptreeWorkload,
    LooptreeWorkloadDependencyAnalyzer
)

from pytimeloop.fastfusion.exploration import explore_fusion_sets
from pytimeloop.fastfusion.fusionset import FusionSet
from pytimeloop.fastfusion.pareto import OpData, Pareto

from .level_mapper.compute import ComputeLevelMapper
from .level_mapper.exhaustive import ExhaustiveLevelMapper
from .level_mapper.top_level import TopLevelMapper
from .stepped_model import Stats, SteppedModel, SteppedModelState


def mapper(config, spec, tmp_path, verbose_stream=None):
    workload = LooptreeWorkload.parse_cfg(config.root['problem'])
    analyzer = LooptreeWorkloadDependencyAnalyzer(workload)

    def print_info(s):
        if verbose_stream is not None:
            print(s, file=verbose_stream)

    adj_list = get_neighbors(workload)

    einsum_name_to_id = workload.einsum_name_to_id()
    per_einsum_mappings = deque()
    for einsum_name, einsum_id in einsum_name_to_id.items():
        tensors = (
            workload.tensors_read_by_einsum(einsum_id)
            |
            workload.tensors_written_by_einsum(einsum_id)
        )
        fusable_tensors = tensors & get_intermediate_tensors(workload)

        print_info(f'Generating mappings for Einsum {einsum_name}')
        data = layer_mapper(config,
                            einsum_name,
                            fusable_tensors,
                            adj_list[einsum_id],
                            workload,
                            analyzer,
                            spec,
                            tmp_path)
        per_einsum_mappings.append((
            einsum_id,
            [FusionSet({op_comp}, pareto) for op_comp, pareto in data.items()]
        ))
    print_info('Exploring fusion sets')
    explore_fusion_sets(per_einsum_mappings, verbose_stream)


def layer_mapper(config,
                 name_of_einsum_to_eval,
                 fusable_tensors,
                 neighbors,
                 workload,
                 analyzer,
                 spec,
                 tmp_path):
    einsum_name_to_id = workload.einsum_name_to_id()
    id_of_einsum_to_eval = einsum_name_to_id[name_of_einsum_to_eval]

    bindings, max_spatial, max_capacity = get_hardware_levels(spec.architecture)

    ranks = workload.einsum_ospace_dimensions(id_of_einsum_to_eval)
    tensors = (
        workload.tensors_read_by_einsum(id_of_einsum_to_eval)
        |
        workload.tensors_written_by_einsum(id_of_einsum_to_eval)
    )

    # Shape is given as *inclusive* (min, max) by workload
    einsum_shape = {
        rank_id: workload.get_rank_shape(rank_id)[1]+1 for rank_id in ranks
    }

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
        model.add_level_uneven(state,
                               level,
                               temporal_loops,
                               spatial_loops,
                               retained_tensors)
        return Stats()



    cur_mapper = None
    for hw_level in reversed(range(1, len(bindings))):
        if hw_level in max_capacity:
            level_max_cap = max_capacity[hw_level]
        else:
            level_max_cap = None
        if cur_mapper is None:  # Compute level
            cur_mapper = ComputeLevelMapper(hw_level,
                                            ranks,
                                            tensors,
                                            max_spatial=max_spatial[hw_level],
                                            max_capacity=level_max_cap,
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
                                               max_capacity=level_max_cap,
                                               can_bypass=True,
                                               lower_mapper=cur_mapper,
                                               analyzer=analyzer,
                                               partial_model=partial(partial_model,
                                                                     level=hw_level),
                                               step_back_model=step_back_model)

    hw_level = 0
    if hw_level in max_capacity:
        level_max_cap = max_capacity[hw_level]
    else:
        level_max_cap = None
    cur_mapper = TopLevelMapper(hw_level,
                                ranks,
                                tensors,
                                fusable_tensors,
                                id_of_einsum_to_eval,
                                neighbors=neighbors,
                                lower_mapper=cur_mapper,
                                model=model,
                                bits_per_word=8,
                                analyzer=analyzer,
                                partial_model=partial(partial_model, level=0),
                                step_back_model=step_back_model,
                                max_spatial=max_spatial[hw_level],
                                max_capacity=level_max_cap)

    cur_mapper.run(einsum_shape)

    result = cur_mapper.get_result()
    before_pareto_size = sum(v.shape[0] for v in result.values())

    result_dict = {}
    op_data = OpData(frozenset({id_of_einsum_to_eval}), frozenset(tensors))
    after_pareto_size = 0
    for op_comp, data in result.items():
        result_dict[op_comp] = Pareto({op_data: data})
        after_pareto_size += sum(
            v.shape[0] for v in result_dict[op_comp].data.values()
        )

    print('mapspace size:', before_pareto_size)
    print('mapspace after pareto size:', after_pareto_size)

    return result_dict


def get_hardware_levels(arch):
    bindings = {}
    fanout = {}
    max_capacity = {}
    for node in arch['nodes']:
        bindings_id = len(bindings)
        bindings[bindings_id] = node['name']
        fanout[bindings_id] = (node.spatial.meshX, node.spatial.meshY)
        attribute = node.attributes
        if 'width' in attribute and 'depth' in attribute:
            width = attribute.width
            depth = attribute.depth
            datawidth = attribute.datawidth
            if all(x is not None for x in (width, depth, datawidth)):
                max_capacity[bindings_id] = \
                    attribute.width * attribute.depth / attribute.datawidth
    return bindings, fanout, max_capacity


def get_neighbors(workload):
    adj_list = defaultdict(lambda: list())
    for einsum_u_id in workload.einsum_id_to_name():
        for einsum_v_id in workload.einsum_id_to_name():
            u_written_tensor = workload.tensors_written_by_einsum(einsum_u_id)
            v_read_tensors = workload.tensors_read_by_einsum(einsum_v_id)
            if u_written_tensor is not None and u_written_tensor in v_read_tensors:
                adj_list[einsum_u_id].append(einsum_v_id)
                adj_list[einsum_v_id].append(einsum_u_id)
                continue
            u_read_tensors = workload.tensors_read_by_einsum(einsum_u_id)
            v_written_tensor = workload.tensors_written_by_einsum(einsum_v_id)
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
                    result.add(tensor)
                    break

    return result
