from functools import partial

from ruamel.yaml import YAML
yaml = YAML(typ='safe')

from bindings.looptree import LooptreeWorkloadDependencyAnalyzer

from pytimeloop.fastfusion.pareto import OpData, Pareto

from .level_mapper.exhaustive import ExhaustiveLevelMapper
from .level_mapper.top_level import TopLevelMapper
from .stepped_model import Stats, SteppedModel, SteppedModelState


def mapper(config, spec, workload, name_of_einsum_to_eval, tmp_path):
    einsum_name_to_id = workload.einsum_name_to_id()
    id_of_einsum_to_eval = einsum_name_to_id[name_of_einsum_to_eval]

    bindings, max_spatial = get_hardware_levels(spec.architecture)

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
                                neighbors,
                                lower_mapper,
                                partial_model,
                                step_back_model,
                                max_spatial,
                                max_capacity)

    cur_mapper.run(einsum_shape)

    result = cur_mapper.get_result()
    op_data = OpData(frozenset({id_of_einsum_to_eval}), frozenset(tensors))
    pareto = Pareto({op_data: result})

    return pareto


def get_hardware_levels(arch):
    bindings = {}
    fanout = {}
    for node in arch['nodes']:
        bindings_id = len(bindings)
        bindings[bindings_id] = node['name']
        fanout[bindings_id] = (node.spatial.meshX, node.spatial.meshY)
    return bindings, fanout

