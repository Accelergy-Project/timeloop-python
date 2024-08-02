import copy
import ruamel.yaml
from ...v4.specification import Specification
from ...v4 import arch, constraints
from ...v4.arch import Attributes, Nothing, Spatial
from ..nodes import isempty
import logging


def transpile(spec: Specification, for_model: bool = False):
    """Dump a v4 specification to v3 format.
    !@param spec Specification object to dump.
    !@param for_model If True, dump the specification for timelooop-model.
                      Else, for timeloop-mapper.
    !@return A string containing the dumped specification in V3 YAML format.
    """
    spec = copy.deepcopy(spec)
    prob = spec.problem
    top_node = spec.architecture
    constraint_list = []
    sparse_opt_list = []
    level = {
        "name": "top_level",
        "attributes": spec.variables,
        "local": [],
        "subtree": [],
    }
    stack = [(top_node, False)]
    meshX, meshY = 1, 1
    cur_power_gating = None
    prev_node = None
    arch_attrs = {}
    add_power_gate_next = False

    next_meshX, next_meshY = 1, 1

    first_node = True

    node = None

    while stack:
        meshX *= int(next_meshX)
        meshY *= int(next_meshY)
        next_meshX, next_meshY = 1, 1
        local = level["local"]
        node, is_parallel = stack.pop(0)
        is_container = isinstance(node, arch.Container)
        if not getattr(node, "enabled", True):
            logging.debug("Skipping disabled node %s", node.get_name())
            continue
        logging.debug("Processing node %s", node.get_name())
        if isinstance(node, arch.Parallel):
            stack = [(n, True) for n in node.nodes] + stack
            continue
        elif isinstance(node, arch.Branch):
            stack = [(n, False) for n in node.nodes] + stack
            continue
        elif isinstance(node, Nothing):
            continue

        has_fanout = isinstance(node, arch.Leaf) and (node.spatial.get_fanout() != 1)

        if isinstance(node, arch.Container):
            node: arch.Container
            arch_attrs.update(node.attributes)

        attrs = node.get("attributes", {})
        spatial = node.get("spatial", {})

        has_power_gating = attrs.get("has_power_gating", False)

        to_place = [node]

        if first_node and is_container:
            level["name"] = node.name + "_top_level"
            level["attributes"].update(node.attributes)

        first_node = False

        if is_container or has_fanout:
            if is_parallel:
                if is_container:
                    raise ValueError(
                        f"Containers under a Parallel are not "
                        f"supported: {node.name}"
                    )
                if has_fanout:
                    raise ValueError(
                        f"Components with fanout under a Parallel "
                        f"are not supported: {node.name}"
                    )

            next_meshX *= spatial.get("meshX", 1)
            next_meshY *= spatial.get("meshY", 1)

            if has_fanout:
                logging.debug("Adding dummy for %s", node.get_name())
                dummy_name = f"inter_{node.name}_spatial"
                dummy = arch.dummy_storage(dummy_name)
                dummy.constraints.combine(
                    constraints.dummy_constraints(prob, not has_fanout)
                )
                dummy.constraints.spatial.combine(node.constraints.spatial)

                # dummy.spatial = node.spatial
                logging.debug("Dummy name is %s", dummy.get_name())
                # dummy.attributes = node.attributes

            if has_fanout:
                to_place = [dummy]
            if not is_container:
                to_place.append(node)
                node.constraints.spatial = None
            for n in to_place:
                n.clean_empties()
                if n.get("attrs", None) is not None:
                    n.get("attrs").clean_empties()

        for node in to_place:
            attrs = node.get("attributes", {})
            if not (isinstance(node, arch.Container)):
                node.clean_empties()
                if "spatial" in node:
                    del node.spatial
                local.append(node)
                if "constraints" in node:
                    logging.debug("Adding constraints for %s", node.get_name())
                    for k, v in node.constraints.items():
                        v["type"] = k
                        v["target"] = node.name
                        logging.debug(
                            "Adding constraint %s for %s", v["type"], v["target"]
                        )
                        constraint_list.append(v)
                        if "permutation" in v:
                            v["permutation"] = "".join(v["permutation"])
                    node.pop("constraints")
                if "sparse_optimizations" in node:
                    sparse_opt_list.append(node.pop("sparse_optimizations"))
                    sparse_opt_list[-1]["name"] = node.name
                attrs = node.setdefault("attributes", Attributes())
                attrs["meshX"] = meshX
                attrs["meshY"] = meshY
                for k, v in arch_attrs.items():
                    if k not in attrs:
                        attrs[k] = v
                if not isinstance(node, arch.Network):
                    node.name += f"[1..{meshX*meshY}]"

                    if add_power_gate_next:
                        cur_power_gating = node.name.split("[")[0]
                    add_power_gate_next = has_power_gating

            attrs.setdefault(
                "power_gated_at",
                ruamel.yaml.scalarstring.DoubleQuotedScalarString(
                    cur_power_gating or node.name.split("[")[0]
                ),
            )
            meshX *= int(next_meshX)
            meshY *= int(next_meshY)
            next_meshX, next_meshY = 1, 1

    assert node is not None, "No nodes to add to the architecture"
    assert isinstance(node, arch.Compute), (
        "The last node in the architecture tree must be a compute node. "
        f"With 'class: compute'. Got node {node.get_name()} of type {type(node)}"
    )

    if not level["subtree"]:
        del level["subtree"]

    for constraint in constraint_list:
        if constraint["type"] == "dataspace":
            constraint["type"] = "bypass"
        if isinstance(constraint, constraints.Iteration):
            # Can't access directly because it may have been cleaaned at
            # this point
            if isinstance(constraint.get("factors", None), list):
                constraint.factors = ",".join(constraint.factors)

    rval = {
        "dumped_by_timeloop_front_end": True,
        "architecture": {"version": "0.4", "subtree": [level]},
        "architecture_constraints": {
            "targets": constraint_list if not for_model else []
        },
        "problem": spec.problem,
        "compound_components": spec.components,
        "mapping": constraint_list if for_model else [],
    }
    if sparse_opt_list:
        rval["sparse_optimizations"] = {"targets": sparse_opt_list}
    if spec.get("mapper", None):
        rval["mapper"] = spec.mapper
    if spec.get("mapspace", None):
        rval["mapspace"] = spec.mapspace
    if spec.get("globals", None):
        rval["globals"] = spec.globals
    if not isempty(spec.get("ART", None)):
        rval["ART"] = spec.ART
    if not isempty(spec.get("ART", None)):
        rval["ERT"] = spec.ERT
    return rval
