from collections import defaultdict
import pydot
from typing import Any
from pytimeloop.fastfusion.sim import Tiling, TensorStorage, Loop
from pytimeloop.fastfusion.util import expfmt
from pytimeloop.fastfusion.pareto import IN_PROGRESS_STATS

PYDOT_NODE_DEFAULTS = {
    "shape": "box",
    "fontsize": "10",
    # "margin": "0.0",
    "margin": "0.1,0.0",
}

class Node:
    def __init__(self):
        self.this_level = []
        self.children = []

    def access_level(self, index: int) -> "Node":
        if index == -1:
            return self
        if index == None:
            return self.children[-1].access_level(None) if self.children else self
        return self.children[-1].access_level(index - 1)

    def _to_yaml(self):
        this_level = [t.to_yaml() for t in self.this_level]
        if self.children:
            this_level.append(
                {
                    "type": "sequential",
                    "children": [c._to_yaml() for c in self.children],
                }
            )
        return this_level

    def to_yaml(self):
        return {"mapping": "fused", "nodes": self._to_yaml()}

    def to_pydot(self, graph, parent=None, invisible_root: bool = True):
        label_lines = []
        for t in self.this_level:
            label_lines.append(t.pydot_str() if hasattr(t, "pydot_str") else str(t))
        node_label = "\n".join(sorted(label_lines))
        if invisible_root:
            node = None
        else:
            node = pydot.Node(id(self), label=node_label, **PYDOT_NODE_DEFAULTS)
            graph.add_node(node)
        if parent:
            graph.add_edge(pydot.Edge(parent, node))
        for child in self.children:
            child.to_pydot(graph, node, invisible_root=False)

    def add_stats(self, stats: dict[str, Any]):
        if self.children:
            self.children[-1].add_stats(stats)
        else:
            for k, v in stats.items():
                self.this_level.append(f"{k}: {expfmt(v)}")


def tilings2looptree(mappings: dict[str, Tiling], stats: dict[str, Any], tensors: dict[str, list[TensorStorage]], partial_stats: dict[str, Any]):
    prev_tiling = None
    root = Node()
    einsum_ids = list(mappings.keys())
    assert set(einsum_ids) == set(stats.keys())
    assert set(einsum_ids) == set(tensors.keys())
    for einsum_id in einsum_ids:
        tiling = mappings[einsum_id]
        index = (
            prev_tiling.shared_loop_index(tiling.tensor_names) if prev_tiling else -1
        )
        n = root.access_level(index)
        loops = tiling.loops[index:] if index != -1 else tiling.loops
        total_resources = defaultdict(lambda: 0)
        for l in loops:
            n.children.append(Node())
            n = n.children[-1]
        n.children.append(Node()) # Leaf node
        for tensor in tiling.tensors:
            root.access_level(tensor.above_loop_index).this_level.append(tensor)
        for i, l in enumerate(loops):
            root.access_level(index + i + 1).this_level.append(l)
        root.add_stats(stats[einsum_id])
        last_level = root.access_level(None).this_level
        for tensor in tiling.tensors:
            if tensor not in last_level:
                last_level.append(tensor)
                total_resources[tensor.backer_id] += tensor.tile_size
        for tensor in tensors[einsum_id]:
            if tensor not in last_level:
                last_level.append(tensor.pydot_str() + "**")
                total_resources[tensor.backer_id] += tensor.tile_size
        for k, v in total_resources.items():
            last_level.append(f"({k}) TOTAL: {expfmt(v)}")
            
        for k, v in partial_stats[einsum_id].items():
            last_level.append(f"_PARTIAL {k}: {expfmt(v)}")


        prev_tiling = tiling
    return root

def tilings2svg(mappings: dict[str, Tiling], stats: dict[str, Any], tensors: dict[str, list[TensorStorage]], partial_stats: dict[str, Any]):
    root = tilings2looptree(mappings, stats, tensors, partial_stats)
    graph = pydot.Dot(graph_type="digraph")
    root.to_pydot(graph)
    return graph.create_svg()

def tilings2yaml(mappings: dict[str, Tiling], stats: dict[str, Any], tensors: dict[str, list[TensorStorage]], partial_stats: dict[str, Any]):
    root = tilings2looptree(mappings, stats, tensors, partial_stats)
    return root.to_yaml()

