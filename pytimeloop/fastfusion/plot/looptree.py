from collections import defaultdict
import pydot
from typing import Any, Iterable
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

    def to_pydot(self, graph, parent=None, invisible_root: bool = False):
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
            reservations = "\n".join(sorted(f"[{k}] {expfmt(v)}" for k, v in self.get_reservations().items()))
            graph.add_edge(pydot.Edge(parent, node, label=reservations))
        for child in self.children:
            child.to_pydot(graph, node, invisible_root=False)

    def add_stats(self, stats: dict[str, Any]):
        if self.children:
            self.children[-1].add_stats(stats)
        else:
            for k, v in stats.items():
                self.this_level.append(f"{k}: {expfmt(v)}")
                
    def get_reservations(self) -> dict[str, int]:
        reservations = defaultdict(lambda: 0)
        for c in self.children:
            for k, v in c.get_reservations().items():
                reservations[k] = max(reservations[k], v)
        for t in self.this_level:
            if isinstance(t, TensorStorage):
                reservations[t.backer_id] += t.tile_size
        return reservations

def tilings2looptree(mappings: dict[str, Tiling], stats: dict[str, Any], tensors: dict[str, list[TensorStorage]], partial_stats: dict[str, Any], skip_backing_tensors: Iterable[str] = ()):
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
        id2tensor = defaultdict(lambda: [])
        for t in tiling.tensors:
            id2tensor[t.tensor_id].append(t)
        id2tensor = {k: sorted(v, key=lambda x: (x.above_loop_index, x.backer_id)) for k, v in id2tensor.items()}
        for tensor_id, storages in id2tensor.items():
            if tensor_id in skip_backing_tensors:
                storages = storages[1:]
            for tensor in storages:
                if tensor not in n.this_level:
                    root.access_level(tensor.above_loop_index).this_level.append(tensor)
        for i, l in enumerate(loops):
            root.access_level(index + i + 1).this_level.append(l)
        last_level = root.access_level(None).this_level
        first_level = root.access_level(0).this_level
        for tensor in tensors[einsum_id]:
            if tensor.tensor_id not in skip_backing_tensors:
                if tensor not in mappings[einsum_id].tensors:
                    # tensor = TensorStorage(
                    #     f"*{tensor.tensor_id}", 
                    #     tensor.backer_id, 
                    #     tensor.above_loop_index, 
                    #     tensor.tile_size
                    # )
                    first_level.append(tensor)
                    total_resources[tensor.backer_id] += tensor.tile_size
        for k, v in total_resources.items():
            last_level.append(f"({k}) TOTAL: {expfmt(v)}")
        root.add_stats(stats[einsum_id])
        for k, v in partial_stats[einsum_id].items():
            last_level.append(f"_PARTIAL {k}: {expfmt(v)}")
        prev_tiling = tiling
    return root

def tilings2svg(mappings: dict[str, Tiling], stats: dict[str, Any], tensors: dict[str, list[TensorStorage]], partial_stats: dict[str, Any]):
    root = tilings2looptree(mappings, stats, tensors, partial_stats)
    graph = pydot.Dot(graph_type="digraph", ranksep="0.2", nodesep="0.2")
    root.to_pydot(graph)
    return graph.create_svg()

def tilings2yaml(mappings: dict[str, Tiling], stats: dict[str, Any], tensors: dict[str, list[TensorStorage]], partial_stats: dict[str, Any]):
    root = tilings2looptree(mappings, stats, tensors, partial_stats)
    return root.to_yaml()

