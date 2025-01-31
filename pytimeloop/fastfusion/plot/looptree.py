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
    
    def get_all_storages(self, _entry: bool = True) -> list[TensorStorage]:
        storages = set(t for t in self.this_level if isinstance(t, TensorStorage))
        for c in self.children:
            storages.update(c.get_all_storages(_entry=False))
        return sorted(storages) if _entry else storages
    
    def get_backing_stores(self):
        return TensorStorage.get_backing_stores(self.get_all_storages())

def tilings2looptree(mappings: dict[str, Tiling], stats: dict[str, Any], skip_backing_tensors_in_right_branch: Iterable[str] = (), still_live_tensors: set[str] = (), shared_loop_index: int = -1):
    prev_tilings = []
    root = Node()
    einsum_ids = list(mappings.keys())
    if stats is not None:
        assert set(einsum_ids) == set(stats.keys())
    
    

    # If a tensor appears in non-back-to-back Einsums, then we need to store it for
    # all Einsums in between
    tensors_lifetimes = {e: [] for e in einsum_ids}
    all_tensors = set().union(*[set(t.tensors) for t in mappings.values()])
    backers = TensorStorage.get_backing_stores(all_tensors)
    for t in backers:
        first_appearance = min(i for i, ts in enumerate(mappings.values()) if t in ts.tensors)
        last_appearance = max(i for i, ts in enumerate(mappings.values()) if t in ts.tensors)
        if t.tensor_id in still_live_tensors:
            last_appearance = len(einsum_ids) - 1
        for i, l in enumerate(tensors_lifetimes.values()):
            if first_appearance <= i <= last_appearance and t not in l:
                l.append(t)
    
    # Add each Einsum to the tree
    for i, einsum_id in enumerate(einsum_ids):
        skip_backing_tensors = () #if i < len(einsum_ids) - 1 else skip_backing_tensors_in_right_branch
        tiling = mappings[einsum_id]
        if not prev_tilings:
            index = -1
        else:
            index = max(p.shared_loop_index(tiling.tensor_names) for p in prev_tilings)
        n = root.access_level(index)
        loops = tiling.loops[index:] if index != -1 else tiling.loops
        for l in loops:
            n.children.append(Node())
            n = n.children[-1]
            n.this_level.append(l)

        # Add the tensors
        n.children.append(Node()) # Leaf node
        n.children[-1].this_level.append(f"Einsum {einsum_id}")
        id2tensor = defaultdict(set)
        for t in sorted(tiling.tensors) + tensors_lifetimes[einsum_id]:
            id2tensor[t.tensor_id].add(t)
        id2tensor = {k: sorted(v) for k, v in id2tensor.items()}
        for tensor_id, storages in id2tensor.items():
            if tensor_id in skip_backing_tensors:
                storages = storages[1:]
            for tensor in storages:
                n = root.access_level(tensor.above_loop_index)
                # TODO if tensor not in n.this_level or tensor not in backers:
                if tensor not in n.this_level or tensor not in backers:
                    n.this_level.append(tensor)
        if stats is not None:
            root.add_stats(stats[einsum_id])
        # for k, v in partial_stats[einsum_id].items():
        #     last_level.append(f"_PARTIAL {k}: {expfmt(v)}")
        prev_tilings.append(tiling)

    n = root
    skip_backing_tensors_in_right_branch= set(skip_backing_tensors_in_right_branch)
    while n is not None:
        i = 0
        while i < len(n.this_level):
            t = n.this_level[i]
            if t in backers and t.tensor_id in skip_backing_tensors_in_right_branch:
                n.this_level.pop(i)
            else:
                i += 1
        n = n.children[-1] if n.children else None
        
    return root

def tilings2svg(mappings: dict[str, Tiling], stats: dict[str, Any], ):
    root = tilings2looptree(mappings, stats)
    graph = pydot.Dot(graph_type="digraph", ranksep="0.2", nodesep="0.2")
    root.to_pydot(graph)
    return graph.create_svg()

def tilings2yaml(mappings: dict[str, Tiling], stats: dict[str, Any]):
    root = tilings2looptree(mappings, stats)
    return root.to_yaml()

