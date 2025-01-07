import copy
from abc import ABC, abstractmethod
from typing import Union
from pytimeloop.fastfusion.sim import Tiling, Loop, TensorStorage
import pydot

class Node(ABC):
    def __init__(self, name: str, children: Union[list["Node"], "Node"]):
        self.name = name
        self.children = children
    
    @abstractmethod
    def _merge_right(self, node: "Node") -> list["Node"]:
        pass

    def __str__(self, name=None):
        name = name if name else self.name
        if not self.children:
            return f"{name}"
        return f"{name}\n\t" + "\n\t".join(str(child).replace("\n", "\n\t") for child in self.children)

    def _rename(self, rank_renaming: dict[str, str], tensor_renaming: dict[str, str]):
        if not isinstance(self.name, str):
            self.name = self.name.rename(rank_renaming, tensor_renaming)
        for child in self.children:
            child._rename(rank_renaming, tensor_renaming)
        return self

    def rename(self, rank_renaming: dict[str, str], tensor_renaming: dict[str, str]):
        return copy.deepcopy(self)._rename(rank_renaming, tensor_renaming)
    
    def get_label(self):
        return str(self.name)
    
    def _to_dot(self, dot, parent):
        unique_name = f"{self.name}_{id(self)}"
        node = pydot.Node(unique_name, label=self.get_label())
        dot.add_node(node)
        if parent:
            dot.add_edge(pydot.Edge(parent, node))
        for child in self.children:
            child._to_dot(dot, node)
        return dot

class LoopNode(Node):
    def __init__(self, name: str, children: list[Node] | Node, storages_below: list[str]):
        super().__init__(name, children)
        self.storages_below = storages_below

    def _merge_right(self, node: Node) -> list[Node]:
        if self.name != node.name or not isinstance(node, LoopNode):
            return [self, node]
        assert len(node.children) <= 1, "Cannot right merge with >1 right children"

        if not self.children:
            self.children = node.children
        else:
            self.children += self.children.pop(-1)._merge_right(node.children[0])

        self.storages_below += node.storages_below
        return [self]
    
    def get_label(self):
        return "\n".join(
            [str(self.name)] + 
            [str(s) for s in self.storages_below]
        )
    
    def __str__(self):
        return super().__str__(f"{self.name} ({self.storages_below})")
    
    def _rename(self, rank_renaming: dict[str, str], tensor_renaming: dict[str, str]):
        super()._rename(rank_renaming, tensor_renaming)
        self.storages_below = [t.rename(rank_renaming, tensor_renaming) for t in self.storages_below]
        return self

class Root(LoopNode):
    def __init__(self, name: str, children: list[Node] | Node, storages_below: list[str] = []):
        super().__init__(name, children, storages_below)
    
    def merge_right(self, root: "Root") -> "Root":
        self, root = copy.deepcopy(self), copy.deepcopy(root)
        merged = self._merge_right(root)
        assert len(merged) == 1, "Root merge resulted in >1 root"
        return merged[0]
    
    def to_dot(self):
        dot = pydot.Dot()
        return self._to_dot(dot, None)
        

if __name__ == '__main__':
    children = [
        LoopNode("a1", [], ["a1"]),
        LoopNode("a2", [], ["a2"]),
    ]
    a = Root("a root", children)
    a.children = children
    children = [
        LoopNode("b1", [], ["b1"]),
    ]
    b = Root("b root", children)
    c = a._merge_right(b)
    print(a)
    print(b)
    print(c)
