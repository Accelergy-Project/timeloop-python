from math import ceil
import pydot
from pytimeloop.timeloopfe.v4.arch import Storage, Container, ArchNode, Branch, Leaf
from typing import Dict, List, Set, Union
from ...common.processor import Processor
from ...v4 import Specification
from ..specification import Specification


def wt(text: str, width: int = 20) -> str:
    nlines = ceil(len(text) / width)
    width = ceil(len(text) / nlines)
    # Wrap text by inserting a \n every 15 characters
    return "-\n".join([text[i : i + width] for i in range(0, len(text), width)])


class ToDiagramProcessor(Processor):
    """Generates a Graphviz diagram of the architecture."""

    def __init__(
        self,
        container_names: Union[str, List[str]] = (),
        ignore_containers: Union[str, List[str]] = (),
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ds2color = None
        self.counter = 0
        if isinstance(container_names, str):
            container_names = [container_names]
        self.container_names = container_names
        if isinstance(ignore_containers, str):
            ignore_containers = [ignore_containers]
        self.ignore_containers = ignore_containers

    def get_ds2color(self, spec: Specification) -> Dict[str, str]:
        ds_names = [ds.name for ds in spec.problem.shape.data_spaces]
        ds2color = {
            "Inputs": "blue",
            "Weights": "green",
            "Outputs": "red",
        }
        other_colors = [
            "purple",
            "orange",
            "yellow",
            "cyan",
            "magenta",
            "brown",
            "grey",
        ]
        for ds in ds_names:
            if ds not in ds2color:
                ds2color[ds] = other_colors.pop(0)
        return ds2color

    def get_node_kwargs(self, **kwargs):
        defaults = {
            "shape": "box",
            "style": "filled",
            "fillcolor": "#ffffff",
            "penwidth": 1,
            "margin": "0,0.1",
        }
        defaults.update(kwargs)
        return defaults

    def container_kwargs(self, **kwargs):
        return self.get_node_kwargs(
            shape="box3d", style='"filled,dashed"', fillcolor="#ffffff", **kwargs
        )

    def storage_kwargs(self, shape: str, **kwargs):
        return self.get_node_kwargs(fillcolor="#fdeada", shape=shape, **kwargs)

    def section_kwargs(self, **kwargs):
        kw = self.get_node_kwargs(
            fillcolor="#f0f0f0",
            style='"filled"',
            fontsize=16,
        )
        if "margin" not in kwargs:
            del kw["margin"]
        return kw

    def get_edge(
        self,
        from_node: pydot.Node,
        to_node: pydot.Node,
        color: str = "black",
        vertical: bool = False,
        horizontal: bool = False,
        **kwargs,
    ) -> pydot.Edge:
        defaults = {
            "penwidth": 2,
            "arrowhead": "normal",
            "arrowtail": "none",
            "dir": "both",
        }
        defaults.update(kwargs)
        edge = pydot.Edge(
            from_node,
            to_node,
            color=color,
            **defaults,
        )
        if vertical:
            edge.set_constraint("vertical")
        if horizontal:
            edge.set_constraint("horizontal")
        return edge

    def add_edge(self, graph, *args, **kwargs) -> pydot.Edge:
        e = self.get_edge(*args, **kwargs)
        graph.add_edge(e)
        return e

    def network_kwargs(self, **kwargs):
        return self.get_node_kwargs(shape="oval", **kwargs)

    def get_node_names(self, n: ArchNode, ds_reused_temporally: List[str]) -> List[str]:
        fanout = n.spatial.get_fanout()
        name = [wt(n.name)]

        def attrstr(n, attr):
            if isinstance(attr, list):
                return [attrstr(n, a) for a in attr]
            return f"{attr}: {n.attributes.get(attr, None)}"

        # if isinstance(n, Storage):
        #     name += attrstr(n, ["datawidth", "width"])
        #     if ds_reused_temporally:
        #         name += attrstr(n, ["depth"])
        name = "\n".join(name)

        if fanout == 1:
            names = [name]
        elif fanout <= 4:
            names = [f"{name}\n{i}" for i in range(1, fanout + 1)]
        else:
            names = [
                f"{name}\n1",
                f"{name}\n2",
                f"{name}\n3...",
                f"{name}\n{fanout}",
            ]
        return names

    def edge_and_dummies(self, graph: pydot.Graph, stack_under=None, **edge_kwargs):
        n1 = pydot.Node(
            f"{self.counter}", label="From", **self.storage_kwargs(shape="box3d")
        )
        n2 = pydot.Node(
            f"{self.counter+1}", label="To", **self.storage_kwargs(shape="box3d")
        )
        graph.add_node(n1)
        graph.add_node(n2)
        self.counter += 2
        self.add_edge(graph, n1, n2, horizontal=True, **edge_kwargs)
        if stack_under:
            self.add_edge(graph, stack_under, n1, **edge_kwargs, style="invis")
        return n2

    def process(self, spec: Specification):
        # Add an argument to make nodes as small as possible (while still fitting the label)
        graph = pydot.Dot(
            graph_type="digraph",
            penwidth=3,
            splines="curve",
            rankdir="TB",
            nodesep=0.2,
            ranksep=0.02,
        )
        ds_names = [ds.name for ds in spec.problem.shape.data_spaces]
        ds2color = self.get_ds2color(spec)
        first_subgraph = None

        added = []

        for i, g in enumerate(
            self.create_sub_graph(
                spec,
                "root",
                spec.architecture.nodes,
                ds_names,
                set(),
                ds2color,
            )
        ):
            if (
                not self.container_names
                or any(f" {c} " in g.get_label() for c in self.container_names)
            ) and not any(f" {c} " in g.get_label() for c in self.ignore_containers):
                if i == 0 and len(g.get_nodes()) == 1:
                    continue
                graph.add_subgraph(g)
                if first_subgraph is None:
                    first_subgraph = g
                added.append(g)

        # Create a legened
        legend = pydot.Cluster("legend", label="Legend", **self.section_kwargs())
        graph.add_subgraph(legend)

        # Add invisible nodes to make the legend stack vertically
        # legend.add_node(lg := pydot.Node("legend invisible top", style="invis"))

        # Edge from the last node of first_subgraph to the first node of the legend
        # self.add_edge(legend, lg, first_subgraph.get_nodes()[-1], style="invis")

        dspace_sub = pydot.Cluster(
            "Data Spaces", label="Data Spaces", **self.section_kwargs()
        )
        legend.add_subgraph(dspace_sub)
        for ds, color in ds2color.items():
            kwargs = {}
            if spec.problem.shape.name2dataspace(ds).read_write:
                kwargs["arrowtail"] = "normal"

            self.edge_and_dummies(
                dspace_sub, None, color=color, label=f"{ds}\nMovement", **kwargs
            )

        element_sub = pydot.Cluster(
            "Elements", label="Elements", **self.section_kwargs()
        )
        legend.add_subgraph(element_sub)

        element_sub.add_node(
            pydot.Node("Storage", **self.storage_kwargs(shape="cylinder"))
        )

        element_sub.add_node(pydot.Node("Container", **self.container_kwargs()))
        element_sub.add_node(
            pydot.Node("Other Component", **self.storage_kwargs(shape="box3d"))
        )

        element_sub.add_node(pydot.Node("Container", **self.container_kwargs()))
        element_sub.add_node(net := pydot.Node("Network", **self.network_kwargs()))

        # Add an edge from the last node of the legend to the first node of the first subgraph
        for subgraph in added:
            self.add_edge(graph, net, subgraph.get_nodes()[0], style="invis")

        return graph

    def create_sub_graph(
        self,
        spec: Specification,
        name: str,
        nodes: List[ArchNode],
        dataspaces: List[str],
        dataspaces_used: Set[str],
        ds2color: Dict[str, str],
        my_fanout: int = 1,
        cur_fanout: int = 1,
        prev_name: str = None,
    ) -> list:
        self.logger.info(f"Creating subgraph for {name}")
        label = f"{my_fanout}x {name} ({cur_fanout} total)"
        graph = pydot.Cluster(name, label=label, **self.section_kwargs())
        nodes = [n for n in nodes]

        cur_ports = {}

        if dataspaces_used:
            node = pydot.Node(
                f"From outside {name}...",
                label=f"From {prev_name or 'System'}",
                **self.get_node_kwargs(),
            )
            graph.add_node(node)
            cur_ports = {ds: node for ds in dataspaces_used}

        while nodes:
            n = nodes.pop(0)
            if isinstance(n, Branch):
                nodes = n.nodes + nodes
                continue

            # Must be a leaf node. Let's add it to the graph.
            self.logger.info(f"  Adding node {n.name} to {name}")
            is_container = isinstance(n, Container)
            fanout = n.spatial.get_fanout()

            # Figure out which dataspaces are kept in this node
            ds_kept = [
                ds for ds in dataspaces if ds not in n.constraints.dataspace.bypass
            ]
            ds_stored = ds_kept if not isinstance(n, Container) else []
            ds_reused_temporally = [
                ds
                for ds in ds_kept
                if ds not in n.constraints.temporal.no_reuse
                and ds not in n.constraints.dataspace.no_coalesce
            ]
            names = self.get_node_names(n, ds_reused_temporally)

            kw = {}
            if is_container:
                node_kwargs = self.container_kwargs()
            else:
                if not ds_reused_temporally:
                    node_kwargs = self.storage_kwargs(shape="box3d")
                else:
                    node_kwargs = self.storage_kwargs(shape="cylinder")

            subnodes = []

            # If there are multiple nodes, create a cluster for them
            sub_cluster = graph
            if len(names) > 1:
                sub_cluster = pydot.Cluster(f"{n.name} cluster", style="invis")
                graph.add_subgraph(sub_cluster)

            # Create subnodes
            for cname in names:
                node = pydot.Node(
                    cname,
                    label='\"...\"' if "..." in cname else cname,
                    **self.get_node_kwargs(**node_kwargs),
                )
                sub_cluster.add_node(node)
                subnodes.append(node)

            # Create multicast & unicast network nodes
            if len(subnodes) > 1:
                mcast = pydot.Node(
                    f"{cname} multicast",
                    label=f"Multicast x{fanout}",
                    **self.network_kwargs(),
                )
                ucast = pydot.Node(
                    f"{cname} unicast",
                    label=f"Unicast x{fanout}",
                    **self.network_kwargs(),
                )

            # Add connections for each dataspace
            for ds in set(cur_ports.keys()) & set(ds_kept):
                kwargs = {"color": ds2color[ds]}
                if spec.problem.shape.name2dataspace(ds).read_write:
                    kwargs["arrowtail"] = "normal"
                is_multicast = ds not in n.constraints.spatial.no_reuse

                if len(subnodes) > 1:
                    net = mcast if is_multicast else ucast
                    graph.add_node(net)
                    kw = {**kwargs}
                    f = 1 if is_multicast else fanout
                    self.add_edge(graph, cur_ports[ds], net, **kw)
                    cur_ports[ds] = net

                    start = net
                    for i, s in enumerate(subnodes):
                        self.add_edge(graph, start, s, **kwargs)

                else:  # Only one subnode
                    s = subnodes[0]
                    self.add_edge(graph, cur_ports[ds], s, **kwargs)

            for ds in ds_stored:
                cur_ports[ds] = subnodes[-1]

            if isinstance(n, Container):
                return [graph] + self.create_sub_graph(
                    spec,
                    n.name,
                    nodes,
                    dataspaces,
                    dataspaces_used | set(cur_ports.keys()),
                    ds2color,
                    fanout,
                    cur_fanout * fanout,
                    name,
                )

        return [graph]
