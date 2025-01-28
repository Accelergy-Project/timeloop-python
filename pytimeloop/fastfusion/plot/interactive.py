from numbers import Number
from typing import Any, Iterable, Optional, Union
import plotly
import pydot
from IPython.display import SVG, display
import plotly.graph_objs as go
from ipywidgets import Output, VBox, HBox
import pandas as pd
import plotly.express as px

from pytimeloop.fastfusion.sim import Loop, TensorStorage, Tiling
from pytimeloop.fastfusion.util import expfmt
from pytimeloop.fastfusion.plot.looptree import tilings2svg
from pytimeloop.fastfusion.pareto import MAPPING, STATS, TENSORS, IN_PROGRESS_STATS, MAPPING_HASH

import pandas as pd

def mapping2svg(mapping: pd.Series):
    return SVG(tilings2svg(mapping[MAPPING], mapping.get(STATS, None)))

def mapping2svg(mapping: pd.Series):
    return SVG(tilings2svg(mapping[MAPPING], mapping.get(STATS, None)))

def diplay_mappings_on_fig(fig: plotly.graph_objs.FigureWidget, data: dict[str, pd.DataFrame]):
    # fig = go.FigureWidget(fig)
    out = Output()
    DUAL_OUT = False
    @out.capture()
    def display_mapping(trace, points, selector):
        if not points.point_inds:
            return
        out.clear_output()
        d = data[trace.name]
        index = points.point_inds[0]
        display(mapping2svg(d.iloc[index]))
        backing_tensors = set(t for tn in d.iloc[index][MAPPING].values() for t in tn.tensors)
        backing_tensors = TensorStorage.get_backing_stores(backing_tensors)
        for t in sorted(backing_tensors):
            print(f"{t.__repr__()},")
        for t in sorted(list(d.iloc[index][MAPPING].values())[-1].tensors):
            print(f"{t.__repr__()},")
        for v in d.iloc[index][MAPPING].values():
            print(v)
    out2 = Output()
    @out2.capture()
    def display_mapping_2(trace, points, selector):
        if not points.point_inds:
            return
        out2.clear_output()
        d = data[trace.name]
        index = points.point_inds[0]
        display(mapping2svg(d.iloc[index]))
        backing_tensors = set(t for tn in d.iloc[index][MAPPING].values() for t in tn.tensors)
        backing_tensors = TensorStorage.get_backing_stores(backing_tensors)
        for t in sorted(backing_tensors):
            print(f"{t.__repr__()},")
        for t in sorted(list(d.iloc[index][MAPPING].values())[-1].tensors):
            print(f"{t.__repr__()},")
        for v in d.iloc[index][MAPPING].values():
            print(v)
    for i in fig.data:
        i.on_hover(display_mapping)
        i.on_click(display_mapping_2)
    if not DUAL_OUT:
        return VBox([fig, out])
    out.layout.width = '50%'
    out2.layout.width = '50%'
    return VBox([fig, HBox([out, out2])])


def plotly_show(
    data: Union[pd.DataFrame, dict[str, pd.DataFrame]],
    x: str,
    y: str,
    category: Optional[str] = None,
    title: Optional[str] = None,
    show_mapping: Optional[bool] = True,
    logscales: bool = False
):
    fig = go.FigureWidget()
    if isinstance(data, dict):
        for k, v in data.items():
            v.sort_values(by=[x, y], inplace=True)
            fig.add_scatter(x=v[x], y=v[y], name=k, line={"shape": 'hv'})
    else:
        data.sort_values(by=[x, y], inplace=True)
        fig.add_scatter(x=data[x], y=data[y], name="", line={"shape": 'hv'})
        data = {"" : data}
    if title is not None:
        fig.update_layout(title=title)
    if logscales:
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log")
    fig.update_xaxes(title_text=x)
    fig.update_yaxes(title_text=y)
    fig.update_layout(showlegend=True)
    # fig = px.scatter(data, x=x, y=y, color=category, title=title, log_x=logscales, log_y=logscales)
    if show_mapping:
        return diplay_mappings_on_fig(fig, data)
    return fig
