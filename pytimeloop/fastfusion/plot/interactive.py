from numbers import Number
from typing import Any, Iterable, Optional
import plotly
import pydot
from IPython.display import SVG, display
import plotly.graph_objs as go
from ipywidgets import Output, VBox, HBox
import pandas as pd
import plotly.express as px

from pytimeloop.fastfusion.sim import Loop, Tiling
from pytimeloop.fastfusion.util import expfmt
from pytimeloop.fastfusion.plot.looptree import tilings2svg
from pytimeloop.fastfusion.pareto import MAPPING, STATS, TENSORS, IN_PROGRESS_STATS, MAPPING_HASH

import pandas as pd

def mapping2svg(mapping: pd.Series):
    return SVG(tilings2svg(mapping[MAPPING], mapping[STATS], mapping[TENSORS], mapping[IN_PROGRESS_STATS]))

def mapping2svg(mapping: pd.Series):
    return SVG(tilings2svg(mapping[MAPPING], mapping[STATS], mapping[TENSORS], mapping[IN_PROGRESS_STATS]))

def diplay_mappings_on_fig(fig: plotly.graph_objs.FigureWidget, data: pd.DataFrame):
    fig = go.FigureWidget(fig)
    out = Output()
    DUAL_OUT = False
    @out.capture(clear_output=True)
    def display_mapping(trace, points, selector):
        index = points.point_inds[0]
        display(mapping2svg(data.iloc[index]))
        all_tensors = set(
            t for tn in data.iloc[index][TENSORS].values() for t in tn
        )
        for t in sorted(all_tensors):
            print(f"{t.__repr__()},")
        for k, v in data.iloc[index][MAPPING_HASH].items():
            print(f"{k}: {v},")
    out2 = Output()
    @out2.capture(clear_output=True)
    def display_mapping_2(trace, points, selector):
        index = points.point_inds[0]
        display(mapping2svg(data.iloc[index]))
        all_tensors = set(
            t for tn in data.iloc[index][TENSORS].values() for t in tn
        )
        for t in sorted(all_tensors):
            print(f"{t.__repr__()},")
        for k, v in data.iloc[index][MAPPING_HASH].items():
            print(f"{k}: {v},")
    fig.data[0].on_hover(display_mapping)
    fig.data[0].on_click(display_mapping_2)
    if not DUAL_OUT:
        return VBox([fig, out])
    out.layout.width = '50%'
    out2.layout.width = '50%'
    return VBox([fig, HBox([out, out2])])


def plotly_show(
    data: pd.DataFrame,
    x: str,
    y: str,
    category: Optional[str] = None,
    title: Optional[str] = None,
    show_mapping: Optional[bool] = True,
    logscales: bool = False
):
    fig = px.scatter(data, x=x, y=y, color=category, title=title, log_x=logscales, log_y=logscales)
    if show_mapping:
        return diplay_mappings_on_fig(fig, data)
    return fig
