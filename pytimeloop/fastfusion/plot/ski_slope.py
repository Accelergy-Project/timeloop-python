import matplotlib.axes as mpax
import matplotlib.pyplot as plt
import pandas as pd

from pytimeloop.fastfusion.pareto import makepareto


DATAFLOW_COLUMN = "dataflow"


def plot_ski_slope(data: pd.DataFrame,
                   categorize_by_dataflow: bool=False,
                   split_by_dataflow: bool=False,
                   ax: mpax.Axes=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if categorize_by_dataflow or split_by_dataflow:
        _add_dataflow_to_data(data)

    if not split_by_dataflow:
        data = makepareto(data)

    separated_datas = []
    labels = []
    if categorize_by_dataflow or split_by_dataflow:
        for dataflow, sub_df in data.groupby(by=DATAFLOW_COLUMN):
            separated_datas.append(makepareto(sub_df))
            labels.append(dataflow)
    else:
        separated_datas.append(data)
        labels.append(None)

    for label, sub_df in zip(labels, separated_datas):
        ax.plot(sub_df["Occupancy"], sub_df["Offchip_Ac"], label=label)

    ax.set_xlabel("Capacity")
    ax.set_ylabel("Off-chip Accesses")

    return fig, ax


def _add_dataflow_to_data(data: pd.DataFrame):
    data[DATAFLOW_COLUMN] = data["_Mappings"].apply(_dataflow_from_fulltiling)


def _dataflow_from_fulltiling(fulltiling: str):
    fulltiling = fulltiling.strip("[")
    fulltiling = fulltiling.strip("]")
    dataflow = []
    for term in fulltiling.split(","):
        if term[0] == "T":
            dataflow.append(int(term[1:].split(" size ")))
    return tuple(dataflow)

