import matplotlib.pyplot as plt

from pytimeloop.looptree.des import LooptreeOutput


def plot_occupancy_graph(output: LooptreeOutput, workload):
    einsum_rank_to_shape = {
        einsum: {
            rank: workload.get_rank_shape(rank)
            for rank in workload.einsum_ospace_dimensions(einsum)
        }
        for einsum in workload.einsum_id_to_name()
    }

    
