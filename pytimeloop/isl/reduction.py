import islpy as isl

def make_reduction_map(space, dims_out_first, n_dims_out):
    """
    Creates a reduction map to remove a dimension, e.g., { [x, y] -> [x, y, z]}.
    """
    return isl.Map.identity(space.map_from_set())\
                .project_out(isl.dim_type.in_, dims_out_first, n_dims_out)
