import islpy as isl

def make_reduction_map(space, dims_out_first, n_dims_out):
    """
    Creates a reduction map to remove a dimension, e.g., { [x, y] -> [x, y, z]}.
    """
    return isl.Map.identity(space.map_from_set())\
                .project_out(isl.dim_type.in_, dims_out_first, n_dims_out)


def make_reduction_map_from_mask(space, mask):
    """
    Creates a reduction map that removes dimensions marked True in `mask`,
    e.g., if mask is [False, True, False] and space is { [x, y, z] }, the
    result is { [x, z] -> [x, y, z] }
    """
    isl_map = isl.Map.identity(space.map_from_set())
    for i, is_reduced in reversed(list(enumerate(mask))):
        if is_reduced:
            isl_map = isl_map.project_out(isl.dim_type.in_, i, 1)
    return isl_map
