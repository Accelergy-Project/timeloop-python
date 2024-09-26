For layers that are TILED FUSED:
    - For NON-SHARED tensors GLB utilization: Summed
    - For SHARED tensors: GLB utilization maxed
    - Everything else: Summed
    - PE utilization : Summed if pipelined, else maxed
    
For layers that are NOT (TILED FUSED):
    - Overall GLB utilization: Maxed
    - Everything else: Summed
    - PE utilization : Maxed (pipeline)

is_pipelined:
- Record the spatial loops
- Invalid if utilization > array size