"""@package pytimeloop.model

@brief Contains the classes that wrap around the C++ bindings for namespace model."""
from __future__ import annotations

# The C++ Bindings that we're wrapping.
import bindings

# Typing QoL
import typing

from .accelergy_interface import invoke_accelergy

import logging


class ArchSpecs(bindings.model.ArchSpecs):
    def __init__(self, config, is_sparse_topology: bool = False):
        super().__init__(config, is_sparse_topology)

    def generate_tables(self, config, semi_qualified_prefix, out_dir,
                        out_prefix, log_level=logging.INFO):
        # Setup logger
        logger = logging.getLogger(__name__ + '.' + __class__.__name__)
        logger.setLevel(log_level)

        root_node = config.get_root()
        if 'ERT' in root_node:
            logger.info('Found Accelergy ERT, replacing internal energy model')
            self.parse_accelergy_ert(root_node['ERT'])
            if 'ART' in root_node:
                logger.info(
                    'Found Accelergy ART, replacing internal area model')
                self.parse_accelergy_art(root_node['ART'])
        else:
            arch_cfg = root_node['architecture']
            if 'subtree' in arch_cfg or 'local' in arch_cfg:
                with open('tmp-accelergy.yaml', 'w+') as f:
                    f.write(config.dump_yaml_str())
                result = invoke_accelergy(['tmp-accelergy.yaml'],
                                          semi_qualified_prefix, out_dir)
                logger.info('Generated Accelergy ERT to replace internal '
                            'energy model')
                self.parse_accelergy_ert(result.ert)

                logger.info('Generated Accelergy ART to replace internal '
                            'energy model')
                self.parse_accelergy_art(result.art)


class SparseOptimizationInfo(bindings.model.SparseOptimizationInfo):
    def __init__(self, sparse_config, arch_specs: ArchSpecs):
        super().__init__(sparse_config, arch_specs)

## @todo The following segfaults on deallocation and needs to be fixed.
# class Engine(bindings.model.Engine):
#     """@brief The evaluation engine. Wraps around the C++ bindings."""
#     @property
#     def topology(self) -> Topology:
#         """@brief Returns the topology of the accelerator."""
#         # Gets the topology from the C++ bindings.
#         topology: bindings.model.Topology = super().get_topology()

#         # Converts it to a Python Topology, allowable like this because
#         # Python functions only differ on the functions they offer.
#         topology.__class__ = Topology

#         return topology

#     def __dict__(self):
#         """@brief Returns a dictionary of all the variables and their values."""
#         # Gets all the variables in the class.
#         var_names: list[str] = {
#             var_name for var_name in dir(self) if not callable(getattr(self, var_name))
#         } - {"__doc__", "__module__"}

#         # Creates a dictionary of all the variables and their values.
#         var_dict: dict[str, typing.Any] = {
#             var_name: getattr(self, var_name) for var_name in var_names
#         }

#         return var_dict


# class Topology(bindings.model.Topology):
#     """@brief Topology of the accelerator. Wraps around the C++ bindings."""
#     @property
#     def buffer_levels(self) -> list[Topology.BufferLevel]:
#         """@brief Returns a list of all the BufferLevels in the Topology.

#         Overrides the C++ bindings to return a list of Python BufferLevels."""
#         # Grabs the BufferLevels from the C++ bindings.
#         buffer_levels: list[bindings.model.BufferLevel] = super().buffer_levels

#         # Converts them all to Topology.BufferLevel
#         for level in buffer_levels:
#             # We can just change the __class__ var as the C++ bindings and Python
#             # classes contain the same variables, the Python bindings just have
#             # more QoL functions.
#             level.__class__ = Topology.BufferLevel
        
#         return buffer_levels
        

#     @property
#     def stats(self) -> Topology.Stats:
#         """@brief Returns the stats of this Topology.
        
#         Overrides the C++ bindings to return a Python Topology.Stats."""
#         # Grabs the stats from the C++ bindings.
#         stats = super().stats
#         # Converts it to a Python Topology.Stats, allowable like this because
#         # Python functions only differ on the functions they offer.
#         stats.__class__ = Topology.Stats

#         return stats


#     def __dict__(self) -> dict[str, typing.Any]:
#         # Gets all the variables in the class.
#         var_names: list[str] = {
#             var_name for var_name in dir(self) if not callable(getattr(self, var_name))
#         } - {"__doc__", "__module__"}

#         # Creates a dictionary of all the variables and their values.
#         var_dict: dict[str, typing.Any] = {
#             var_name: getattr(self, var_name) for var_name in var_names
#         }

#         return var_dict


#     class BufferLevel(bindings.model.BufferLevel):
#         """@brief A given BufferLevel of a topology. Wraps around the C++ bindings."""
    
#         def __dict__(self) -> dict[str, typing.Any]:
#             """@brief Returns a dictionary of all the variables and their values."""
#             # Gets all the variables in the class.
#             var_names: list[str] = {
#                 var_name for var_name in dir(self) if not callable(getattr(self, var_name))
#             } - {"__doc__", "__module__"}

#             # Creates a dictionary of all the variables and their values.
#             var_dict: dict[str, typing.Any] = {
#                 var_name: getattr(self, var_name) for var_name in var_names
#             }

#             return var_dict
#         '''
#         @todo Uncomment this block once the todo in buffer.cpp for Spec is done.
#         class Specs(bindings.model.BufferLevel.Specs):
#             """@brief Specifications of a BufferLevel. Wraps around the C++ bindings."""

#             def __dict__(self) -> dict[str, typing.Any]:
#                 """@brief Returns a dictionary of all the variables and their values."""
#                 # Gets all the variables in the class.
#                 var_names: list[str] = {
#                     var_name for var_name in dir(self) if not callable(getattr(self, var_name))
#                 } - {"__doc__", "__module__"}

#                 # Creates a dictionary of all the variables and their values.
#                 var_dict: dict[str, typing.Any] = {
#                     var_name: getattr(self, var_name) for var_name in var_names
#                 }

#                 return var_dict
#         '''

#         class Stats(bindings.model.BufferLevel.Stats):
#             """@brief Stats of a BufferLevel. Wraps around the C++ bindings."""

#             def __dict__(self) -> dict[str, typing.Any]:
#                 """@brief Returns a dictionary of all the variables and their values."""
#                 # Gets all the variables in the class.
#                 var_names: list[str] = {
#                     var_name for var_name in dir(self) if not callable(getattr(self, var_name))
#                 } - {"__doc__", "__module__"}

#                 # Creates a dictionary of all the variables and their values.
#                 var_dict: dict[str, typing.Any] = {
#                     var_name: getattr(self, var_name) for var_name in var_names
#                 }

#                 return var_dict
    
#     class Stats(bindings.model.Topology.Stats):
#         """@brief Stats of a Topology. Wraps around the C++ bindings."""
        
#         def __dict__(self) -> dict[str, typing.Any]:
#             """@brief Returns a dictionary of all the variables and their values."""
#             # Gets all the variables in the class.
#             var_names: list[str] = {
#                 var_name for var_name in dir(self) if not callable(getattr(self, var_name))
#             } - {"__doc__", "__module__"}

#             # Creates a dictionary of all the variables and their values.
#             var_dict: dict[str, typing.Any] = {
#                 var_name: getattr(self, var_name) for var_name in var_names
#             }

#             return var_dict
