import bindings
from bindings import Dimension, ID
from .config import Config
from .model import ArchSpecs
from .problem import Workload

from io import StringIO
import logging
import sys


class MapSpace(bindings.MapSpace):
    @staticmethod
    def parse_and_construct(config: Config, arch_constraints: Config,
                            arch_specs: ArchSpecs, workload: Workload,
                            log_level=logging.INFO):
        _, native_config_node = config.get_native()
        _, native_arch_const_node = arch_constraints.get_native()

        logger = logging.getLogger(__name__ + '.' + MapSpace.__name__)
        logger.setLevel(log_level)

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = captured_stdout = StringIO()
            sys.stderr = captured_stderr = StringIO()
            mapspace = bindings.MapSpace.parse_and_construct(
                native_config_node, native_arch_const_node, arch_specs,
                workload)
        finally:
            sys.stdout = old_stderr
            sys.stderr = old_stderr
        if captured_stdout.getvalue():
            logger.info(captured_stdout.getvalue())
        if captured_stderr.getvalue():
            logger.error(captured_stderr.getvalue())

        return mapspace

