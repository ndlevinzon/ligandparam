from typing import Optional, Any

from ligandparam.stages.abstractstage import AbstractStage
from ligandparam.log import get_logger


class TestStage(AbstractStage):
    """Minimal stage used for testing the stage interface (dev/stub; not exported)."""

    def __init__(self, name, **kwargs) -> None:
        self.name = name
        self._parse_inputoptions(kwargs)
        return
    
    def execute(self, dry_run=False, nproc: Optional[int]=None, mem: Optional[int]=None) -> Any:
        self.logger.info("This worked!")
        return

