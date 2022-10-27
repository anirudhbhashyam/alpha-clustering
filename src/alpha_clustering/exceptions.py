from dataclasses import dataclass, field
from .logger import Logger

@dataclass
class AlphaValueError(ValueError):
    alpha: float
    obj: str
    logger: Logger
    message: str = field(
        init = False
    )
    def __post_init__(self) -> None:
        self.message = f"The alpha value {self.alpha} is not valid for the object {self.obj} or produced no simplices."
        self.logger.error(self.message)
        super().__init__(self.message)

@dataclass
class NotFittedError(AttributeError):
    obj: str
    logger: Logger
    message: str = field(
        init = False
    )
    def __post_init__(self) -> None:
        self.message = f"The object {self.obj} is not fitted."
        self.logger.error(self.message)
        super().__init__(self.message)


@dataclass
class UnsupportedFileType(ValueError):
    file_type: str
    logger: Logger
    message: str = field(
        init = False
    )
    def __post_init__(self, file_type):
        self.message = f"Unsupported data file type: {file_type}"
        super().__init__(self.message)