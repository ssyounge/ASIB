# utils/exceptions.py

"""Custom exceptions for ASIB framework."""


class ASIBError(Exception):
    """Base exception for ASIB framework."""
    pass


class ConfigurationError(ASIBError):
    """Raised when there's a configuration error."""
    pass


class ModelRegistryError(ASIBError):
    """Raised when there's an error with model registry."""
    pass


class TrainingError(ASIBError):
    """Raised when there's an error during training."""
    pass


class DataError(ASIBError):
    """Raised when there's an error with data loading."""
    pass


class ValidationError(ASIBError):
    """Raised when validation fails."""
    pass


class CheckpointError(ASIBError):
    """Raised when there's an error with checkpoints."""
    pass


class DeviceError(ASIBError):
    """Raised when there's an error with device setup."""
    pass 