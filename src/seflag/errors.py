class SeflagError(Exception):
    """Base class for all errors raised by SEFLAG."""


class ConfigError(SeflagError):
    """The evaluation configuration is invalid."""


class MissingDependencyError(SeflagError):
    """An optional package or model required by the configuration is not installed."""


class DataError(SeflagError):
    """A dataset could not be read or does not contain what the task needs."""
