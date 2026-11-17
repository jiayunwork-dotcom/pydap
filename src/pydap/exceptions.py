"""DAP exceptions.

These exceptions are mostly used by the server. When an exception is captured,
proper error message is displayed (according to the DAP 2.0 spec), with
information about the exception.

"""


class DapError(Exception):
    """Base DAP exception."""

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class ClientError(DapError):
    """Generic error with the client."""

    pass


class ServerError(DapError):
    """Generic error with the server."""

    pass


class ConstraintExpressionError(ServerError):
    """Exception raised when an invalid constraint expression is given."""

    pass


class HandlerError(DapError):
    """Generic error with a handler."""

    pass


class ExtensionNotSupportedError(HandlerError):
    """Exception raised when opening a file not supported by any handlers."""

    pass


class OpenFileError(HandlerError):
    """Exception raised when unable to open a file."""

    pass


class DAP4ChecksumError(ServerError):
    """Exception raised when a DAP4 variable checksum does not match the
    checksum computed from the received binary payload."""

    pass


class DAP4TruncatedResponseError(ServerError):
    """Exception raised when a DAP4 binary response ends before all the
    variables described by its DMR have been decoded."""

    pass


class BufferLimitExceededError(ClientError):
    """Exception raised when a single required observation slice is larger
    than the configured dynamic data buffer budget."""

    pass
