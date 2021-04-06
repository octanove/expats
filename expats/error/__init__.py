
class InternalProfilerError(Exception):
    """
    Internally defined base error in this project
    """


class InstantiationError(InternalProfilerError):
    """
    Error raised when creating instance
    """


class ArtifactNotFoundError(InternalProfilerError):
    # TODO: docstring
    pass


class DeserializationError(InternalProfilerError):
    # TODO: docstring
    pass


class SerializationError(InternalProfilerError):
    # TODO: docstring
    pass
