class PipelineError(Exception):
    """An error whose message is safe and useful to show to the end user."""


class AudioError(PipelineError):
    """The uploaded audio is missing, empty, unsupported, corrupt or silent."""
