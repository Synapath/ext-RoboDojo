class EmptyAnnotatorDataError(RuntimeError):
    """Raised while a newly attached Replicator annotator is still warming up."""


def get_data_with_warmup_retry(camera, annotator_name, out, render, max_attempts=5):
    """Retry an empty newly attached annotator after advancing the renderer."""
    for attempt in range(max_attempts):
        try:
            return camera.get_data(annotator_name, out=out)
        except EmptyAnnotatorDataError:
            if attempt == max_attempts - 1:
                raise
            render()
    raise AssertionError("unreachable")
