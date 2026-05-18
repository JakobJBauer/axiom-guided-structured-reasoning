from .datapoint_serializer import (
    CodebookDatapointSerializer,
    SerializedCodebookDatapoint,
)

__all__ = [
    "CodebookDatapointSerializer",
    "SerializedCodebookDatapoint",
    "load_model_and_processor",
]


def load_model_and_processor(*args, **kwargs):
    from .model_loader import load_model_and_processor as _load

    return _load(*args, **kwargs)
