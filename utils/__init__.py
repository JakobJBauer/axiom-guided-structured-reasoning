from .datapoint_serializer import (
    CodebookDatapointSerializer,
    SerializedCodebookDatapoint,
)

__all__ = [
    "CodebookDatapointInferencer",
    "CodebookDatapointSerializer",
    "SerializedCodebookDatapoint",
    "load_model_and_processor",
]


def __getattr__(name: str):
    if name == "CodebookDatapointInferencer":
        from .datapoint_inference import CodebookDatapointInferencer

        return CodebookDatapointInferencer
    if name == "load_model_and_processor":
        from .model_loader import load_model_and_processor

        return load_model_and_processor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
