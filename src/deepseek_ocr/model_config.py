from typing import TypedDict


class ModelConfig(TypedDict):
    """Type definition for a model's configuration dictionary."""

    base_size: int
    image_size: int
    crop_mode: bool
