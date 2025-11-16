import logging
import os
from typing import Dict, Final

import torch
from transformers import AutoModel, AutoTokenizer

from deepseek_ocr.model_config import ModelConfig
from deepseek_ocr.model_type import ModelType

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class DeepSeekOCRModel:
    """
    A professional class for loading and running inference with the DeepSeek-OCR model.
    The model configuration (`model_type`) is specified at the time of inference.
    """

    MODEL_NAME: Final[str] = "deepseek-ai/DeepSeek-OCR"

    CONFIGS: Final[Dict[ModelType, ModelConfig]] = {
        "tiny": {"base_size": 512, "image_size": 512, "crop_mode": False},
        "small": {"base_size": 640, "image_size": 640, "crop_mode": False},
        "base": {"base_size": 1024, "image_size": 1024, "crop_mode": False},
        "large": {"base_size": 1280, "image_size": 1280, "crop_mode": False},
        "gundam": {"base_size": 1024, "image_size": 640, "crop_mode": True},
    }

    def __init__(self) -> None:
        """
        Initializes the model and tokenizer, moving the model to the GPU.
        The specific configuration for inference is handled by run_inference.

        Raises:
            RuntimeError: If CUDA is not available.
        """
        if not torch.cuda.is_available():
            self.logger.critical("CUDA device not found. This model requires a GPU.")
            raise RuntimeError("CUDA device not found. This model requires a GPU.")

        self._init_logger()
        self.logger.info(f"Loading DeepSeek-OCR model: {self.MODEL_NAME}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.MODEL_NAME, trust_remote_code=True
            )

            self.model = AutoModel.from_pretrained(
                self.MODEL_NAME,
                _attn_implementation="flash_attention_2",
                trust_remote_code=True,
                use_safetensors=True,
            )
        except Exception as e:
            self.logger.error(f"Failed to load model or tokenizer: {e}", exc_info=True)
            raise e

        # Move to GPU, set to evaluation mode, and use bfloat16
        self.model = self.model.eval().cuda().to(torch.bfloat16)

        self.logger.info("Model base components initialized successfully on GPU.")

    def _init_logger(self) -> None:
        """
        Initializes the logger.
        """
        # Setup Logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            ch = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

    def run_inference(
        self,
        prompt: str,
        image_path: str,
        model_type: ModelType = "gundam",
        output_dir_path: str = "output_dir",
    ) -> None:
        """
        Runs the OCR inference using the loaded model with a specified configuration.

        Args:
            prompt: The text prompt to prepend with '<image>\\n'.
            image_path: Path to the input image file.
            model_type: The configuration preset to use for this specific inference call.
            output_dir_path: Directory where results will be saved.

        Raises:
            ValueError: If an invalid model_type is provided.
        """
        if model_type not in self.CONFIGS:
            self.logger.error(
                f"Invalid model_type specified: {model_type}. Must be one of {list(self.CONFIGS.keys())}"
            )
            raise ValueError(f"Invalid model_type: {model_type}")

        # Get Configuration at Inference Time
        config = self.CONFIGS[model_type]
        self.logger.info(f"Starting inference with configuration: '{model_type}'")
        self.logger.debug(
            f"Config parameters: Base Size={config['base_size']}, Image Size={config['image_size']}"
        )

        full_prompt = "<image>\n" + prompt

        try:
            self.model.infer(
                self.tokenizer,
                prompt=full_prompt,
                image_file=image_path,
                output_path=output_dir_path,
                base_size=config["base_size"],
                image_size=config["image_size"],
                crop_mode=config["crop_mode"],
                save_results=True,
                test_compress=True,
            )
            self.logger.info(
                f"Inference complete for '{model_type}'. Results saved to: {output_dir_path}"
            )

        except Exception as e:
            self.logger.error(
                f"Inference failed for image {image_path} using type '{model_type}': {e}",
                exc_info=True,
            )
            raise e
