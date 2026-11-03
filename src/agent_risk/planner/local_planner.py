"""Local Planner: Local Llama / LoRA inference engine for structured DAG generation."""

import json
import logging
from pathlib import Path
from typing import Any, Optional, Union

from agent_risk.config import settings
from agent_risk.planner.api_planner import BasePlanner, PLANNER_SYSTEM_PROMPT
from agent_risk.planner.schema import PlanDAG

logger = logging.getLogger(__name__)


class LocalPlanner(BasePlanner):
    """Local inference engine using HuggingFace / PEFT LoRA checkpoints."""

    def __init__(
        self,
        base_model_name_or_path: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        adapter_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        load_in_4bit: bool = False,
    ):
        self.base_model_name = base_model_name_or_path
        self.adapter_path = Path(adapter_path or settings.local_model_path)
        self.device = device or settings.device
        self.load_in_4bit = load_in_4bit
        self._model = None
        self._tokenizer = None

    def load_model(self) -> None:
        """Lazy loader for tokenizer and model weights with LoRA adapter."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel

            logger.info(f"Loading base tokenizer from {self.base_model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)

            logger.info(f"Loading base model: {self.base_model_name}")
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )

            if self.adapter_path.exists():
                logger.info(f"Applying LoRA adapter from {self.adapter_path}")
                self._model = PeftModel.from_pretrained(model, str(self.adapter_path))
            else:
                logger.warning(
                    f"Adapter path '{self.adapter_path}' does not exist. Using base model."
                )
                self._model = model

            self._model.eval()

        except ImportError as e:
            logger.error(
                f"Missing required ML libraries for LocalPlanner: {e}. "
                "Ensure torch, transformers, and peft are installed."
            )
            raise

    def generate_plan(self, prompt: str, **kwargs: Any) -> PlanDAG:
        """Generate structured plan using local model."""
        if self._model is None or self._tokenizer is None:
            self.load_model()

        messages = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        prompt_str = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self._tokenizer(prompt_str, return_tensors="pt").to(self._model.device)
        max_new_tokens = kwargs.get("max_new_tokens", 1024)
        temperature = kwargs.get("temperature", 0.1)

        outputs = self._model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=temperature if temperature > 0.0 else None,
        )

        generated_text = self._tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )

        # Attempt to parse json
        try:
            data = json.loads(generated_text)
            return PlanDAG.model_validate(data)
        except Exception as e:
            logger.error(f"Failed to parse model output into PlanDAG schema: {e}")
            raise

