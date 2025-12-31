from dataclasses import dataclass, field
from typing import Optional
from trl import ScriptArguments

from ..gold.gold_config import GOLDConfig


@dataclass
class GOLDMultimodalScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: [],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )


@dataclass
class GOLDMultimodalConfig(GOLDConfig):
    r"""
    Configuration class for [`GOLDMultimodalTrainer`].
    """

    # "vqa" or "vqa_thinking"
    dataset_type: str = "vqa"
    dataset_from_disk: bool = True

    # With alpha = 1.0, the model will perform only distillation.
    # With alpha = 0.0, the model will perform GPRO only.
    alpha: float = 1.0
    beta_rl: float = 1.0

    # Number of generations to sample for GPRO.
    num_generations: int = 6