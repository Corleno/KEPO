# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, load_from_disk
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    AutoModelForCausalLM,
)

from .trainer import Qwen2VLOPDTrainer
from trl import GRPOConfig, ModelConfig, ScriptArguments, TrlParser, get_peft_config


@dataclass
class OPDScriptArguments(ScriptArguments):
    """
    Script arguments for the OPD (On-Policy Distillation) training script.

    Args:
        teacher_model_name_or_path (`str`):
            Path to the teacher model for OPD training.
    """

    teacher_model_name_or_path: str = field(
        default=None,
        metadata={"help": "Path to the teacher model for OPD training"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )


# OPD doesn't need reward functions - removed


def main(script_args, training_args, model_args):
    # Load the dataset
    dataset = load_from_disk(script_args.dataset_name)
    
    # Load teacher model for OPD
    teacher_model = None
    if script_args.teacher_model_name_or_path is not None:
        teacher_model_id = script_args.teacher_model_name_or_path
        teacher_model_kwargs = dict(
            trust_remote_code=model_args.trust_remote_code,
            attn_implementation=model_args.attn_implementation,
            torch_dtype=model_args.dtype,
        )
        
        # Load teacher model based on model type
        if "Qwen2-VL" in teacher_model_id:
            teacher_model = Qwen2VLForConditionalGeneration.from_pretrained(
                teacher_model_id, **teacher_model_kwargs
            )
        elif "Qwen2.5-VL" in teacher_model_id:
            teacher_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                teacher_model_id, **teacher_model_kwargs
            )
        elif "Qwen3-VL" in teacher_model_id:
            teacher_model = Qwen3VLForConditionalGeneration.from_pretrained(
                teacher_model_id, **teacher_model_kwargs
            )
        else:
            teacher_model = AutoModelForCausalLM.from_pretrained(
                teacher_model_id, **teacher_model_kwargs
            )
        print(f"Loaded teacher model from: {teacher_model_id}")


    # Format into conversation
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

    # def make_conversation_image(example):
    #     return {
    #         "prompt": [
    #             {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
    #             {
    #                 "role": "user",
    #                 "content": [
    #                     {"type": "image"},
    #                     {"type": "text", "text": example["problem"]},
    #                 ],
    #             },
    #         ],
    #     }

    QUESTION_TEMPLATE = "{Question} Your task: 1. Provide the correct single-letter choice (A, B, C, D,...) inside <answer>...</answer> tags. 2. No extra information or text outside of this tag."

    def make_conversation_image(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                    ],
                },
            ],
        }


    if "image" in dataset[script_args.dataset_train_split].features:
        print("has image in dataset")
        dataset = dataset.map(make_conversation_image)  # Utilize multiprocessing for faster mapping
        # dataset = dataset.remove_columns(["original_question", "original_answer"])

    else:
        print("no image in dataset")
        dataset = dataset.map(make_conversation)
        dataset = dataset.remove_columns("messages")

    
    # Initialize the OPD trainer
    trainer = Qwen2VLOPDTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=[],  # OPD doesn't need reward functions
        teacher_model=teacher_model,  # Teacher from ref_model entrance (engineering trick)
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((OPDScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
