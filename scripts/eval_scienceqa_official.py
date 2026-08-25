#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ScienceQA evaluation for LLaVA-v1.6-Vicuna-7B.

Supports:
    1. Base model
    2. Unmerged LoRA adapter + Base model
    3. Merged fine-tuned model

Inference:
    - LLaVA-v1.6 AnyRes
    - same image preprocessing
    - same conversation template

Evaluation:
    - follows the official ScienceQA answer parsing logic
    - supports:
        A
        A. xxx
        The answer is A.
    - exact option-index accuracy
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model


# ============================================================
# Default paths
# ============================================================

DEFAULT_BASE = "/root/autodl-tmp/models/llava-v1.6-vicuna-7b"

DEFAULT_LORA = (
    "/root/autodl-tmp/outputs/llava_scienceqa_lora"
)

DEFAULT_MERGED = (
    "/root/autodl-tmp/models/llava-v1.6-scienceqa-merged"
)

DEFAULT_TEST_JSON = (
    "/root/autodl-tmp/LLAVA/playground/data/ScienceQA/"
    "scienceqa_test_llava.json"
)

DEFAULT_IMAGE_DIR = (
    "/root/autodl-tmp/LLAVA/playground/data/ScienceQA/images"
)

OPTIONS = ["A", "B", "C", "D", "E"]


# ============================================================
# Model loading
# ============================================================

def load_llava_model(model_path, model_base=None):

    model_name = get_model_name_from_path(model_path)

    print("\n" + "=" * 80)
    print("Loading LLaVA model")
    print("=" * 80)
    print("Model path :", model_path)
    print("Model base :", model_base)
    print("Model name :", model_name)
    print("=" * 80)

    tokenizer, model, image_processor, context_len = (
        load_pretrained_model(
            model_path=model_path,
            model_base=model_base,
            model_name=model_name,
            load_8bit=False,
            load_4bit=False,
            device="cuda",
        )
    )

    model.eval()

    print("\nModel loaded successfully.")
    print("Model class :", type(model).__name__)
    print("Model dtype :", model.dtype)
    print("Model device:", model.device)
    print("Context len :", context_len)

    config = model.config

    print("\nVision config:")
    print(
        "  model_type               =",
        getattr(config, "model_type", None),
    )
    print(
        "  mm_projector_type        =",
        getattr(config, "mm_projector_type", None),
    )
    print(
        "  image_aspect_ratio       =",
        getattr(config, "image_aspect_ratio", None),
    )
    print(
        "  mm_vision_select_layer   =",
        getattr(config, "mm_vision_select_layer", None),
    )
    print(
        "  mm_vision_select_feature =",
        getattr(config, "mm_vision_select_feature", None),
    )
    print(
        "  mm_patch_merge_type      =",
        getattr(config, "mm_patch_merge_type", None),
    )

    projector = model.get_model().mm_projector

    print("\nProjector:")
    print(projector)

    vision_tower = model.get_vision_tower()

    vision_loaded = (
        vision_tower is not None
        and vision_tower.is_loaded
    )

    print(
        "\nVision tower loaded:",
        vision_loaded,
    )

    if vision_tower is None:
        raise RuntimeError(
            "Vision tower is None."
        )

    if image_processor is None:

        print(
            "image_processor is None; "
            "using vision tower image_processor instead."
        )

        if getattr(
            vision_tower,
            "image_processor",
            None,
        ) is None:
            raise RuntimeError(
                "Vision tower image_processor "
                "is also None."
            )

        image_processor = vision_tower.image_processor

    print(
        "Final image_processor:",
        type(image_processor).__name__,
    )

    return (
        tokenizer,
        model,
        image_processor,
        context_len,
    )


# ============================================================
# Image preprocessing
# ============================================================

def prepare_image(
    image,
    image_processor,
    model,
):

    image_tensor = process_images(
        [image],
        image_processor,
        model.config,
    )

    if isinstance(image_tensor, list):

        image_tensor = [
            tensor.to(
                device=model.device,
                dtype=model.dtype,
            )
            for tensor in image_tensor
        ]

        return image_tensor

    image_tensor = image_tensor.to(
        device=model.device,
        dtype=model.dtype,
    )

    return image_tensor


def inspect_anyres_output(
    image_tensor,
    model,
):

    print("\nImage tensor check:")

    if isinstance(image_tensor, list):

        print("  type = list")

        for i, tensor in enumerate(image_tensor):

            print(
                f"  item[{i}] shape = {tuple(tensor.shape)}"
            )

        return

    print(
        "  type  =",
        type(image_tensor).__name__,
    )

    print(
        "  shape =",
        tuple(image_tensor.shape),
    )

    image_aspect_ratio = getattr(
        model.config,
        "image_aspect_ratio",
        None,
    )

    print(
        "  aspect_ratio =",
        image_aspect_ratio,
    )

    if image_aspect_ratio == "anyres":

        if image_tensor.ndim == 5:

            num_patches = image_tensor.shape[1]

            print(
                "  AnyRes multi-patch = True"
            )

            print(
                "  num_patches =",
                num_patches,
            )

        else:

            print(
                "  WARNING: expected 5D AnyRes tensor, "
                f"got {image_tensor.ndim}D."
            )


# ============================================================
# Official ScienceQA answer parsing
# ============================================================

def parse_scienceqa_answer(prediction):
    """
    Official ScienceQA-style answer parsing.

    Supports:
        A
        A. xxx
        The answer is A.
    """

    if prediction is None:
        return "FAILED"

    pred_text = str(prediction).strip()

    # Case 1:
    # prediction == "A"
    if pred_text in OPTIONS:
        return pred_text

    # Case 2:
    # prediction == "A. xxx"
    if (
        len(pred_text) >= 3
        and pred_text[0] in OPTIONS
        and pred_text[1:3] == ". "
    ):
        return pred_text[0]

    # Case 3:
    # "The answer is A."
    pattern = re.compile(
        r"The answer is ([A-Z])."
    )

    matches = pattern.findall(pred_text)

    if len(matches) == 1:
        return matches[0]

    return "FAILED"


def get_pred_idx(
    prediction,
    choices,
):
    """
    Same logic as official ScienceQA evaluator.

    Convert:
        A/B/C/D/E
    to:
        0/1/2/3/4
    """

    if prediction in OPTIONS[:len(choices)]:
        return OPTIONS.index(prediction)

    return -1


# ============================================================
# Single sample inference
# ============================================================

def generate_answer(
    item,
    image_dir,
    tokenizer,
    model,
    image_processor,
    debug=False,
):

    image_name = item["image"]

    question_prompt = (
        item["conversations"][0]["value"]
    )

    image_path = os.path.join(
        image_dir,
        image_name,
    )

    if not os.path.isfile(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = Image.open(
        image_path
    ).convert("RGB")

    original_image_size = image.size

    image_tensor = prepare_image(
        image,
        image_processor,
        model,
    )

    if debug:
        inspect_anyres_output(
            image_tensor,
            model,
        )

    # IMPORTANT:
    # Use exactly the converted ScienceQA prompt.
    # Do not add another instruction.
    user_prompt = question_prompt

    conv = conv_templates[
        "llava_v1"
    ].copy()

    conv.append_message(
        conv.roles[0],
        user_prompt,
    )

    conv.append_message(
        conv.roles[1],
        None,
    )

    full_prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        full_prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(model.device)

    with torch.inference_mode():

        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[
                original_image_size
            ],
            do_sample=False,
            max_new_tokens=16,
            use_cache=True,
        )

    prediction = tokenizer.batch_decode(
        output_ids,
        skip_special_tokens=True,
    )[0].strip()

    return prediction


# ============================================================
# Dataset validation
# ============================================================

def validate_dataset(
    test_json,
    image_dir,
):

    if not os.path.isfile(test_json):
        raise FileNotFoundError(
            f"Test JSON not found:\n{test_json}"
        )

    if not os.path.isdir(image_dir):
        raise FileNotFoundError(
            f"Image directory not found:\n{image_dir}"
        )

    with open(
        test_json,
        "r",
        encoding="utf-8",
    ) as f:

        samples = json.load(f)

    if not isinstance(samples, list):
        raise TypeError(
            "ScienceQA JSON must be a JSON array."
        )

    print("\n" + "=" * 80)
    print("Dataset check")
    print("=" * 80)

    print(
        "Test samples:",
        len(samples),
    )

    print(
        "Image directory:",
        image_dir,
    )

    missing = []
    invalid = []

    for item in samples:

        if "image" not in item:
            invalid.append(
                f"Missing image: {item.get('id')}"
            )
            continue

        image_path = os.path.join(
            image_dir,
            item["image"],
        )

        if not os.path.isfile(image_path):
            missing.append(image_path)

        if "conversations" not in item:
            invalid.append(
                f"Missing conversations: {item.get('id')}"
            )
            continue

        if len(item["conversations"]) < 2:
            invalid.append(
                f"Invalid conversations: {item.get('id')}"
            )

    print(
        "Missing images:",
        len(missing),
    )

    print(
        "Invalid samples:",
        len(invalid),
    )

    if missing:

        print("\nFirst 20 missing images:")

        for path in missing[:20]:
            print(path)

        raise RuntimeError(
            f"Found {len(missing)} missing images."
        )

    if invalid:

        print("\nFirst 20 invalid samples:")

        for item in invalid[:20]:
            print(item)

        raise RuntimeError(
            f"Found {len(invalid)} invalid samples."
        )

    print("Dataset check passed.")


# ============================================================
# Main evaluation
# ============================================================

def evaluate(args):

    validate_dataset(
        args.test_json,
        args.image_dir,
    )

    with open(
        args.test_json,
        "r",
        encoding="utf-8",
    ) as f:

        samples = json.load(f)

    original_dataset_size = len(samples)

    if args.max_samples is not None:

        samples = samples[
            :args.max_samples
        ]

    print(
        "\nSamples to evaluate:",
        len(samples),
    )

    (
        tokenizer,
        model,
        image_processor,
        context_len,
    ) = load_llava_model(
        model_path=args.model_path,
        model_base=args.model_base,
    )

    print("\n" + "=" * 80)
    print("Evaluation configuration")
    print("=" * 80)

    print(
        "Model label:",
        args.model_label,
    )

    print(
        "Image aspect ratio:",
        getattr(
            model.config,
            "image_aspect_ratio",
            None,
        ),
    )

    print(
        "Patch merge type:",
        getattr(
            model.config,
            "mm_patch_merge_type",
            None,
        ),
    )

    print(
        "Answer parsing:",
        "Official ScienceQA logic",
    )

    print(
        "do_sample:",
        False,
    )

    print(
        "max_new_tokens:",
        16,
    )

    print(
        "use_cache:",
        True,
    )

    print("=" * 80)

    predictions = []

    correct = 0
    failed_samples = 0

    start_time = time.time()

    progress = tqdm(
        enumerate(samples),
        total=len(samples),
        desc=f"Evaluating {args.model_label}",
    )

    for idx, item in progress:

        sample_id = item.get("id")

        question = ""
        ground_truth = ""
        prediction = ""

        try:

            question = (
                item["conversations"][0]["value"]
            )

            ground_truth = (
                item["conversations"][1]["value"]
            )

            prediction = generate_answer(
                item=item,
                image_dir=args.image_dir,
                tokenizer=tokenizer,
                model=model,
                image_processor=image_processor,
                debug=(
                    args.debug_anyres
                    and idx == 0
                ),
            )

            # --------------------------------------------
            # Official ScienceQA parsing
            # --------------------------------------------

            answer = parse_scienceqa_answer(
                prediction
            )

            # Our converted GT is:
            # "A. xxx"
            gt_answer = parse_scienceqa_answer(
                ground_truth
            )

            # Get option indices
            pred_idx = get_pred_idx(
                answer,
                item["conversations"][0]["value"]
                .split("\n"),
            )

            # Better GT/pred mapping:
            # Directly compare the parsed letters.
            is_correct = (
                answer != "FAILED"
                and gt_answer != "FAILED"
                and answer == gt_answer
            )

            score = 1 if is_correct else 0

            if is_correct:
                correct += 1

        except Exception as exc:

            failed_samples += 1

            answer = "FAILED"
            gt_answer = "FAILED"
            score = 0

            error = repr(exc)

            print(
                "\n" + "=" * 80
            )

            print("ERROR")
            print("=" * 80)

            print(
                "Index:",
                idx,
            )

            print(
                "Sample ID:",
                sample_id,
            )

            print(
                "Error:",
                error,
            )

            print("=" * 80)

        else:
            error = None

        result = {
            "id": sample_id,
            "image": item.get("image"),
            "question": question,
            "ground_truth": ground_truth,
            "ground_truth_option": gt_answer,
            "prediction": prediction,
            "prediction_option": answer,
            "correct": bool(score),
            "error": error,
        }

        predictions.append(result)

        if idx < args.print_samples:

            print(
                "\n" + "=" * 80
            )

            print(
                f"Sample {idx + 1}/{len(samples)}"
            )

            print("=" * 80)

            print(
                "Ground Truth:",
                ground_truth,
            )

            print(
                "GT Option:",
                gt_answer,
            )

            print(
                "Prediction:",
                prediction,
            )

            print(
                "Pred Option:",
                answer,
            )

            print(
                "Correct:",
                bool(score),
            )

        current_accuracy = (
            correct / (idx + 1)
        )

        progress.set_postfix(
            acc=f"{current_accuracy * 100:.2f}%"
        )

    # ========================================================
    # Final statistics
    # ========================================================

    elapsed = time.time() - start_time

    accuracy = (
        correct / len(samples)
        if samples
        else 0.0
    )

    seconds_per_sample = (
        elapsed / len(samples)
        if samples
        else 0.0
    )

    samples_per_second = (
        len(samples) / elapsed
        if elapsed > 0
        else 0.0
    )

    # ========================================================
    # Save results
    # ========================================================

    output_path = Path(
        args.output_json
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_data = {

        "model_label":
            args.model_label,

        "model_path":
            args.model_path,

        "model_base":
            args.model_base,

        "test_json":
            args.test_json,

        "image_dir":
            args.image_dir,

        "original_dataset_size":
            original_dataset_size,

        "evaluated_samples":
            len(samples),

        "correct_samples":
            correct,

        "failed_samples":
            failed_samples,

        "accuracy":
            accuracy,

        "accuracy_percent":
            accuracy * 100.0,

        "elapsed_seconds":
            elapsed,

        "seconds_per_sample":
            seconds_per_sample,

        "samples_per_second":
            samples_per_second,

        "model_config":
        {
            "image_aspect_ratio":
                getattr(
                    model.config,
                    "image_aspect_ratio",
                    None,
                ),

            "mm_patch_merge_type":
                getattr(
                    model.config,
                    "mm_patch_merge_type",
                    None,
                ),

            "mm_vision_select_layer":
                getattr(
                    model.config,
                    "mm_vision_select_layer",
                    None,
                ),

            "mm_vision_select_feature":
                getattr(
                    model.config,
                    "mm_vision_select_feature",
                    None,
                ),
        },

        "generation_config":
        {
            "do_sample": False,
            "max_new_tokens": 16,
            "use_cache": True,
            "conversation_template": "llava_v1",
        },

        "evaluation_protocol":
            "Official ScienceQA answer parsing",

        "predictions":
            predictions,
    }

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output_data,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ========================================================
    # Final result
    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print("FINAL RESULT")

    print("=" * 80)

    print(
        "Model label:",
        args.model_label,
    )

    print(
        "Model path :",
        args.model_path,
    )

    print(
        "Model base :",
        args.model_base,
    )

    print(
        "Evaluated samples:",
        len(samples),
    )

    print(
        "Correct samples:",
        correct,
    )

    print(
        "Failed samples:",
        failed_samples,
    )

    print(
        f"Accuracy: "
        f"{accuracy * 100:.4f}%"
    )

    print(
        f"Elapsed time: "
        f"{elapsed / 60:.2f} minutes"
    )

    print(
        f"Time per sample: "
        f"{seconds_per_sample:.3f} seconds"
    )

    print(
        f"Throughput: "
        f"{samples_per_second:.3f} samples/sec"
    )

    print(
        "Results saved to:",
        output_path,
    )

    print("=" * 80)


# ============================================================
# Command line
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate LLaVA-v1.6-Vicuna-7B "
            "on ScienceQA test set."
        )
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--model-base",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--model-label",
        type=str,
        default="model",
    )

    parser.add_argument(
        "--test-json",
        type=str,
        default=DEFAULT_TEST_JSON,
    )

    parser.add_argument(
        "--image-dir",
        type=str,
        default=DEFAULT_IMAGE_DIR,
    )

    parser.add_argument(
        "--output-json",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--print-samples",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--debug-anyres",
        action="store_true",
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is not available."
        )

    print(
        "CUDA available:",
        torch.cuda.is_available(),
    )

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    print(
        "PyTorch:",
        torch.__version__,
    )

    print(
        "CUDA runtime:",
        torch.version.cuda,
    )

    evaluate(args)


if __name__ == "__main__":
    main()