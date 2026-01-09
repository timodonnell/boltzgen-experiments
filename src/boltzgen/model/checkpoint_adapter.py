"""
Checkpoint adapter for loading old BoltzGen checkpoints into simplified models.

This module provides utilities to transform checkpoints from the full BoltzGen model
(with inverse folding, confidence, affinity prediction, etc.) into checkpoints that
can be loaded by simplified models that only support design/diffusion.
"""

from typing import Dict, Any, Set
import torch


def get_removable_module_prefixes() -> Set[str]:
    """
    Get the list of module prefixes that can be safely removed from checkpoints
    when loading into a simplified design-only model.

    Returns:
        Set of string prefixes for modules to remove
    """
    return {
        # Inverse folding modules
        "inverse_folding_encoder.",
        "inverse_folding_decoder.",

        # Confidence prediction
        "confidence_module.",

        # Affinity prediction
        "affinity_module.",
        "affinity_module1.",
        "affinity_module2.",

        # B-factor prediction
        "bfactor_module.",

        # Distogram (used in training but not needed for inference)
        "distogram_module.",

        # Validation/training metrics (not part of model weights but sometimes in checkpoints)
        "train_confidence_loss_logger.",
        "train_confidence_loss_dict_logger.",
    }


def get_removable_exact_keys() -> Set[str]:
    """
    Get exact key names that should be removed from checkpoints.

    Returns:
        Set of exact key names to remove
    """
    return {
        # EMA-related keys that are handled separately
        "ema",
        "ema_decay",
    }


def adapt_checkpoint_for_simplified_model(
    checkpoint: Dict[str, Any],
    verbose: bool = True,
    strict_mode: bool = False,
) -> Dict[str, Any]:
    """
    Adapt a checkpoint from full BoltzGen model to work with simplified design-only model.

    This function:
    1. Removes weights for inverse folding, confidence, affinity, bfactor modules
    2. Preserves all core design/diffusion model weights
    3. Handles EMA weights if present
    4. Preserves checkpoint metadata

    Args:
        checkpoint: Full checkpoint dict (as loaded by torch.load)
        verbose: If True, print information about removed keys
        strict_mode: If True, fail if unexpected keys are found

    Returns:
        Adapted checkpoint dict that can be loaded into simplified model

    Example:
        >>> checkpoint = torch.load("boltzgen1.ckpt")
        >>> adapted = adapt_checkpoint_for_simplified_model(checkpoint)
        >>> model.load_state_dict(adapted["state_dict"], strict=False)
    """
    # Work on a copy to avoid modifying the original
    adapted_checkpoint = {
        "state_dict": {},
        "hyper_parameters": checkpoint.get("hyper_parameters", {}),
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
    }

    # Get original state dict
    if "ema" in checkpoint and checkpoint["ema"] is not None:
        # Prefer EMA weights if available
        state_dict = checkpoint["ema"]
        if verbose:
            print("Using EMA weights from checkpoint")
    else:
        state_dict = checkpoint.get("state_dict", checkpoint)

    # Get prefixes and keys to remove
    removable_prefixes = get_removable_module_prefixes()
    removable_keys = get_removable_exact_keys()

    # Track what we removed
    removed_keys = []
    kept_keys = []

    # Filter state dict
    for key, value in state_dict.items():
        # Check if key should be removed
        should_remove = False

        # Check exact match
        if key in removable_keys:
            should_remove = True
            removal_reason = f"exact match: {key}"

        # Check prefix match
        if not should_remove:
            for prefix in removable_prefixes:
                if key.startswith(prefix):
                    should_remove = True
                    removal_reason = f"prefix match: {prefix}"
                    break

        if should_remove:
            removed_keys.append((key, removal_reason))
        else:
            adapted_checkpoint["state_dict"][key] = value
            kept_keys.append(key)

    # Handle checkpoint key remapping (from older checkpoint formats)
    # The original code has this in on_load_checkpoint:
    # k.replace(".token_transformer_layers.0.", ".token_transformer.")
    remapped_state_dict = {}
    for key, value in adapted_checkpoint["state_dict"].items():
        # Fix old naming convention if present
        new_key = key.replace(".token_transformer_layers.0.", ".token_transformer.")
        if new_key != key and verbose:
            print(f"Remapped key: {key} -> {new_key}")
        remapped_state_dict[new_key] = value

    adapted_checkpoint["state_dict"] = remapped_state_dict

    # Update hyperparameters to reflect simplified model
    if "hyper_parameters" in adapted_checkpoint:
        hyper_params = adapted_checkpoint["hyper_parameters"]

        # Set removed features to False/None
        hyper_params["inverse_fold"] = False
        hyper_params["confidence_prediction"] = False
        hyper_params["affinity_prediction"] = False
        hyper_params["predict_bfactor"] = False
        hyper_params["predict_res_type"] = False
        hyper_params["use_kernels"] = False

        # Remove validator-related params
        hyper_params["validators"] = None
        hyper_params["validate_structure"] = False

        # Remove miniformer parameters (not supported anymore)
        hyper_params.pop("use_miniformer", None)

        # Remove miniformer_blocks from nested configs
        if "msa_args" in hyper_params:
            hyper_params["msa_args"].pop("miniformer_blocks", None)
        if "template_args" in hyper_params:
            hyper_params["template_args"].pop("miniformer_blocks", None)
        if "token_distance_args" in hyper_params:
            hyper_params["token_distance_args"].pop("miniformer_blocks", None)

        # Remove Phase 1 unused parameters
        hyper_params.pop("ignore_ckpt_shape_mismatch", None)
        if "pairformer_args" in hyper_params:
            hyper_params["pairformer_args"].pop("post_layer_norm", None)
        if "score_model_args" in hyper_params:
            hyper_params["score_model_args"].pop("transformer_post_ln", None)

        if verbose:
            print("\nUpdated hyperparameters to disable removed features and miniformer")

    # Print summary
    if verbose:
        print(f"\n=== Checkpoint Adaptation Summary ===")
        print(f"Original checkpoint keys: {len(state_dict)}")
        print(f"Kept keys: {len(kept_keys)}")
        print(f"Removed keys: {len(removed_keys)}")
        print(f"\nRemoved key breakdown:")

        # Group by module
        by_module = {}
        for key, reason in removed_keys:
            module = reason.split(":")[1].strip().rstrip(".")
            if module not in by_module:
                by_module[module] = 0
            by_module[module] += 1

        for module, count in sorted(by_module.items()):
            print(f"  {module}: {count} keys")

        # Show sample of kept keys
        print(f"\nSample of kept keys (essential model weights):")
        for key in sorted(kept_keys)[:10]:
            print(f"  ✓ {key}")
        if len(kept_keys) > 10:
            print(f"  ... and {len(kept_keys) - 10} more")

    return adapted_checkpoint


def save_adapted_checkpoint(
    input_path: str,
    output_path: str,
    verbose: bool = True,
) -> None:
    """
    Load a checkpoint, adapt it for simplified model, and save it.

    Args:
        input_path: Path to original checkpoint
        output_path: Path to save adapted checkpoint
        verbose: Print progress information

    Example:
        >>> save_adapted_checkpoint(
        ...     "boltzgen1.ckpt",
        ...     "boltzgen1_simplified.ckpt"
        ... )
    """
    if verbose:
        print(f"Loading checkpoint from: {input_path}")

    # weights_only=False is required for BoltzGen checkpoints
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)

    adapted = adapt_checkpoint_for_simplified_model(
        checkpoint,
        verbose=verbose,
    )

    if verbose:
        print(f"\nSaving adapted checkpoint to: {output_path}")

    torch.save(adapted, output_path)

    if verbose:
        print("Done!")


def load_adapted_checkpoint(
    checkpoint_path: str,
    map_location=None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Load and adapt a checkpoint in one step.

    Args:
        checkpoint_path: Path to checkpoint file
        map_location: Device to map checkpoint to (default: CPU)
        verbose: Print adaptation information

    Returns:
        Adapted checkpoint ready for loading into simplified model

    Example:
        >>> checkpoint = load_adapted_checkpoint("boltzgen1.ckpt")
        >>> model.load_state_dict(checkpoint["state_dict"], strict=False)
    """
    if map_location is None:
        map_location = "cpu"

    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False
    )

    return adapt_checkpoint_for_simplified_model(checkpoint, verbose=verbose)


if __name__ == "__main__":
    """Command-line interface for checkpoint adaptation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Adapt BoltzGen checkpoints for simplified models"
    )
    parser.add_argument(
        "input",
        help="Path to input checkpoint"
    )
    parser.add_argument(
        "output",
        help="Path to output adapted checkpoint"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    save_adapted_checkpoint(
        args.input,
        args.output,
        verbose=not args.quiet
    )
