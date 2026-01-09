"""
Tests for checkpoint adapter functionality.
"""
import pytest
import tempfile
from pathlib import Path
import torch

from boltzgen.model.checkpoint_adapter import (
    adapt_checkpoint_for_simplified_model,
    get_removable_module_prefixes,
    save_adapted_checkpoint,
    load_adapted_checkpoint,
)


class TestCheckpointAdapter:
    """Tests for checkpoint adaptation."""

    def test_removable_prefixes(self):
        """Test that removable prefixes are defined correctly."""
        prefixes = get_removable_module_prefixes()

        assert "inverse_folding_encoder." in prefixes
        assert "inverse_folding_decoder." in prefixes
        assert "confidence_module." in prefixes
        assert "affinity_module." in prefixes
        assert "bfactor_module." in prefixes
        assert "distogram_module." in prefixes

    def test_adapt_mock_checkpoint(self):
        """Test adaptation with a mock checkpoint."""
        # Create a mock checkpoint
        mock_checkpoint = {
            "state_dict": {
                # Core model weights (should be kept)
                "input_embedder.linear.weight": torch.randn(10, 10),
                "structure_module.score_model.token_transformer.layers.0.weight": torch.randn(5, 5),
                "pairformer_module.layer1.weight": torch.randn(8, 8),

                # Weights to remove
                "inverse_folding_encoder.layer1.weight": torch.randn(6, 6),
                "inverse_folding_decoder.layer1.weight": torch.randn(6, 6),
                "confidence_module.head.weight": torch.randn(4, 4),
                "affinity_module.predictor.weight": torch.randn(3, 3),
                "bfactor_module.output.weight": torch.randn(2, 2),
                "distogram_module.bins.weight": torch.randn(7, 7),
            },
            "hyper_parameters": {
                "inverse_fold": True,
                "confidence_prediction": True,
                "affinity_prediction": True,
            },
            "epoch": 100,
            "global_step": 50000,
        }

        # Adapt checkpoint
        adapted = adapt_checkpoint_for_simplified_model(
            mock_checkpoint,
            verbose=False
        )

        # Check that core weights are kept
        assert "input_embedder.linear.weight" in adapted["state_dict"]
        assert "structure_module.score_model.token_transformer.layers.0.weight" in adapted["state_dict"]
        assert "pairformer_module.layer1.weight" in adapted["state_dict"]

        # Check that unwanted weights are removed
        assert "inverse_folding_encoder.layer1.weight" not in adapted["state_dict"]
        assert "inverse_folding_decoder.layer1.weight" not in adapted["state_dict"]
        assert "confidence_module.head.weight" not in adapted["state_dict"]
        assert "affinity_module.predictor.weight" not in adapted["state_dict"]
        assert "bfactor_module.output.weight" not in adapted["state_dict"]
        assert "distogram_module.bins.weight" not in adapted["state_dict"]

        # Check hyperparameters were updated
        assert adapted["hyper_parameters"]["inverse_fold"] == False
        assert adapted["hyper_parameters"]["confidence_prediction"] == False
        assert adapted["hyper_parameters"]["affinity_prediction"] == False

        # Check metadata preserved
        assert adapted["epoch"] == 100
        assert adapted["global_step"] == 50000

    def test_ema_weights_preferred(self):
        """Test that EMA weights are used if available."""
        mock_checkpoint = {
            "state_dict": {
                "model.weight": torch.ones(5, 5),
            },
            "ema": {
                "model.weight": torch.zeros(5, 5),
            },
        }

        adapted = adapt_checkpoint_for_simplified_model(
            mock_checkpoint,
            verbose=False
        )

        # Should use EMA weights (zeros) not state_dict weights (ones)
        assert torch.all(adapted["state_dict"]["model.weight"] == 0)

    def test_key_remapping(self):
        """Test that old key naming is remapped."""
        mock_checkpoint = {
            "state_dict": {
                "structure_module.score_model.token_transformer_layers.0.attention.weight": torch.randn(4, 4),
                "other.module.weight": torch.randn(3, 3),
            },
        }

        adapted = adapt_checkpoint_for_simplified_model(
            mock_checkpoint,
            verbose=False
        )

        # Old naming should be remapped
        expected_key = "structure_module.score_model.token_transformer.attention.weight"
        assert expected_key in adapted["state_dict"]
        assert "structure_module.score_model.token_transformer_layers.0.attention.weight" not in adapted["state_dict"]

        # Other keys unchanged
        assert "other.module.weight" in adapted["state_dict"]

    def test_save_and_load_adapted(self):
        """Test saving and loading adapted checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock checkpoint
            original_path = Path(tmpdir) / "original.ckpt"
            adapted_path = Path(tmpdir) / "adapted.ckpt"

            mock_checkpoint = {
                "state_dict": {
                    "model.weight": torch.randn(5, 5),
                    "confidence_module.weight": torch.randn(3, 3),
                },
                "epoch": 50,
            }

            # Save original
            torch.save(mock_checkpoint, original_path)

            # Adapt and save
            save_adapted_checkpoint(
                str(original_path),
                str(adapted_path),
                verbose=False
            )

            # Load adapted
            loaded = torch.load(adapted_path)

            # Check adaptation worked
            assert "model.weight" in loaded["state_dict"]
            assert "confidence_module.weight" not in loaded["state_dict"]
            assert loaded["epoch"] == 50

    @pytest.mark.slow
    def test_adapt_real_checkpoint(self):
        """Test adaptation with actual BoltzGen checkpoint if available."""
        # Try to find a real checkpoint
        import subprocess

        # Check if model is downloaded
        try:
            result = subprocess.run(
                ["boltzgen", "download"],
                capture_output=True,
                text=True,
                timeout=10
            )
        except:
            pytest.skip("Could not download model")

        # Find checkpoint in cache
        import os
        cache_dir = Path(os.path.expanduser("~/.cache/huggingface/hub"))
        checkpoints = list(cache_dir.glob("**/boltzgen1*.ckpt"))

        if not checkpoints:
            pytest.skip("No real checkpoint available for testing")

        checkpoint_path = checkpoints[0]
        print(f"\nTesting with real checkpoint: {checkpoint_path}")

        # Load and adapt
        adapted = load_adapted_checkpoint(
            str(checkpoint_path),
            verbose=True
        )

        # Verify core modules are present
        assert any("structure_module" in k for k in adapted["state_dict"].keys())
        assert any("pairformer_module" in k for k in adapted["state_dict"].keys())

        # Verify removed modules are gone
        assert not any("inverse_folding" in k for k in adapted["state_dict"].keys())
        assert not any("confidence_module" in k for k in adapted["state_dict"].keys())
        assert not any("affinity_module" in k for k in adapted["state_dict"].keys())

        print(f"\n✓ Successfully adapted real checkpoint")
        print(f"  Original keys: {len(torch.load(checkpoint_path, map_location='cpu')['state_dict'])}")
        print(f"  Adapted keys: {len(adapted['state_dict'])}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
