"""
Test suite for the design step of the BoltzGen pipeline.

This test suite captures the behavior of the design step to enable
safe refactoring of the codebase.
"""
import os
import shutil
import tempfile
from pathlib import Path
import pytest
import yaml
import numpy as np
import gemmi


@pytest.fixture
def test_output_dir():
    """Create a temporary output directory for tests."""
    temp_dir = tempfile.mkdtemp(prefix="boltzgen_test_")
    yield temp_dir
    # Cleanup after test
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def simple_design_spec(tmp_path):
    """Create a simple design specification YAML for testing."""
    # Create a simple design spec with a designed protein
    design_spec = {
        "entities": [
            {
                "protein": {
                    "id": "A",
                    "sequence": "10..15"  # Small for fast testing
                }
            }
        ]
    }

    spec_path = tmp_path / "simple_design.yaml"
    with open(spec_path, 'w') as f:
        yaml.dump(design_spec, f)

    return str(spec_path)


@pytest.fixture
def protein_target_design_spec(tmp_path):
    """Create a design spec with a protein target."""
    # Use the example from the repo
    example_dir = Path(__file__).parent.parent / "example" / "vanilla_protein"
    if not example_dir.exists():
        pytest.skip("Example directory not found")

    # Copy the example files to tmp directory
    spec_file = example_dir / "1g13prot.yaml"
    target_file = example_dir / "1g13.cif"

    if not spec_file.exists() or not target_file.exists():
        pytest.skip("Example files not found")

    # Copy files
    shutil.copy(spec_file, tmp_path / "design.yaml")
    shutil.copy(target_file, tmp_path / "1g13.cif")

    return str(tmp_path / "design.yaml")


class TestDesignStepBasic:
    """Basic tests for the design step."""

    def test_design_imports(self):
        """Test that we can import the necessary modules."""
        from boltzgen.task.predict.predict import Predict
        from boltzgen.model.models.boltz import Boltz
        from boltzgen.data.parse.schema import YamlDesignParser
        assert Predict is not None
        assert Boltz is not None
        assert YamlDesignParser is not None

    def test_design_spec_parsing(self, simple_design_spec):
        """Test that design specifications can be parsed."""
        from boltzgen.data.parse.schema import YamlDesignParser
        from boltzgen.data.tokenize.tokenizer import Tokenizer
        from boltzgen.data.feature.featurizer import Featurizer

        # Parse the design spec
        parser = YamlDesignParser(
            yaml_path=simple_design_spec,
            tokenizer=Tokenizer(),
            featurizer=Featurizer(),
            moldir=None
        )

        # Check that parsing succeeds
        assert parser is not None


class TestDesignStepOutputs:
    """Tests for design step outputs."""

    @pytest.mark.slow
    def test_design_generates_outputs(self, simple_design_spec, test_output_dir):
        """Test that the design step generates expected output files."""
        import subprocess

        # Run design step only
        cmd = [
            "boltzgen", "run",
            simple_design_spec,
            "--output", test_output_dir,
            "--protocol", "protein-anything",
            "--steps", "design",
            "--num_designs", "2",
            "--diffusion_batch_size", "1"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Check that command succeeded
        if result.returncode != 0:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
        assert result.returncode == 0, f"Design step failed: {result.stderr}"

        # Check output directory structure
        output_path = Path(test_output_dir)
        assert output_path.exists()

        # Check for config files
        config_dir = output_path / "config"
        assert config_dir.exists()
        assert (config_dir / "design.yaml").exists()

        # Check for intermediate designs
        designs_dir = output_path / "intermediate_designs"
        assert designs_dir.exists()

        # Check for .cif and .npz files
        cif_files = list(designs_dir.glob("*.cif"))
        npz_files = list(designs_dir.glob("*.npz"))

        assert len(cif_files) == 2, f"Expected 2 CIF files, got {len(cif_files)}"
        assert len(npz_files) == 2, f"Expected 2 NPZ files, got {len(npz_files)}"

    @pytest.mark.slow
    def test_design_output_structure(self, simple_design_spec, test_output_dir):
        """Test that generated designs have correct structure."""
        import subprocess

        # Run design step
        cmd = [
            "boltzgen", "run",
            simple_design_spec,
            "--output", test_output_dir,
            "--protocol", "protein-anything",
            "--steps", "design",
            "--num_designs", "1",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0

        # Load and check a generated CIF file
        designs_dir = Path(test_output_dir) / "intermediate_designs"
        cif_files = list(designs_dir.glob("*.cif"))
        assert len(cif_files) > 0

        # Parse CIF file
        structure = gemmi.read_structure(str(cif_files[0]))

        # Check basic structure properties
        assert len(structure) > 0, "Structure should have at least one model"
        model = structure[0]
        assert len(model) > 0, "Model should have at least one chain"

        # Check that chain has residues
        for chain in model:
            assert len(chain) > 0, f"Chain {chain.name} should have residues"

    @pytest.mark.slow
    def test_design_npz_contents(self, simple_design_spec, test_output_dir):
        """Test that NPZ files contain expected data."""
        import subprocess

        # Run design step
        cmd = [
            "boltzgen", "run",
            simple_design_spec,
            "--output", test_output_dir,
            "--protocol", "protein-anything",
            "--steps", "design",
            "--num_designs", "1",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0

        # Load NPZ file
        designs_dir = Path(test_output_dir) / "intermediate_designs"
        npz_files = list(designs_dir.glob("*.npz"))
        assert len(npz_files) > 0

        # Load and check contents
        data = np.load(npz_files[0], allow_pickle=True)

        # Check for expected keys (this may need adjustment based on actual output)
        # At minimum, we expect some coordinate data
        assert len(data.files) > 0, "NPZ file should contain data"


class TestDesignStepParameters:
    """Tests for design step with different parameters."""

    @pytest.mark.slow
    def test_design_multiple_samples(self, simple_design_spec, test_output_dir):
        """Test generating multiple designs."""
        import subprocess

        num_designs = 3
        cmd = [
            "boltzgen", "run",
            simple_design_spec,
            "--output", test_output_dir,
            "--protocol", "protein-anything",
            "--steps", "design",
            "--num_designs", str(num_designs),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0

        # Check we got the right number of outputs
        designs_dir = Path(test_output_dir) / "intermediate_designs"
        cif_files = list(designs_dir.glob("*.cif"))
        assert len(cif_files) == num_designs

    @pytest.mark.slow
    def test_design_with_protein_target(self, protein_target_design_spec, test_output_dir):
        """Test design with a protein target."""
        import subprocess

        cmd = [
            "boltzgen", "run",
            protein_target_design_spec,
            "--output", test_output_dir,
            "--protocol", "protein-anything",
            "--steps", "design",
            "--num_designs", "1",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
        assert result.returncode == 0

        # Verify outputs
        designs_dir = Path(test_output_dir) / "intermediate_designs"
        cif_files = list(designs_dir.glob("*.cif"))
        assert len(cif_files) > 0


class TestDesignStepReuse:
    """Tests for design step reuse functionality."""

    @pytest.mark.slow
    def test_design_reuse(self, simple_design_spec, test_output_dir):
        """Test that --reuse flag works correctly."""
        import subprocess

        # First run: generate 2 designs
        cmd1 = [
            "boltzgen", "run",
            simple_design_spec,
            "--output", test_output_dir,
            "--protocol", "protein-anything",
            "--steps", "design",
            "--num_designs", "2",
        ]

        result1 = subprocess.run(cmd1, capture_output=True, text=True)
        assert result1.returncode == 0

        # Count initial files
        designs_dir = Path(test_output_dir) / "intermediate_designs"
        initial_cif_files = set(designs_dir.glob("*.cif"))
        assert len(initial_cif_files) == 2

        # Second run with reuse: ask for 3 total (should generate only 1 more)
        cmd2 = [
            "boltzgen", "run",
            simple_design_spec,
            "--output", test_output_dir,
            "--protocol", "protein-anything",
            "--steps", "design",
            "--num_designs", "3",
            "--reuse"
        ]

        result2 = subprocess.run(cmd2, capture_output=True, text=True)
        assert result2.returncode == 0

        # Count final files
        final_cif_files = set(designs_dir.glob("*.cif"))
        assert len(final_cif_files) == 3

        # Original files should still exist
        assert initial_cif_files.issubset(final_cif_files)


class TestDesignStepCheckCommand:
    """Tests for the boltzgen check command."""

    def test_check_command(self, simple_design_spec, test_output_dir):
        """Test that check command works."""
        import subprocess

        output_dir = Path(test_output_dir) / "check_output"

        cmd = [
            "boltzgen", "check",
            simple_design_spec,
            "--output", str(output_dir)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
        assert result.returncode == 0

        # Check output file exists
        # boltzgen check creates a directory with the design spec name as .cif
        cif_files = list(output_dir.glob("*.cif"))
        assert len(cif_files) > 0, f"No CIF files found in {output_dir}"

        # Verify it's a valid CIF
        structure = gemmi.read_structure(str(cif_files[0]))
        assert len(structure) > 0


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
