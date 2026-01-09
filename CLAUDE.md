# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

BoltzGen is a universal binder design tool for protein/peptide design against various targets (proteins, peptides, small molecules, DNA). It uses a diffusion model for backbone generation, inverse folding for sequence design, and Boltz-2 for structure prediction and validation.

## Essential Commands

### Development Setup
```bash
# Install in development mode (includes training dependencies)
pip install -e .[dev]

# Install for inference only
pip install -e .
```

### Running BoltzGen

**Basic design workflow:**
```bash
# Check design specification
boltzgen check example/vanilla_peptide_with_target_binding_site/beetletert.yaml

# Run full pipeline (design, inverse fold, fold, analyze, filter)
boltzgen run example/vanilla_protein/1g13prot.yaml \
  --output workbench/test_run \
  --protocol protein-anything \
  --num_designs 10 \
  --budget 2

# Run specific pipeline steps only
boltzgen run example/cyclotide/3ivq.yaml \
  --output workbench/partial-run \
  --protocol peptide-anything \
  --steps design inverse_folding \
  --num_designs 2

# Rerun filtering with different thresholds (very fast, ~15-20 seconds)
boltzgen run example/binding_disordered_peptides/tpp4.yaml \
  --output workbench/tpp4 \
  --protocol protein-anything \
  --steps filtering \
  --refolding_rmsd_threshold 3.0 \
  --filter_biased=false \
  --alpha 0.2

# Resume interrupted runs (no progress lost)
boltzgen run example/vanilla_protein/1g13prot.yaml \
  --output workbench/test_run \
  --protocol protein-anything \
  --reuse
```

**Separate configure and execute:**
```bash
# Generate config files without running
boltzgen configure example/cyclotide/3ivq.yaml \
  --output workbench/test-peptide-protein \
  --protocol peptide-anything \
  --num_designs 2

# Execute pre-configured pipeline
boltzgen execute workbench/test-peptide-protein
```

**Merge multiple runs:**
```bash
# Merge results from parallel runs
boltzgen merge workbench/run_a workbench/run_b workbench/run_c \
  --output workbench/merged_run

# Rerun filtering on merged set
boltzgen run example/vanilla_protein/1g13prot.yaml \
  --steps filtering \
  --output workbench/merged_run \
  --protocol protein-anything \
  --budget 60
```

### Training Models

```bash
# Train small model (development, 8 GPUs, gradient accumulation 16)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python src/boltzgen/resources/main.py src/boltzgen/resources/config/train/boltzgen_small.yaml \
  name=boltzgen_small

# Train large model
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python src/boltzgen/resources/main.py src/boltzgen/resources/config/train/boltzgen.yaml \
  name=boltzgen_large

# Train inverse-folding model
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python src/boltzgen/resources/main.py src/boltzgen/resources/config/train/inverse_folding.yaml \
  name=boltzgen_if

# Resume from checkpoint
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python src/boltzgen/resources/main.py src/boltzgen/resources/config/train/boltzgen_small.yaml \
  pretrained=./training_data/boltzgen1_structuretrained_small.ckpt \
  name=boltzgen_small_pretrained
```

### Testing & Linting
```bash
# Run tests
pytest

# Lint code
ruff check src
```

### Docker
```bash
# Build image
docker build -t boltzgen .

# Build with pre-downloaded weights
docker build -t boltzgen:weights --build-arg DOWNLOAD_WEIGHTS=true .

# Run example
mkdir -p workdir cache
docker run --rm --gpus all \
  -v "$(realpath workdir)":/workdir \
  -v "$(realpath cache)":/cache \
  -v "$(realpath example)":/example \
  boltzgen \
  boltzgen run /example/vanilla_protein/1g13prot.yaml \
    --output /workdir/test \
    --protocol protein-anything \
    --num_designs 2
```

## Architecture

### Pipeline Flow

BoltzGen uses a modular pipeline architecture orchestrated by `src/boltzgen/cli/boltzgen.py`:

1. **Configuration Phase** (`boltzgen configure` or first step of `boltzgen run`):
   - Parses design specification YAML
   - Generates individual config files in `OUTPUT/config/<step>.yaml`
   - Creates manifest `OUTPUT/steps.yaml` listing enabled steps

2. **Execution Phase** (`boltzgen execute` or automatic in `boltzgen run`):
   - Each step runs as a subprocess: `python src/boltzgen/resources/main.py <config.yaml>`
   - `main.py` is a wrapper that instantiates the appropriate `Task` subclass and calls `.run()`
   - Steps run sequentially unless explicitly parallelized externally (e.g., SLURM job arrays)

### Pipeline Steps

All steps inherit from `Task` abstract base class (`src/boltzgen/task/task.py`):

1. **design** - Generate backbone structures using diffusion model (`src/boltzgen/task/predict/predict.py`, GPU)
2. **inverse_folding** - Design sequences for backbones (`src/boltzgen/task/predict/predict.py`, GPU)
3. **folding** - Refold designed binders with targets using Boltz-2 (`src/boltzgen/task/predict/predict.py`, GPU)
4. **design_folding** - Refold designed binders alone, no target (`src/boltzgen/task/predict/predict.py`, GPU, disabled for peptide/nanobody)
5. **affinity** - Predict binding affinity for small molecule binders (`src/boltzgen/task/predict/predict.py`, GPU, protein-small_molecule only)
6. **analysis** - Compute CPU metrics and aggregate GPU outputs (`src/boltzgen/task/analyze/analyze.py`, CPU)
7. **filtering** - Rank designs and apply diversity optimization (`src/boltzgen/task/filter/filter.py`, CPU, ~20s)

### Core Components

**Model Architecture** (`src/boltzgen/model/`):
- `models/boltz.py` - Main BoltzGen model (diffusion + inverse folding + folding)
- `layers/` - Pairformer, triangular attention, outer product mean, etc.
- `loss/` - Diffusion loss, confidence loss, distogram, etc.
- `optim/` - Schedulers, EMA
- `validation/` - RCSB validation, refolding validation, design validation

**Data Pipeline** (`src/boltzgen/data/`):
- `data.py` - Core data structures (Structure, Chain, Residue, Atom, Token, etc.)
- `parse/schema.py` - `YamlDesignParser` converts design YAML to internal data structures
- `parse/mmcif.py` - mmCIF parser
- `parse/pdb_parser.py` - PDB parser
- `tokenize/tokenizer.py` - Converts structures to model tokens
- `feature/featurizer.py` - Generates model features
- `mol.py` - Small molecule handling via RDKit
- `write/mmcif.py` - Writes output structures to mmCIF format

**Task Implementations** (`src/boltzgen/task/`):
- `predict/predict.py` - GPU tasks (design, inverse folding, folding, affinity)
- `analyze/analyze.py` - CPU analysis metrics
- `filter/filter.py` - Fast filtering and ranking
- `train/train.py` - Model training

**CLI** (`src/boltzgen/cli/boltzgen.py`):
- Orchestrates all commands: `run`, `configure`, `execute`, `check`, `merge`, `download`
- Manages protocol-specific configs and config overrides
- Launches subprocesses for each pipeline step

### Protocols

Protocols are predefined configurations for different design scenarios (defined in `src/boltzgen/cli/boltzgen.py`):

- **protein-anything**: Design proteins to bind proteins/peptides (includes design folding step)
- **peptide-anything**: Design (cyclic) peptides (no Cys in inverse folding, no design folding, no hydrophobic patch)
- **protein-small_molecule**: Design proteins to bind small molecules (includes affinity prediction)
- **antibody-anything**: Design antibody CDRs (no Cys, no design folding, no hydrophobic patch)
- **nanobody-anything**: Design nanobody CDRs (same as antibody-anything)

### Configuration System

Uses Hydra for hierarchical configuration:
- Default configs: `src/boltzgen/resources/config/`
- Step configs: `design.yaml`, `inverse_fold.yaml`, `fold.yaml`, `analysis.yaml`, `filtering.yaml`
- Training configs: `train/boltzgen.yaml`, `train/boltzgen_small.yaml`, `train/inverse_folding.yaml`
- Override format: `--config <step_name> arg1=value1 arg2=value2`

## Design Specification YAML

**Critical conventions:**
- All residue indices start at 1
- Use `label_asym_id` (canonical mmCIF index), NOT `auth_asym_id` (author index)
- Verify indices in https://molstar.org/viewer/ (hover shows label_seq_id on bottom right)
- File references in YAML are relative to the YAML file's directory

**Basic structure:**
```yaml
entities:
  - protein:  # Designed protein
      id: B
      sequence: 80..140  # Random length 80-140
  - file:  # Target from structure file
      path: 6m1u.cif  # relative to YAML location
      include:
        - chain: {id: A}
      binding_types:  # Optional binding site specification
        - chain: {id: A, binding: 5..7,13}
      structure_groups:  # Optional structure visibility control
        - group: {visibility: 1, id: A, res_index: 10..13}
      design:  # Optional: also redesign some target residues
        - chain: {id: A, res_index: 14..19}
      secondary_structure:  # Optional: specify secondary structure
        - chain: {id: A, loop: 14, helix: 15..17, sheet: 19}

constraints:  # Optional: covalent bonds (disulfides, staples)
  - bond:
      atom1: [R, 4, SG]
      atom2: [Q, 1, CK]
```

See `example/design_spec_showcasing_all_functionalities.yaml` for comprehensive examples.

## Important Notes

### File Locations
- Models download to `~/.cache` (override with `--cache` or `$HF_HOME`)
- Training data default location: `./training_data/`
- Output structure: `OUTPUT/intermediate_designs/`, `OUTPUT/intermediate_designs_inverse_folded/`, `OUTPUT/final_ranked_designs/`

### Performance
- Design step: most expensive, scales with `--num_designs` (e.g., 10,000-60,000 typical)
- Filtering step: very fast (~15-20 seconds), iterate on thresholds using `--steps filtering`
- Use `--reuse` to resume interrupted runs without losing progress
- Parallelization: Use SLURM job arrays (see `slurm-example/`) or run steps on multiple GPUs within a step

### Key Parameters
- `--num_designs`: Total designs to generate (e.g., 10,000-60,000 for production)
- `--budget`: Final diversity-optimized set size
- `--diffusion_batch_size`: Designs per batch (default: 1 if <100 designs, else 10)
- `--alpha`: Diversity vs quality tradeoff (0.0=quality-only, 1.0=diversity-only)
- `--protocol`: Determines default settings for design type
- `--inverse_fold_avoid`: Disallowed residues (default: 'C' for peptide/nanobody)

### Output Files
- `intermediate_designs/*.cif, *.npz` - Designed backbones
- `intermediate_designs_inverse_folded/refold_cif/` - Primary output: refolded complexes
- `final_ranked_designs/final_<budget>_designs/` - Final diversity-optimized set
- `final_ranked_designs/results_overview.pdf` - Plots
- `final_ranked_designs/all_designs_metrics.csv` - All design metrics
- `final_ranked_designs/final_designs_metrics_<budget>.csv` - Final set metrics

### SLURM Parallelization
Use job arrays to run many single-GPU jobs in parallel:
1. Modify `slurm-example/run.sh` and `slurm-example/run_job_array.slurm`
2. Submit: `bash run.sh submit`
3. After completion, merge: `bash run.sh process`

## Common Workflows

### Standard Design Run
1. Create design YAML with target and binder specification
2. Validate: `boltzgen check design.yaml` → visualize output in PyMOL/Molstar
3. Test run: `boltzgen run design.yaml --output test --protocol <protocol> --num_designs 50 --budget 10`
4. Production run: Increase `--num_designs` to 10,000-60,000
5. Iterate on filtering: `boltzgen run design.yaml --output test --steps filtering --budget 10 --alpha 0.05 --refolding_rmsd_threshold 3.0`

### Inverse Folding Only
```bash
boltzgen run example/inverse_folding/1brs.yaml \
  --output workbench/if-only \
  --only_inverse_fold \
  --inverse_fold_num_sequences 2
```

### Parallel Design with SLURM
1. Modify `slurm-example/run.sh` for your site/design
2. Submit array: `bash run.sh submit` (runs many small jobs)
3. Merge results: `bash run.sh process` (combines all runs)
4. Apply final filtering on merged set

### Training New Models
1. Download training data (targets, MSAs, mols, checkpoints) to `./training_data/`
2. Edit training YAML paths if needed (`src/boltzgen/resources/config/train/*.yaml`)
3. Launch training (8 GPUs recommended): See training commands above
4. Optionally resume from structure-pretrained checkpoint
