import math
from pathlib import Path
import time
from typing import Any, Dict, Optional, List

import numpy as np
import torch
from pytorch_lightning import Callback, LightningModule
from torch import Tensor, nn

import boltzgen.model.layers.initialize as init
from boltzgen.data import const
from boltzgen.data.mol import (
    minimum_lddt_symmetry_dist,
)
from boltzgen.model.layers.pairformer import PairformerModule
from boltzgen.model.loss.distogram import distogram_loss
from boltzgen.model.loss.res_type import res_type_loss_fn

from boltzgen.model.modules.diffusion import AtomDiffusion
from boltzgen.model.modules.diffusion_conditioning import (
    DiffusionConditioning,
)
from boltzgen.model.modules.encoders import RelativePositionEncoder
from boltzgen.model.modules.masker import BoltzMasker
from boltzgen.model.modules.trunk import (
    ContactConditioning,
    DistogramModule,
    InputEmbedder,
    MSAModule,
    TemplateModule,
    TokenDistanceModule,
)
from boltzgen.model.optim.ema import EMA
from boltzgen.model.optim.scheduler import AlphaFoldLRScheduler

class Boltz(LightningModule):
    """Boltz Implementation - Simplified for Design/Diffusion Only."""

    def __init__(
        self,
        atom_s: int,
        atom_z: int,
        token_s: int,
        token_z: int,
        num_bins: int,
        training_args: Dict[str, Any],
        validation_args: Dict[str, Any],
        embedder_args: Dict[str, Any],
        msa_args: Dict[str, Any],
        pairformer_args: Dict[str, Any],
        score_model_args: Dict[str, Any],
        diffusion_process_args: Dict[str, Any],
        diffusion_loss_args: Dict[str, Any],
        masker_args: dict[str, Any] = {},
        atom_feature_dim: int = 128,
        template_args: Optional[Dict] = None,
        atoms_per_window_queries: int = 32,
        atoms_per_window_keys: int = 128,
        ema: bool = False,
        ema_decay: float = 0.999,
        min_dist: float = 2.0,
        max_dist: float = 22.0,
        aggregate_distogram: bool = True,
        bond_type_feature: bool = False,
        no_random_recycling_training: bool = False,
        conditioning_cutoff_min: float = 4.0,
        conditioning_cutoff_max: float = 20.0,
        use_templates: bool = False,
        use_token_distances: bool = False,
        token_distance_args: Optional[Dict] = None,
        log_loss_every_steps: int = 50,
        checkpoint_diffusion_conditioning: bool = False,
        freeze_template_weights: bool = False,
        predict_res_type: bool = False,
        inference_logging: bool = False,
        # Legacy parameters that may appear in checkpoints - absorbed but ignored
        **kwargs,
    ) -> None:
        super().__init__()
        """
        Simplified Boltz module for design/diffusion training only.

        Removed features:
        - Inverse folding
        - Confidence prediction (ConfidenceModule)
        - Affinity prediction (AffinityModule)
        - B-factor prediction (BFactorModule)
        - Validation infrastructure
        - Kernel optimizations
        """
        self.save_hyperparameters()
        self.inference_logging = inference_logging

        # No random recycling
        self.no_random_recycling_training = no_random_recycling_training
        self.log_loss_every_steps = log_loss_every_steps

        # EMA
        self.use_ema = ema
        self.ema_decay = ema_decay

        # Arguments
        self.training_args = training_args
        self.validation_args = validation_args
        self.diffusion_loss_args = diffusion_loss_args
        self.predict_res_type = predict_res_type

        # Distogram
        self.num_bins = num_bins
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.aggregate_distogram = aggregate_distogram

        # Masker
        self.masker = BoltzMasker(**masker_args)

        # Input embeddings
        full_embedder_args = {
            "atom_s": atom_s,
            "atom_z": atom_z,
            "token_s": token_s,
            "token_z": token_z,
            "atoms_per_window_queries": atoms_per_window_queries,
            "atoms_per_window_keys": atoms_per_window_keys,
            "atom_feature_dim": atom_feature_dim,
            **embedder_args,
        }
        self.input_embedder = InputEmbedder(**full_embedder_args)

        self.s_init = nn.Linear(token_s, token_s, bias=False)
        self.z_init_1 = nn.Linear(token_s, token_z, bias=False)
        self.z_init_2 = nn.Linear(token_s, token_z, bias=False)

        self.rel_pos = RelativePositionEncoder(token_z)

        self.token_bonds = nn.Linear(
            1,
            token_z,
            bias=False,
        )
        self.bond_type_feature = bond_type_feature
        if bond_type_feature:
            self.token_bonds_type = nn.Embedding(len(const.bond_types) + 1, token_z)

        self.contact_conditioning = ContactConditioning(
            token_z=token_z,
            cutoff_min=conditioning_cutoff_min,
            cutoff_max=conditioning_cutoff_max,
        )

        # Normalization layers
        self.s_norm = nn.LayerNorm(token_s)
        self.z_norm = nn.LayerNorm(token_z)

        # Recycling projections
        self.s_recycle = nn.Linear(token_s, token_s, bias=False)
        self.z_recycle = nn.Linear(token_z, token_z, bias=False)
        init.gating_init_(self.s_recycle.weight)
        init.gating_init_(self.z_recycle.weight)

        # Pairwise stack
        self.use_token_distances = use_token_distances
        if self.use_token_distances:
            self.token_distance_module = TokenDistanceModule(
                token_z, **token_distance_args
            )

        self.freeze_template_weights = freeze_template_weights
        self.use_templates = use_templates

        if use_templates:
            self.template_module = TemplateModule(token_z, **template_args)

        self.msa_module = MSAModule(
            token_z=token_z,
            token_s=token_s,
            **msa_args,
        )
        self.pairformer_module = PairformerModule(
            token_s, token_z, **pairformer_args
        )
        self.checkpoint_diffusion_conditioning = checkpoint_diffusion_conditioning

        self.diffusion_conditioning = DiffusionConditioning(
            token_s=token_s,
            token_z=token_z,
            atom_s=atom_s,
            atom_z=atom_z,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            atom_encoder_depth=score_model_args["atom_encoder_depth"],
            atom_encoder_heads=score_model_args["atom_encoder_heads"],
            token_transformer_depth=score_model_args["token_transformer_depth"],
            token_transformer_heads=score_model_args["token_transformer_heads"],
            atom_decoder_depth=score_model_args["atom_decoder_depth"],
            atom_decoder_heads=score_model_args["atom_decoder_heads"],
            atom_feature_dim=atom_feature_dim,
            conditioning_transition_layers=score_model_args[
                "conditioning_transition_layers"
            ],
        )

        # Output modules
        self.structure_module = AtomDiffusion(
            score_model_args={
                "token_s": token_s,
                "atom_s": atom_s,
                "atoms_per_window_queries": atoms_per_window_queries,
                "atoms_per_window_keys": atoms_per_window_keys,
                "predict_res_type": predict_res_type,
                **score_model_args,
            },
            **diffusion_process_args,
        )
        self.distogram_module = DistogramModule(token_z, num_bins)

        if self.freeze_template_weights:
            for pn, p in self.named_parameters():
                if "template_module" in pn:
                    p.requires_grad = False
        self.timestamp = time.time()

        self.training_args.skip_batch_by_single_rep = getattr(
            self.training_args, "skip_batch_by_single_rep", False
        )
        if self.training_args.skip_batch_by_single_rep:
            self.skip_step_by_single_rep = False
            print(
                "skip_batch_by_single_rep is on. Will skip training step if single representation has unstable magnitude."
            )

    def on_before_optimizer_step(self, optimizer):
        for name, param in self.named_parameters():
            if param.grad is None and not (
                "template_module" in name and self.freeze_template_weights
            ):
                print("Grad is None for:", name)

        if self.training_args.skip_batch_by_single_rep and self.skip_step_by_single_rep:
            print(
                "detected unstable magnitude of single rep. not updating model parameters."
            )
            self.zero_grad()
            self.skip_step_by_single_rep = False

    def forward(
        self,
        feats: Dict[str, Tensor],
        recycling_steps: int = 0,
        num_sampling_steps: Optional[int] = None,
        multiplicity_diffusion_train: int = 1,
        step_scale: Optional[float] = None,
        noise_scale: Optional[float] = None,
    ) -> Dict[str, Tensor]:
        dict_out = {}
        if self.inference_logging:
            print("\nRunning Trunk.\n")

        s_inputs = self.input_embedder(feats)

        # Initialize the sequence embeddings
        s_init = self.s_init(s_inputs)

        # Initialize pairwise embeddings
        z_init = (
            self.z_init_1(s_inputs)[:, :, None]
            + self.z_init_2(s_inputs)[:, None, :]
        )
        relative_position_encoding = self.rel_pos(feats)
        z_init = z_init + relative_position_encoding
        z_init = z_init + self.token_bonds(feats["token_bonds"].float())
        if self.bond_type_feature:
            z_init = z_init + self.token_bonds_type(feats["type_bonds"].long())
        z_init = z_init + self.contact_conditioning(feats)

        # Perform rounds of the pairwise stack
        s = torch.zeros_like(s_init)
        z = torch.zeros_like(z_init)

        # Compute pairwise mask
        mask = feats["token_pad_mask"].float()
        pair_mask = mask[:, :, None] * mask[:, None, :]

        for i in range(recycling_steps + 1):
            with torch.set_grad_enabled(
                (
                    self.training
                    and (i == recycling_steps)
                )
            ):
                # Issue with unused parameters in autocast
                if (
                    self.training
                    and (i == recycling_steps)
                    and torch.is_autocast_enabled()
                ):
                    torch.clear_autocast_cache()

                # Apply recycling
                s = s_init + self.s_recycle(self.s_norm(s))
                z = z_init + self.z_recycle(self.z_norm(z))

                # Compute pairwise stack
                if self.use_token_distances:
                    z = z + self.token_distance_module(
                        z, feats, pair_mask, relative_position_encoding
                    )

                # Compute pairwise stack
                if self.use_templates:
                    z = z + self.template_module(
                        z, feats, pair_mask
                    )

                z = z + self.msa_module(
                    z, s_inputs, feats
                )

                s, z = self.pairformer_module(
                    s,
                    z,
                    mask=mask,
                    pair_mask=pair_mask,
                )

        pdistogram = self.distogram_module(z)
        dict_out["pdistogram"] = pdistogram.float()

        if self.checkpoint_diffusion_conditioning:
            (
                q,
                c,
                to_keys,
                atom_enc_bias,
                atom_dec_bias,
                token_trans_bias,
            ) = torch.utils.checkpoint.checkpoint(
                self.diffusion_conditioning,
                s,
                z,
                relative_position_encoding,
                feats,
            )
        else:
            (
                q,
                c,
                to_keys,
                atom_enc_bias,
                atom_dec_bias,
                token_trans_bias,
            ) = self.diffusion_conditioning(
                s_trunk=s,
                z_trunk=z,
                relative_position_encoding=relative_position_encoding,
                feats=feats,
            )
        diffusion_conditioning = {
            "q": q,
            "c": c,
            "to_keys": to_keys,
            "atom_enc_bias": atom_enc_bias,
            "atom_dec_bias": atom_dec_bias,
            "token_trans_bias": token_trans_bias,
        }

        # Inference mode - sample structures
        if not self.training:
            if self.inference_logging:
                print("\nRunning Structure Module.\n")
            with torch.autocast("cuda", enabled=False):
                struct_out = self.structure_module.sample(
                    s_trunk=s.float(),
                    s_inputs=s_inputs.float(),
                    feats=feats,
                    num_sampling_steps=num_sampling_steps,
                    atom_mask=feats["atom_pad_mask"].float(),
                    multiplicity=1,
                    diffusion_conditioning=diffusion_conditioning,
                    step_scale=step_scale,
                    noise_scale=noise_scale,
                    inference_logging=self.inference_logging,
                )

                dict_out.update(struct_out)

        # Training mode - compute diffusion loss
        if self.training:
            atom_coords = feats["coords"]
            B, K, L = atom_coords.shape[0:3]
            assert K in (
                multiplicity_diffusion_train,
                1,
            )
            atom_coords = atom_coords.reshape(B * K, L, 3)
            atom_coords = atom_coords.repeat_interleave(
                multiplicity_diffusion_train // K, 0
            )
            feats["coords"] = atom_coords  # (multiplicity, L, 3)
            assert len(feats["coords"].shape) == 3

            for idx in range(feats["token_index"].shape[0]):
                minimum_lddt_symmetry_dist(
                    pred_distogram=pdistogram[idx],
                    feats=feats,
                    index_batch=idx,
                )

            with torch.autocast("cuda", enabled=False):
                struct_out = self.structure_module(
                    s_trunk=s.float(),
                    s_inputs=s_inputs.float(),
                    feats=feats,
                    multiplicity=multiplicity_diffusion_train,
                    diffusion_conditioning=diffusion_conditioning,
                )
                dict_out.update(struct_out)

        # For stability checking
        dict_out["s_trunk"] = s
        return dict_out

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        start = time.time()

        # Sample recycling steps
        if self.no_random_recycling_training:
            recycling_steps = self.training_args.recycling_steps
        else:
            rgn = np.random.default_rng(self.global_step)
            recycling_steps = rgn.integers(
                0, self.training_args.recycling_steps + 1
            ).item()

        if self.training_args.get("sampling_steps_random", None) is not None:
            rgn_samplng_steps = np.random.default_rng(self.global_step)
            sampling_steps = rgn_samplng_steps.choice(
                self.training_args.sampling_steps_random
            )
        else:
            sampling_steps = self.training_args.sampling_steps

        # Mask features for conditioning
        feat_masked = self.masker(batch)

        # Compute the forward pass
        out = self(
            feats=feat_masked,
            recycling_steps=recycling_steps,
            num_sampling_steps=sampling_steps,
            multiplicity_diffusion_train=self.training_args.diffusion_multiplicity,
        )

        batch["coords"] = feat_masked["coords"].clone()

        # Compute losses
        disto_loss, _ = distogram_loss(
            out,
            batch,
            aggregate_distogram=self.aggregate_distogram,
        )
        try:
            diffusion_loss_dict = self.structure_module.compute_loss(
                batch,
                out,
                multiplicity=self.training_args.diffusion_multiplicity,
                **self.diffusion_loss_args,
            )
        except Exception as e:
            print(f"Skipping batch {batch_idx} due to error: {e}")
            return None

        if self.predict_res_type:
            res_type_loss, res_type_acc = res_type_loss_fn(out, batch)
        else:
            res_type_loss, res_type_acc = 0.0, 0.0

        # Skip step if single representation has unstable magnitude.
        if self.training_args.skip_batch_by_single_rep:
            s_trunk = out["s_trunk"]
            magnitudes = torch.linalg.norm(s_trunk, dim=-1)
            if torch.any(magnitudes > 40000):
                self.skip_step_by_single_rep = True
            self.log("train/single_norm", torch.mean(magnitudes), prog_bar=False)

        # Aggregate losses
        loss = (
            self.training_args.diffusion_loss_weight * diffusion_loss_dict["loss"]
            + self.training_args.distogram_loss_weight * disto_loss
            + self.training_args.get("res_type_loss_weight", 0.0) * res_type_loss
        )

        if not (self.global_step % self.log_loss_every_steps):
            # Log losses
            self.log("train/distogram_loss", disto_loss)
            self.log("train/res_type_loss", res_type_loss)
            self.log("train/res_type_acc", res_type_acc)
            self.log("train/diffusion_loss", diffusion_loss_dict["loss"])
            for k, v in diffusion_loss_dict["loss_breakdown"].items():
                self.log(f"train/{k}", v)

            self.log("train/loss", loss)
            self.log("train/forward_dur", time.time() - start)
            self.log("train/step_dur", time.time() - self.timestamp)
            self.timestamp = time.time()
            self.training_log()
        return loss

    def training_log(self):
        self.log("train/grad_norm", self.gradient_norm(self), prog_bar=False)
        self.log("train/param_norm", self.parameter_norm(self), prog_bar=False)

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", lr, prog_bar=False)

        self.log(
            "train/param_norm_msa_module",
            self.parameter_norm(self.msa_module),
            prog_bar=False,
        )

        self.log(
            "train/param_norm_pairformer_module",
            self.parameter_norm(self.pairformer_module),
            prog_bar=False,
        )

        self.log(
            "train/param_norm_structure_module",
            self.parameter_norm(self.structure_module),
            prog_bar=False,
        )

    def on_train_epoch_end(self):
        # Simplified - no validation metrics to log
        pass

    def gradient_norm(self, module):
        parameters = [
            p.grad.norm(p=2) ** 2
            for p in module.parameters()
            if p.requires_grad and p.grad is not None
        ]
        if len(parameters) == 0:
            return torch.tensor(
                0.0, device="cuda" if torch.cuda.is_available() else "cpu"
            )
        norm = torch.stack(parameters).sum().sqrt()
        return norm

    def parameter_norm(self, module):
        parameters = [p.norm(p=2) ** 2 for p in module.parameters() if p.requires_grad]
        if len(parameters) == 0:
            return torch.tensor(
                0.0, device="cuda" if torch.cuda.is_available() else "cpu"
            )
        norm = torch.stack(parameters).sum().sqrt()
        return norm

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer."""
        param_dict = dict(self.named_parameters())

        all_parameter_names = [
            pn for pn, p in self.named_parameters() if p.requires_grad
        ]

        if self.training_args.get("weight_decay", 0.0) > 0:
            w_decay = self.training_args.get("weight_decay", 0.0)
            if self.training_args.get("weight_decay_exclude", False):
                nodecay_params_names = [
                    pn
                    for pn in all_parameter_names
                    if (
                        "norm" in pn
                        or "rel_pos" in pn
                        or ".s_init" in pn
                        or ".z_init_" in pn
                        or "token_bonds" in pn
                        or "embed_atom_features" in pn
                        or "dist_bin_pairwise_embed" in pn
                    )
                ]
                nodecay_params = [param_dict[pn] for pn in nodecay_params_names]
                decay_params = [
                    param_dict[pn]
                    for pn in all_parameter_names
                    if pn not in nodecay_params_names
                ]
                optim_groups = [
                    {"params": decay_params, "weight_decay": w_decay},
                    {"params": nodecay_params, "weight_decay": 0.0},
                ]
                optimizer = torch.optim.AdamW(
                    optim_groups,
                    betas=(
                        self.training_args.adam_beta_1,
                        self.training_args.adam_beta_2,
                    ),
                    eps=self.training_args.adam_eps,
                    lr=self.training_args.base_lr,
                )

            else:
                optimizer = torch.optim.AdamW(
                    [param_dict[pn] for pn in all_parameter_names],
                    betas=(
                        self.training_args.adam_beta_1,
                        self.training_args.adam_beta_2,
                    ),
                    eps=self.training_args.adam_eps,
                    lr=self.training_args.base_lr,
                    weight_decay=self.training_args.get("weight_decay", 0.0),
                )
        else:
            optimizer = torch.optim.AdamW(
                [param_dict[pn] for pn in all_parameter_names],
                betas=(self.training_args.adam_beta_1, self.training_args.adam_beta_2),
                eps=self.training_args.adam_eps,
                lr=self.training_args.base_lr,
                weight_decay=self.training_args.get("weight_decay", 0.0),
            )

        if self.training_args.lr_scheduler == "af3":
            scheduler = AlphaFoldLRScheduler(
                optimizer,
                base_lr=self.training_args.base_lr,
                max_lr=self.training_args.max_lr,
                warmup_no_steps=self.training_args.lr_warmup_no_steps,
                start_decay_after_n_steps=self.training_args.lr_start_decay_after_n_steps,
                decay_every_n_steps=self.training_args.lr_decay_every_n_steps,
                decay_factor=self.training_args.lr_decay_factor,
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        elif self.training_args.lr_scheduler == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.training_args.max_lr,
                total_steps=self.trainer.estimated_stepping_batches,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }

        return optimizer

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # Apply checkpoint adapter to remove weights from unused modules
        from boltzgen.model.checkpoint_adapter import (
            get_removable_module_prefixes,
            get_removable_exact_keys,
        )

        # Get original state dict
        state_dict = checkpoint.get("state_dict", {})

        # Remove unwanted module weights
        removable_prefixes = get_removable_module_prefixes()
        removable_keys = get_removable_exact_keys()

        filtered_state_dict = {}
        removed_count = 0
        for key, value in state_dict.items():
            should_remove = False

            # Check exact match
            if key in removable_keys:
                should_remove = True

            # Check prefix match
            if not should_remove:
                for prefix in removable_prefixes:
                    if key.startswith(prefix):
                        should_remove = True
                        break

            if not should_remove:
                filtered_state_dict[key] = value
            else:
                removed_count += 1

        if removed_count > 0:
            print(f"Checkpoint adapter: Removed {removed_count} unused module weights")

        checkpoint["state_dict"] = filtered_state_dict

        # Remap old key naming
        remapped_state_dict = {
            k.replace(".token_transformer_layers.0.", ".token_transformer."): v
            for k, v in checkpoint["state_dict"].items()
        }
        checkpoint["state_dict"] = remapped_state_dict

        # Update hyperparameters to reflect simplified model
        if "hyper_parameters" in checkpoint:
            hyper_params = checkpoint["hyper_parameters"]

            # Set removed features to False/None
            hyper_params["inverse_fold"] = False
            hyper_params["confidence_prediction"] = False
            hyper_params["affinity_prediction"] = False
            hyper_params["predict_bfactor"] = False
            hyper_params["predict_res_type"] = False
            hyper_params["use_kernels"] = False
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

        # Ignore the lr from the checkpoint
        lr = self.training_args.max_lr
        weight_decay = self.training_args.weight_decay

        if "optimimzer_states" in checkpoint:
            for state in checkpoint["optimizer_states"]:
                for group in state["param_groups"]:
                    group["lr"] = lr
                    group["weight_decay"] = weight_decay
        if "lr_schedulers" in checkpoint:
            for scheduler in checkpoint["lr_schedulers"]:
                scheduler["max_lr"] = lr
                scheduler["base_lrs"] = [lr] * len(scheduler["base_lrs"])
                scheduler["_last_lr"] = [lr] * len(scheduler["_last_lr"])

        # Ignore the training diffusion_multiplicity and recycling steps from the checkpoint
        if "hyper_parameters" in checkpoint:
            checkpoint["hyper_parameters"]["training_args"]["max_lr"] = lr
            checkpoint["hyper_parameters"]["training_args"][
                "diffusion_multiplicity"
            ] = self.training_args.diffusion_multiplicity
            checkpoint["hyper_parameters"]["training_args"]["recycling_steps"] = (
                self.training_args.recycling_steps
            )
            checkpoint["hyper_parameters"]["training_args"]["weight_decay"] = (
                self.training_args.weight_decay
            )

    def configure_callbacks(self) -> List[Callback]:
        """Configure model callbacks."""
        return [EMA(self.ema_decay)] if self.use_ema else []
