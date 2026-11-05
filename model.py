from enum import StrEnum
from typing import Any

import lightning as L
import torch
import torch.nn.functional as F
import torchmetrics
from transformers.optimization import get_scheduler

from config import ModelType, RunConfig
from dblock_modules import BlockPartition, EDMCoefficients, NoiseConfig, edm_loss_weight
from vit import ViTDiTForImageClassification, load_vit

Batch = dict[str, torch.Tensor]


class Split(StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class ViTModel(L.LightningModule):
    num_blocks: int = 1  # models that are not block-trained count as a single block

    def __init__(self, cfg: RunConfig) -> None:
        super().__init__()
        if cfg.image_size is None or cfg.num_labels is None:
            raise ValueError(
                "image_size and num_labels must be set from the DataModule"
            )
        self.args = cfg
        self.image_size: int = cfg.image_size
        self.num_labels: int = cfg.num_labels
        self.valid_metrics = torchmetrics.MetricCollection(
            {
                "acc": torchmetrics.Accuracy(
                    task="multiclass", num_classes=self.num_labels
                ),
                "f1": torchmetrics.F1Score(
                    task="multiclass", num_classes=self.num_labels
                ),
            },
            prefix="val/",
        )
        self.test_metrics = self.valid_metrics.clone(prefix="test/")
        self.save_hyperparameters(self.hparams_dict())

    def hparams_dict(self) -> dict[str, Any]:
        return self.args.to_dict()

    def configure_model(self) -> None:
        self.model = load_vit(image_size=self.image_size, num_labels=self.num_labels)
        print(self.model)
        if self.args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )
        scheduler = get_scheduler(
            name=self.args.scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.args.num_warmup_steps,
            num_training_steps=int(self.trainer.estimated_stepping_batches),
            scheduler_specific_kwargs=self.args.scheduler_specific_kwargs,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def forward(self, **kwargs: Any) -> torch.Tensor:
        return self.model(**kwargs).logits

    def metrics(
        self, step: Split, logits: torch.Tensor, labels: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        match step:
            case Split.VAL:
                return self.valid_metrics(
                    logits.view(-1, self.num_labels), labels.view(-1)
                )
            case Split.TEST:
                return self.test_metrics(
                    logits.view(-1, self.num_labels), labels.view(-1)
                )
            case _:
                raise NotImplementedError(f"Step {step} is not supported")

    def shared_step(
        self,
        batch: Batch,
        step: Split = Split.TRAIN,
        return_metrics: bool = False,
        **kwargs: Any,
    ) -> Any:
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]
        logits = self(pixel_values=pixel_values, **kwargs)
        if return_metrics:
            return self.metrics(step, logits, labels)

        loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
        loss_dict = {f"{step}/loss": loss}
        return loss, loss_dict

    def get_model_kwargs(self, batch: Batch) -> dict[str, Any]:
        return {}

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        batch_size = batch["pixel_values"].shape[0]
        model_kwargs = self.get_model_kwargs(batch)
        loss, loss_dict = self.shared_step(batch, step=Split.TRAIN, **model_kwargs)
        self.log_dict(loss_dict, batch_size=batch_size, prog_bar=True)
        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> None:
        batch_size = batch["pixel_values"].shape[0]
        model_kwargs = self.get_model_kwargs(batch)
        res = self.shared_step(
            batch, step=Split.VAL, return_metrics=True, **model_kwargs
        )
        self.log_dict(res, batch_size=batch_size, prog_bar=True)

    def test_step(self, batch: Batch, batch_idx: int) -> None:
        batch_size = batch["pixel_values"].shape[0]
        model_kwargs = self.get_model_kwargs(batch)
        res = self.shared_step(
            batch, step=Split.TEST, return_metrics=True, **model_kwargs
        )
        self.log_dict(res, batch_size=batch_size, prog_bar=True)


class ViTDBlockModel(ViTModel):
    """Classification as diffusion over L2-normalized label embeddings (paper §5.1). Method maths: dblock_modules."""

    model: ViTDiTForImageClassification
    sigmas: torch.Tensor

    def __init__(self, cfg: RunConfig) -> None:
        self.noise = NoiseConfig()
        self.sigma_data = self.noise.sigma_data
        self.gamma = cfg.gamma
        self.cfg_scale = cfg.cfg_scale
        self.class_dropout_prob = (
            cfg.class_dropout_prob if self.cfg_scale > 0.0 else 0.0
        )
        self.num_inference_steps = cfg.num_inference_steps or cfg.num_blocks
        super().__init__(cfg)
        self.num_blocks = cfg.num_blocks
        self.partition = BlockPartition(cfg.num_blocks, cfg.gamma, self.noise)
        self.layer_assignment: list[list[int]] = []
        self.register_buffer(
            "sigmas",
            self.partition.get_discrete_sigmas(self.num_inference_steps).to(
                self.device
            ),
        )

    def hparams_dict(self) -> dict[str, Any]:
        return super().hparams_dict() | {
            "gamma": self.gamma,
            "num_inference_steps": self.num_inference_steps,
            "cfg_scale": self.cfg_scale,
            "class_dropout_prob": self.class_dropout_prob,
        }

    def configure_model(self) -> None:
        model = load_vit(
            image_size=self.image_size, num_labels=self.num_labels, is_dblock=True
        )
        assert isinstance(model, ViTDiTForImageClassification)
        self.model = model
        self.layer_assignment = self.partition.layer_assignment(
            model.config.num_hidden_layers
        )
        print(self.model)

    def label_embedding_weight(self) -> torch.Tensor:
        return self.model.vit.embeddings.label_embeddings.weight

    def get_embeds(self, input_ids: torch.Tensor) -> torch.Tensor:
        return F.normalize(
            F.embedding(input_ids, self.label_embedding_weight()), p=2, dim=-1
        )

    def get_sigmas(self, n_samples: int) -> torch.Tensor:
        return self.partition.get_sigmas(self.partition.sample_block_idx(), n_samples)

    def get_weights(self, sigmas: torch.Tensor) -> torch.Tensor:
        return edm_loss_weight(sigmas, self.sigma_data)

    def estimate_target_layer(self, sigma: torch.Tensor) -> int:
        return self.partition.estimate_target_layer(sigma)

    def denoise(
        self,
        x: torch.Tensor,
        zt: torch.Tensor,
        sigma: torch.Tensor,
        block_idx: int | None = None,
    ) -> torch.Tensor:
        if block_idx is None:
            block_idx = self.estimate_target_layer(sigma)
        if self.class_dropout_prob > 0.0 and self.training:
            drop_x = torch.rand(x.shape[0], device=x.device) < self.class_dropout_prob
            x = torch.where(drop_x[:, None, None, None], torch.zeros_like(x), x)
        elif not self.training and self.cfg_scale > 0.0:
            x = torch.cat([torch.zeros_like(x), x])
            zt = torch.cat([zt] * 2)
            sigma = torch.cat([sigma] * 2)

        c = EDMCoefficients.from_sigma(sigma, self.sigma_data)
        outputs = self.model.forward_block(
            layer_indices=self.layer_assignment[block_idx],
            pixel_values=x,
            noisy_embeds=zt * c.c_in[:, None],
            timesteps=c.c_noise,
        )
        hidden, conditioning = outputs.last_hidden_state, outputs.conditioning
        assert hidden is not None and conditioning is not None
        model_out = hidden * c.c_out[:, None] + zt * c.c_skip[:, None]
        logits = self.model.forward_output_embeddings(
            model_out.unsqueeze(1), conditioning
        )
        if not self.training and self.cfg_scale > 0.0:
            logits_uncond, logits_cond = logits.chunk(2)
            logits = logits_uncond + self.cfg_scale * (logits_cond - logits_uncond)
        return logits

    def shared_step(
        self,
        batch: Batch,
        step: Split = Split.TRAIN,
        return_metrics: bool = False,
        **kwargs: Any,
    ) -> Any:
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]
        if return_metrics:
            return self.metrics(step, self.diffusion_step(pixel_values), labels)

        z = self.get_embeds(labels)
        sigmas = self.get_sigmas(z.shape[0])
        block_idx = self.estimate_target_layer(sigmas)
        sigmas = sigmas.to(z)
        zt = z + sigmas[:, None] * torch.randn_like(z)
        logits = self.denoise(pixel_values, zt, sigmas, block_idx)
        loss = F.cross_entropy(
            logits.view(-1, self.num_labels), labels.view(-1), reduction="none"
        )
        ce_loss = loss.mean()
        loss = (loss * self.get_weights(sigmas)[:, None]).mean()
        loss_dict = {
            f"{step}/loss": loss,
            f"{step}/loss_{block_idx}": loss,
            f"{step}/ce_loss": ce_loss,
            f"{step}/ce_loss_{block_idx}": ce_loss,
        }
        return loss, loss_dict

    def diffusion_step(self, x: torch.Tensor) -> torch.Tensor:
        """Euler sampler over the label-embedding space; returns logits from a final denoise at σ_min."""
        z = torch.randn(x.shape[0], self.model.config.hidden_size, device=self.device)
        z *= torch.sqrt(1.0 + self.sigmas[0] ** 2.0)
        s_in = x.new_ones([x.shape[0]])
        for i in range(self.sigmas.shape[0] - 1):
            sigma = self.sigmas[i] * s_in
            next_sigma = self.sigmas[i + 1] * s_in
            probs = F.softmax(self.denoise(x, z, sigma), dim=1)
            denoised = F.linear(
                probs, self.label_embedding_weight().t()
            )  # raw (un-normalized) weights, D13
            d = (z - denoised) / sigma[:, None]
            z = z + (next_sigma - sigma)[:, None] * d
        sigmas = torch.full((x.shape[0],), self.sigmas[-1].item(), device=self.device)
        return self.denoise(x, z, sigmas)


MODELS: dict[ModelType, type[ViTModel]] = {
    ModelType.VIT: ViTModel,
    ModelType.DBLOCK: ViTDBlockModel,
}


def load_model(cfg: RunConfig) -> ViTModel:
    return MODELS[cfg.model_type](cfg)
