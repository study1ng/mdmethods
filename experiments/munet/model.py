from experiments.nets.builder import Builder
from experiments.plan import Plan
from experiments.trainer import UNetTrainingModule
from monai.data.utils import iter_patch_position
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from monai.losses import DiceCELoss
from typing import Any, Generator
import json
import math
import platform
from importlib.metadata import PackageNotFoundError, version
from experiments.config import image_key, label_key
from experiments.munet.munet_bottleneck import (
    MUNetBottleneck,
    PatchFeature,
    PatchPosition,
)
from monai.metrics import DiceMetric
from experiments.munet.stitch_utils import split_to_patch, stitch_logits, BlendMode


class MUNetTrainingModule(UNetTrainingModule):
    def __init__(
        self,
        builder: Builder,
        *,
        weights=None,
        loss=DiceCELoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
            batch=True,
            lambda_dice=1.0,
            lambda_ce=1.0,
        ),
        global_positional_encoding_proposition: float = 0.5,
        pos_blend=lambda a, b: a + b,
        plan: Plan,
        overlap_scale: float = 0.25,
        gamma: float = 1e-5,
    ):
        """Training Module for MUNet

        Parameters
        ----------
        builder : Builder
            Builder to build
        weights : None | float | list[float]
            the weight which would be applied if deep_supervision,
            default to [1/2, 1/4, 1/8, ...],
            whose sum would be adjusted to 1.
            first element is about head
        plan : Plan
            plan
        loss : loss, optional
            loss function, by default DiceCELoss( include_background=False, to_onehot_y=True, softmax=True, batch=True, lambda_dice=1.0, lambda_ce=1.0, )
        global_positional_encoding_proposition : float, optional
            the proposition of positional encoding which represents the position the patch in whole image, by default 0.5
        pos_blend : (Tensor, Tensor) -> Tensor, optional
            the function to mix positional encoding and feature map, by default lambda
        """
        self.plan = plan
        self.overlap_scale = overlap_scale  # proposion of self.plan.patch_size
        self.overlap = tuple(int(self.overlap_scale * p) for p in self.plan.patch_size)
        self.global_positional_encoding_proposition = (
            global_positional_encoding_proposition
        )
        self.gamma = gamma
        super().__init__(builder, weights)
        self.unet.deep_supervision = False
        self.unet.decoder.deep_supervision = False
        self.deep_supervision = False
        self.save_hyperparameters()
        self.loss = loss
        self.metric = DiceMetric(
            include_background=False,
            reduction="mean",
            num_classes=self.unet.output_channel,
        )
        self.bottleneck = MUNetBottleneck(
            self.unet.skip_channels[-1],
            global_positional_encoding_proposition=self.global_positional_encoding_proposition,
            pos_blend=pos_blend,
            gamma=self.gamma,
        )
        self.automatic_optimization = False
        self.cache_skip_level = 2
        # Runtime inference switches, independent of checkpoint hyperparameters.
        self.no_mixing = False
        self.no_ppe = False
        self.diagnostics_stream = None

    def split_to_patch(
        self, image: torch.Tensor, label: torch.Tensor | None = None
    ) -> tuple[tuple[torch.Tensor, torch.Tensor | None, tuple[int, ...]], ...]:
        """
        Returns:
            ((patch_image, patch_label, patch_position)*)
        """
        # -> [(patch_image, patch_label, patch_position)*]
        # patch_positionはパッチの開始位置(バッチ次元, チャネル次元は含まない)
        image_split = split_to_patch(
            image, patch_size=self.plan.patch_size, overlap=self.overlap
        )
        if label is not None:
            assert image.shape == label.shape
            label_split = split_to_patch(
                label, patch_size=self.plan.patch_size, overlap=self.overlap
            )
            for i, l in zip(image_split, label_split):
                assert l[1] == i[1]
            patches = tuple(
                (image[0], label[0], image[1])
                for image, label in zip(image_split, label_split)
            )
        else:
            patches = tuple((image[0], None, image[1]) for image in image_split)
        return patches

    def _slice_encoder(self, x: torch.Tensor, end: int) -> tuple[torch.Tensor, ...]:
        if end == 0:
            return ()
        hi = self.unet.encoder.stem(x)  # (B, C, H, W, D)
        ret = [hi]
        for stage in self.unet.encoder.stages:
            if len(ret) >= end:
                return tuple(ret)
            hi = stage(hi)
            ret.append(hi)
        return tuple(ret)

    def forward(
        self,
        patches: tuple[tuple[torch.Tensor, torch.Tensor | None, tuple[int, ...]], ...],
    ) -> Generator[Any, None, None]:
        skips_map = {}
        lasts = []

        for patch_img, _, patch_pos in patches:
            skips = checkpoint(
                self.unet.encoder,
                patch_img,
                use_reentrant=False,
            )
            skips_map[patch_pos] = skips[self.cache_skip_level : -1]
            lasts.append(PatchFeature(skips[-1], patch_pos))

        patch_positions = tuple(last.pos for last in lasts)

        def checkpointed_bottleneck(*features: torch.Tensor):
            # Checkpoint recomputation must follow the same operations as the
            # original forward instead of taking the positional-encoding cache.
            self.bottleneck.pe.cached_penc = None
            self.bottleneck.pe.cached_shape = None
            patch_features = tuple(
                PatchFeature(feature, pos)
                for feature, pos in zip(features, patch_positions, strict=True)
            )
            return tuple(
                patch.feature for patch in self.bottleneck(
                    patch_features, no_mixing=self.no_mixing, no_ppe=self.no_ppe
                )
            )

        bottleneck_tensors = checkpoint(
            checkpointed_bottleneck,
            *(last.feature for last in lasts),
            use_reentrant=False,
        )
        bottleneck_features = tuple(
            PatchFeature(feature, pos)
            for feature, pos in zip(
                bottleneck_tensors,
                patch_positions,
                strict=True,
            )
        )
        # ma = 0.
        # for b, l in zip(bottleneck_features, lasts):
        #     ma = max(ma, (b.feature - l.feature).abs().max().item())
        # print("bottleneck change max: ", ma)

        for skip, patch_img, patch_label, patch_pos in self.to_skips(
            skips_map, bottleneck_features, patches
        ):
            skips = self._slice_encoder(patch_img, self.cache_skip_level)
            bottleneck_skips = (*skips, *skip)

            out = self.unet.decoder(bottleneck_skips)
            # p_out = self.unet.decoder(skips)
            # print("last max between two networks", last.abs().max().item())
            # print("last mean between two networks", last.abs().mean().item())
            # print("diff max between two networks", (out - p_out).abs().max().item())
            # print("diff mean between two networks", (out - p_out).abs().mean().item())
            yield patch_img, patch_label, out, patch_pos

    def _collect_patch_outputs(self, patches, process_patch=None):
        """Process each patch before collecting detached logits on CPU."""
        results = []
        for patch_index, (_, patch_label, out, patch_pos) in enumerate(self(patches)):
            if process_patch is not None:
                process_patch(patch_index, patch_label, out)
            results.append((out.detach().cpu(), patch_pos))
        return tuple(results)

    def _predict_from_patch_outputs(self, results, output_size):
        """Blend CPU logits and produce labels using the training output flow."""
        with torch.no_grad():
            out = stitch_logits(
                results,
                BlendMode.GAUSSIAN,
                output_size=output_size,
            )
        return out.argmax(1, keepdim=True)

    def training_step(self, batch, _):
        image, label = batch[image_key], batch[label_key]
        # image: the whole image
        # label: the whole ground truth
        # B, C, H, W, D, P: ボトルネックにおけるバッチサイズ, チャネル数, 高さ, 幅, 深さ, パッチ数
        patches = self.split_to_patch(image, label)
        try:
            all_loss = 0.0
            opt = self.optimizers()
            opt.zero_grad()

            def process_patch(patch_index, patch_label, out):
                nonlocal all_loss
                # p_out = self.unet(patch_img)
                # if self.deep_supervision:
                #     loss = 0
                #     for i in range(len(p_out)):
                #         loss += self.head_weights[i] * self.loss(
                #             p_out[i], F.interpolate(patch_label, p_out[i].shape[2:], mode="nearest")
                #         )
                #     top_loss = self.loss(p_out[0], label)
                # else:
                #     loss = self.loss(p_out, patch_label)
                #     top_loss = loss
                loss = self.loss(out, patch_label) / len(patches)
                # print(top_loss, loss * len(patches))
                # print(self.global_step)
                self.manual_backward(
                    loss,
                    retain_graph=patch_index < len(patches) - 1,
                )
                all_loss += loss.detach()
            results = self._collect_patch_outputs(patches, process_patch)
            scheduler = self.lr_schedulers()
            opt.step()
            scheduler.step()
        except Exception as e:
            print(image.shape)
            raise e
        # all_loss /= len(patches)

        self.log(
            "gamma",
            self.bottleneck.gamma.abs().max().item(),
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )
        self.log("training loss", all_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("lr", self.optimizers().param_groups[0]["lr"], prog_bar=True)
        pred = self._predict_from_patch_outputs(results, image.shape[2:])
        return {
            "loss": all_loss,
            "image": ("image", image.detach().cpu()),
            "gt": ("image", label.detach().cpu()),
            "out": ("image", pred.detach().cpu()),
        }

    def to_skips(
        self,
        skips_map: dict[PatchPosition, tuple[torch.Tensor, ...]],
        reshaped: tuple[PatchFeature, ...],
        patches: tuple[tuple[torch.Tensor, torch.Tensor | None, PatchPosition], ...],
    ) -> Generator[
        tuple[
            tuple[torch.Tensor, ...],
            torch.Tensor,
            torch.Tensor | None,
            PatchPosition,
        ],
        None,
        None,
    ]:
        """
        Packs several generators into one.

        Returns:
            Generator[(skip feature maps, patch image, patch label, patch position)]
        """
        assert len(reshaped) == len(patches), "Lengths of inputs must match."

        reshaped = {r.pos: r.feature for r in reshaped}

        for patch_image, patch_label, patch_position in patches:
            last = reshaped[patch_position]
            skips = (*skips_map[patch_position], last)
            yield (skips, patch_image, patch_label, patch_position)

    def validation_step(self, batch, _):
        image = batch[image_key]  # (B,C,H,W,D)
        label = batch[label_key]
        patches = self.split_to_patch(image, label)
        results = self._collect_patch_outputs(patches)
        pred = self._predict_from_patch_outputs(results, image.shape[2:])
        self.metric(pred.to(label.device), label)
        return {
            "image": ("image", image.detach().cpu()),
            "gt": ("image", label.detach().cpu()),
            "out": ("image", pred.detach().cpu()),
        }

    def on_validation_epoch_end(self):
        dice = self.metric.aggregate()
        self.log(
            "val_dice",
            dice,
            prog_bar=True,
            sync_dist=True,
        )
        self.metric.reset()

    def _write_diagnostics(self, record):
        def finite_json(value):
            if isinstance(value, float) and not math.isfinite(value):
                return None
            if isinstance(value, dict):
                return {key: finite_json(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [finite_json(item) for item in value]
            return value

        self.diagnostics_stream.write(json.dumps(finite_json(record), ensure_ascii=False, allow_nan=False) + "\n")
        self.diagnostics_stream.flush()

    @staticmethod
    def _diagnostic_stats(value):
        """Summarize finite values; spatial variation is measured per channel."""
        finite = torch.isfinite(value)
        count = finite.sum().item()
        record = {
            "elements": value.numel(), "finite": count == value.numel(),
            "nan_count": torch.isnan(value).sum().item(),
            "inf_count": torch.isinf(value).sum().item(),
            "min": None, "max": None, "mean": None, "mean_abs": None,
        }
        if count:
            valid = value if count == value.numel() else value[finite]
            record.update({
                "min": valid.min().item(), "max": valid.max().item(),
                "mean": valid.mean(dtype=torch.float64).item(),
                "mean_abs": valid.abs().mean(dtype=torch.float64).item(),
                "std": valid.std(correction=0).item(),
                "zero_fraction": (valid == 0).sum().item() / count,
            })
        if value.ndim >= 3 and record["finite"] and value.numel():
            flat = value.reshape(value.shape[0], value.shape[1], -1)
            std = flat.std(dim=-1, correction=0)
            record["spatial_std_per_channel"] = {
                "min": std.min().item(), "mean": std.mean().item(),
                "max": std.max().item(), "spatial_elements": flat.shape[-1],
            }
        return record

    @staticmethod
    def _diagnostic_difference(current, baseline):
        """Use chunked float64 comparisons without retaining difference volumes."""
        if current.shape != baseline.shape:
            return {"status": "shape_mismatch", "reference_shape": list(baseline.shape)}
        if not current.numel():
            return {"status": "empty_tensor"}
        maximum = total = reference_abs = 0.0
        tolerances = (0.0, 1e-8, 1e-6, 1e-4)
        matches = [0] * len(tolerances)
        equal = True
        left, right = current.reshape(-1), baseline.reshape(-1)
        for start in range(0, left.numel(), 262144):
            a = left[start : start + 262144].to(torch.float64)
            b = right[start : start + 262144].to(torch.float64)
            if not (torch.isfinite(a).all().item() and torch.isfinite(b).all().item()):
                return {"status": "nonfinite_tensor"}
            delta = (a - b).abs()
            maximum = max(maximum, delta.max().item())
            total += delta.sum().item()
            reference_abs += b.abs().sum().item()
            equal = equal and torch.equal(a, b)
            for index, atol in enumerate(tolerances):
                matches[index] += (delta <= atol + 1e-5 * b.abs()).sum().item()
        return {
            "status": "ok", "max_abs_diff": maximum,
            "mean_abs_diff": total / current.numel(), "exact_equal": equal,
            "relative_mean_abs_diff": total / reference_abs if reference_abs else None,
            "relative_status": "ok" if reference_abs else "zero_reference",
            "reference_mean_abs": reference_abs / current.numel(),
            "tolerance_rtol": 1e-5,
            "matching_fraction_by_atol": {
                f"{atol:g}": matches[index] / current.numel()
                for index, atol in enumerate(tolerances)
            },
        }

    def _start_diagnostic_session(self):
        if getattr(self, "_diagnostics_session_stream", None) is self.diagnostics_stream:
            return
        self._diagnostics_session_stream = self.diagnostics_stream
        self._diagnostics_case_references = {}
        packages = {}
        for package in ("torch", "lightning", "monai", "mamba-ssm", "causal-conv1d"):
            try:
                packages[package] = version(package)
            except PackageNotFoundError:
                packages[package] = "not_found"
        self._write_diagnostics({
            "event": "diagnostic_environment", "schema_version": 2,
            "python": platform.python_version(), "packages": packages,
            "cuda_version": torch.version.cuda,
            "device": str(self.device),
            "gpu": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
            "matmul_precision": torch.get_float32_matmul_precision(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "training": self.training, "gamma": self.bottleneck.gamma.detach().cpu().tolist(),
            "comparison_layout": "B,C,spatial (flattened for elementwise comparison)",
            "alignment": "local patch coordinates without registration; batch items compared by index",
            "within_case_reference": "first patch in each condition",
            "cross_case_reference": "first patch of the first observed case, same condition",
            "condition_reference": "normal, first patch only for intermediate features",
            "decoder_comparison": "whole stitched logits only; per-patch statistics retained",
            "nonfinite_json_values": "null; inspect finite/nan_count/inf_count/status",
            "scope": "observational; does not prove causal dependence or independence",
            "bimamba": {
                "class": type(self.bottleneck.bottleneck).__qualname__,
                "layer_norm_eps": self.bottleneck.bottleneck.norm.eps,
                "out_proj_has_bias": self.bottleneck.bottleneck.out_proj.bias is not None,
                "inference_params": None,
                "directions": [{
                    "class": f"{type(module).__module__}.{type(module).__qualname__}",
                    "use_fast_path": getattr(module, "use_fast_path", None),
                    "d_state": getattr(module, "d_state", None),
                    "d_conv": getattr(module, "d_conv", None),
                    "training": module.training,
                } for module in self.bottleneck.bottleneck.mambas],
            },
        })
        for name, parameter in self.bottleneck.named_parameters():
            self._write_diagnostics({
                "event": "parameter_stats", "parameter": name,
                "shape": list(parameter.shape), "dtype": str(parameter.dtype),
                "requires_grad": parameter.requires_grad,
                **self._diagnostic_stats(parameter.detach().to(device="cpu", dtype=torch.float32)),
            })

    def on_test_end(self):
        # Release first-case reference tensors even when the module stays alive.
        self._diagnostics_case_references = {}
        self._diagnostics_session_stream = None

    def _diagnostic_prediction(self, batch, patches, output_size, batch_index):
        """Compare four settings and intermediate features against normal inference.

        --diagnostics PATH writes UTF-8 JSON lines as each stage completes.
        patch_comparison compares all patches to the condition's first patch;
        cross_case_comparison compares each case's first patch to the first case.
        condition_comparison uses the normal condition's first patch. References
        contain one patch per stage, never complete decoder output volumes.
        Relative errors use mean absolute reference magnitude, with no epsilon.
        Tolerance fractions use abs(a-b) <= atol + 1e-5*abs(reference).
        Single-patch cases, shape mismatches, zero references and NaN/Inf are
        explicitly marked. No raw tensors are written. Requires one device;
        CPU memory and runtime increase. Whole-logit comparisons are unchanged.
        """
        self._start_diagnostic_session()
        if not patches:
            self._write_diagnostics({"event": "case_skipped", "batch_index": batch_index, "reason": "no_patches"})
            raise ValueError("Diagnostics cannot predict an image with no patches")
        selected_flags = (self.no_ppe, self.no_mixing)
        selected_pred = None
        reference = None
        reference_pred = None
        comparisons = {}
        normal_first = {}
        condition_first = {}
        first_position = tuple(patches[0][2])
        previous_observer = self.bottleneck.diagnostics_callback
        progress = {"stage": "case_start", "position": None}
        self._write_diagnostics({
            "event": "case_start", "batch_index": batch_index,
            "case": str(batch.get("name", "unknown")),
            "source": str(getattr(batch[image_key], "meta", {}).get("filename_or_obj", "unknown")),
            "image_shape": list(batch[image_key].shape),
            "selected_flags": {"no_ppe": selected_flags[0], "no_mixing": selected_flags[1]},
        })

        def observe(stage, value, position, flags):
            progress.update({"stage": stage, "position": None if position is None else list(position)})
            record = {
                "event": "bottleneck_call" if value is None else "diagnostic_stage",
                "batch_index": batch_index, "case": str(batch.get("name", "unknown")),
                "condition": name, "stage": stage,
            }
            if value is None:
                record["received_flags"] = flags
            else:
                # Canonical channel-first layout also aligns flattened PE stages
                # with the spatial BiMamba stages. CPU copy precedes in-place ops.
                comparison_dtype = torch.float64 if value.dtype == torch.float64 else torch.float32
                current = value.detach().to(device="cpu", dtype=comparison_dtype)
                if stage in {"encoder_features", "positional_encoding", "after_positional_encoding"}:
                    current = current.transpose(1, 2).contiguous()
                record.update({
                    "patch_position": list(position), "shape": list(current.shape),
                    "dtype": str(value.dtype), "device": str(value.device),
                    "comparison_dtype": str(current.dtype),
                    **self._diagnostic_stats(current),
                })
                if stage != "decoder_logits":
                    if stage not in condition_first:
                        snapshot = current.clone()
                        condition_first[stage] = snapshot
                        if name == "normal":
                            normal_first[stage] = snapshot
                        record["patch_comparison"] = {
                            "status": "reference_only" if len(patches) > 1 else "single_patch",
                            "reference_position": list(first_position),
                        }
                        record["condition_comparison"] = self._diagnostic_difference(current, normal_first[stage])
                        key = (name, stage)
                        context = {"batch_index": batch_index, "case": str(batch.get("name", "unknown")), "position": list(position)}
                        if key not in self._diagnostics_case_references:
                            self._diagnostics_case_references[key] = (snapshot, context)
                            record["cross_case_comparison"] = {"status": "first_case_reference", "reference": context}
                        else:
                            baseline, context = self._diagnostics_case_references[key]
                            record["cross_case_comparison"] = {
                                **self._diagnostic_difference(current, baseline), "reference": context,
                            }
                    else:
                        record["patch_comparison"] = {
                            **self._diagnostic_difference(current, condition_first[stage]),
                            "reference_position": list(first_position),
                        }
                        record["condition_comparison"] = {"status": "first_patch_only"}
                else:
                    record["patch_comparison"] = {"status": "not_retained_decoder_logits"}
            self._write_diagnostics(record)

        def observe_decoder(patch_index, patch_label, out):
            observe("decoder_logits", out, patches[patch_index][2], None)

        conditions = (
            ("normal", False, False), ("no_ppe", True, False),
            ("no_mixing", False, True), ("no_ppe_no_mixing", True, True),
        )
        name = "not_started"
        try:
            self.bottleneck.diagnostics_callback = observe
            for name, no_ppe, no_mixing in conditions:
                condition_first = {}
                progress.update({"stage": "condition_start", "position": None})
                self.no_ppe, self.no_mixing = no_ppe, no_mixing
                self._write_diagnostics({
                    "event": "condition_start", "batch_index": batch_index,
                    "case": str(batch.get("name", "unknown")), "condition": name,
                    "no_ppe": no_ppe, "no_mixing": no_mixing, "patch_count": len(patches),
                    "patch_positions": [list(position) for _, _, position in patches],
                    "single_patch": len(patches) == 1,
                })
                results = self._collect_patch_outputs(patches, observe_decoder)
                logits = stitch_logits(results, BlendMode.GAUSSIAN, output_size=output_size)
                del results
                self._write_diagnostics({
                    "event": "stitched_logits", "batch_index": batch_index,
                    "condition": name, "min": logits.min().item(),
                    "max": logits.max().item(), "dtype": str(logits.dtype),
                })
                if not torch.isfinite(logits).all().item():
                    raise ValueError(f"Nonfinite diagnostic logits: batch={batch_index}, condition={name}")
                prediction = logits.argmax(1, keepdim=True)
                if (no_ppe, no_mixing) == selected_flags:
                    selected_pred = prediction
                if reference is None:
                    reference, reference_pred = logits, prediction
                else:
                    difference = (logits - reference).abs()
                    changed = (prediction != reference_pred).sum().item()
                    count = prediction.numel()
                    maximum = difference.max().item()
                    total_difference = difference.sum(dtype=torch.float64).item()
                    comparisons[name] = {
                        "logits_max_abs_diff": maximum,
                        "logits_mean_abs_diff": total_difference / difference.numel(),
                        "changed_voxels": changed, "voxels": count,
                        "changed_fraction": changed / count,
                    }
                    totals = self.diagnostics_totals.setdefault(name, {
                        "logits_max_abs_diff": 0.0, "logits_abs_diff_sum": 0.0,
                        "logits_elements": 0, "changed_voxels": 0, "voxels": 0,
                    })
                    totals["logits_max_abs_diff"] = max(totals["logits_max_abs_diff"], maximum)
                    totals["logits_abs_diff_sum"] += total_difference
                    totals["logits_elements"] += difference.numel()
                    totals["changed_voxels"] += changed
                    totals["voxels"] += count
                    totals["logits_mean_abs_diff"] = totals["logits_abs_diff_sum"] / totals["logits_elements"]
                    totals["changed_fraction"] = totals["changed_voxels"] / totals["voxels"]
                    del difference
                del logits, prediction
                self._write_diagnostics({"event": "condition_end", "batch_index": batch_index, "condition": name})
        except BaseException as exc:
            self._write_diagnostics({
                "event": "diagnostic_interrupted", "batch_index": batch_index,
                "condition": name, "last_observed": progress,
                "exception_type": type(exc).__qualname__, "message": str(exc),
            })
            self._diagnostics_case_references = {}
            raise
        finally:
            self.no_ppe, self.no_mixing = selected_flags
            self.bottleneck.diagnostics_callback = previous_observer
        metadata = getattr(batch[image_key], "meta", {})
        self._write_diagnostics({
            "event": "case", "batch_index": batch_index,
            "case": str(batch.get("name", "unknown")),
            "source": str(metadata.get("filename_or_obj", "unknown")),
            "image_shape": list(batch[image_key].shape),
            "patch_count": len(patches),
            "patch_positions": [list(position) for _, _, position in patches],
            "comparisons": comparisons,
        })
        self.diagnostics_cases += 1
        return selected_pred

    def test_step(self, batch, batch_index):
        image = batch[image_key]  # (B,C,H,W,D)
        patches = self.split_to_patch(image)
        if self.diagnostics_stream is None:
            results = self._collect_patch_outputs(patches)
            pred = self._predict_from_patch_outputs(results, image.shape[2:])
        else:
            pred = self._diagnostic_prediction(batch, patches, image.shape[2:], batch_index)
        return {
            "image": ("image", image.detach().cpu()),
            "out": ("image", pred.detach().cpu()),
        }

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            [
                {
                    "params": self.unet.parameters(),
                    "lr": 4e-4,
                },
                {
                    "params": self.bottleneck.parameters(),
                    "lr": 4e-4,
                },
            ],
            eps=1e-5,
            weight_decay=1e-1,
            betas=(0.9, 0.95),
        )
        total_steps = self.trainer.estimated_stepping_batches
        endless = total_steps == float("inf")
        warmup_steps = min(total_steps, 250 * 1000) // 10
        scheduler = (
            torch.optim.lr_scheduler.SequentialLR(
                optim,
                [
                    torch.optim.lr_scheduler.LinearLR(
                        optim,
                        start_factor=1e-10,
                        end_factor=1.0,
                        total_iters=warmup_steps,
                    ),
                    torch.optim.lr_scheduler.CosineAnnealingLR(
                        optim, T_max=total_steps - warmup_steps, eta_min=1e-5
                    ),
                ],
                milestones=[warmup_steps],
            )
            if not endless
            else torch.optim.lr_scheduler.SequentialLR(
                optim,
                [
                    torch.optim.lr_scheduler.LinearLR(
                        optim,
                        start_factor=1e-10,
                        end_factor=1.0,
                        total_iters=warmup_steps,
                    ),
                    torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        optim,
                        T_0=100000,
                        T_mult=2,
                        eta_min=1e-5,
                    ),
                ],
                milestones=[warmup_steps],
            )
        )
        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
