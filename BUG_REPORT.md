# Bug Report: Training Loss Always Reported as 0.0

**Date:** 2026-03-05
**Severity:** Critical — all experiment metrics are invalid
**Status:** FIXED

## Symptom

Every experiment (Phase 1, Experiment 6, Experiment 7) reports:
```
final_train_loss = 0.0
final_val_loss = 0.0
```

Base model and Base+LoRA show identical resolve rates (2/25, same two tasks), making it impossible to tell whether LoRA training had any effect.

## Root Cause

**Key name mismatch in `_MetricsCallback`** (`looper/trainers/lora_trainer.py:67-72`).

The callback checked for `train_info["loss"]` and `val_info["loss"]`, but MLX's training loop (`mlx_lm.tuner.trainer.train`) passes:
- `train_info["train_loss"]` — note the `train_` prefix
- `val_info["val_loss"]` — note the `val_` prefix

Since `"loss"` was never in the dict, the callback lists stayed empty, and the fallback `... if callback.train_losses else 0.0` always returned `0.0`.

**The training itself was working correctly** — LoRA weights were being computed and saved. Only the loss reporting was broken.

### Secondary issue: validation never evaluated

`TrainingArgs.steps_per_eval` defaults to 200, but training only ran 100 iterations. While MLX does evaluate at `it == 1` and `it == args.iters` regardless, this was implicit. Fixed by explicitly setting `steps_per_eval=config.iters`.

## Fix

```diff
-    def on_train_loss_report(self, train_info: dict):
-        if "loss" in train_info:
-            self.train_losses.append(train_info["loss"])
+    def on_train_loss_report(self, train_info: dict):
+        if "train_loss" in train_info:
+            self.train_losses.append(train_info["train_loss"])

-    def on_val_loss_report(self, val_info: dict):
-        if "loss" in val_info:
-            self.val_losses.append(val_info["loss"])
+    def on_val_loss_report(self, val_info: dict):
+        if "val_loss" in val_info:
+            self.val_losses.append(val_info["val_loss"])
```

Also set `steps_per_eval=config.iters` in `TrainingArgs` to ensure validation runs at end of training.

## Verification

After fix, 20-iteration training on existing Phase 1 data:

```
Iter 1:  Val loss  3.422
Iter 10: Train loss 1.591
Iter 20: Val loss  0.974
Iter 20: Train loss 1.006

Reported: final_train_loss=1.006, final_val_loss=0.974
```

Training is working — loss drops from ~3.4 → ~1.0 over 20 iterations. The adapter weights ARE being learned. The 0.0 was purely a logging bug.

## Impact on Prior Results

- **Phase 1 full ablation:** All reported `train_loss=0.0` values are wrong. Training was happening but loss wasn't captured.
- **Experiment 6 (format comparison):** All formats reported `train_loss=0.0` — cannot compare training dynamics between formats without rerunning.
- **Experiment 7 (budget sweep):** Same issue.
- **Resolve rates are still valid** — the adapter weights were saved correctly; only the loss metric was missing. The identical 2/25 resolve rate for base vs LoRA is a real result (the LoRA adapter didn't help on the 7B model with this training data), not an artifact of this bug.

## Files Changed

- `looper/trainers/lora_trainer.py` — lines 68-72 (callback key names), line 135 (steps_per_eval)
