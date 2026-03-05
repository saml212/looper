"""Tests for the looper.trainers module."""

import json
from pathlib import Path

import pytest

from looper.models import TrainingExample
from looper.trainers.data_formatter import prepare_training_dir
from looper.trainers.lora_trainer import LoRAConfig, train_lora, load_model_with_adapter
from looper.trainers.full_replay import full_replay_train


def _make_examples(n: int) -> list[TrainingExample]:
    """Create n dummy training examples."""
    return [
        TrainingExample(
            messages=[
                {"role": "user", "content": f"Question {i}"},
                {"role": "assistant", "content": f"Answer {i}"},
            ],
            source_pair_id=f"pair-{i}",
        )
        for i in range(n)
    ]


# ---------- prepare_training_dir ----------


class TestPrepareTrainingDir:
    def test_creates_train_and_valid_files(self, tmp_path: Path):
        examples = _make_examples(20)
        result = prepare_training_dir(examples, tmp_path)
        assert result == tmp_path
        assert (tmp_path / "train.jsonl").exists()
        assert (tmp_path / "valid.jsonl").exists()

    def test_all_examples_accounted_for(self, tmp_path: Path):
        examples = _make_examples(50)
        prepare_training_dir(examples, tmp_path)

        train_lines = (tmp_path / "train.jsonl").read_text().strip().splitlines()
        valid_lines = (tmp_path / "valid.jsonl").read_text().strip().splitlines()
        assert len(train_lines) + len(valid_lines) == 50

    def test_val_split_approximately_correct(self, tmp_path: Path):
        examples = _make_examples(100)
        prepare_training_dir(examples, tmp_path, val_split=0.2)

        train_lines = (tmp_path / "train.jsonl").read_text().strip().splitlines()
        valid_lines = (tmp_path / "valid.jsonl").read_text().strip().splitlines()
        assert len(valid_lines) == 20
        assert len(train_lines) == 80

    def test_each_line_is_valid_json_with_messages(self, tmp_path: Path):
        examples = _make_examples(10)
        prepare_training_dir(examples, tmp_path)

        for fname in ["train.jsonl", "valid.jsonl"]:
            for line in (tmp_path / fname).read_text().strip().splitlines():
                obj = json.loads(line)
                assert "messages" in obj
                assert isinstance(obj["messages"], list)
                assert len(obj["messages"]) >= 2

    def test_deterministic_with_same_seed(self, tmp_path: Path):
        examples = _make_examples(30)

        dir_a = tmp_path / "a"
        dir_a.mkdir()
        prepare_training_dir(examples, dir_a, seed=123)

        dir_b = tmp_path / "b"
        dir_b.mkdir()
        prepare_training_dir(examples, dir_b, seed=123)

        assert (dir_a / "train.jsonl").read_text() == (dir_b / "train.jsonl").read_text()
        assert (dir_a / "valid.jsonl").read_text() == (dir_b / "valid.jsonl").read_text()


# ---------- LoRAConfig ----------


class TestLoRAConfig:
    def test_default_values(self):
        cfg = LoRAConfig()
        assert cfg.rank == 16
        assert cfg.num_layers == 16
        assert cfg.learning_rate == 1e-4
        assert cfg.iters == 100
        assert cfg.batch_size == 4
        assert cfg.max_seq_length == 2048
        assert cfg.seed == 42

    def test_custom_values(self):
        cfg = LoRAConfig(rank=8, iters=200, learning_rate=5e-5)
        assert cfg.rank == 8
        assert cfg.iters == 200
        assert cfg.learning_rate == 5e-5


# ---------- train_lora ----------


class TestTrainLora:
    def test_calls_mlx_correctly(self, tmp_path: Path, monkeypatch):
        # Set up data dir with dummy files
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "train.jsonl").write_text(
            '{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}\n'
        )
        (data_dir / "valid.jsonl").write_text(
            '{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}\n'
        )

        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()

        # Mock mlx_lm.load
        class FakeModel:
            def freeze(self):
                pass

            def trainable_parameters(self):
                return {}

        fake_model = FakeModel()
        fake_tokenizer = object()
        load_calls = []

        def mock_load(model_name, **kwargs):
            load_calls.append((model_name, kwargs))
            return fake_model, fake_tokenizer

        # Mock linear_to_lora_layers
        lora_layer_calls = []

        def mock_linear_to_lora_layers(model, num_layers, config):
            lora_layer_calls.append((model, num_layers, config))

        # Mock train
        train_calls = []

        def mock_train(**kwargs):
            train_calls.append(True)

        # Mock save_adapters
        def mock_save(*args, **kwargs):
            pass

        # Mock dataset classes
        class MockChatDataset:
            def __init__(self, *args, **kwargs):
                pass

        class MockCacheDataset:
            def __init__(self, ds):
                pass

        monkeypatch.setattr("looper.trainers.lora_trainer.load", mock_load)
        monkeypatch.setattr("looper.trainers.lora_trainer.linear_to_loro_layers", mock_linear_to_lora_layers)
        monkeypatch.setattr("looper.trainers.lora_trainer.mlx_train", mock_train)
        monkeypatch.setattr("looper.trainers.lora_trainer.ChatDataset", MockChatDataset)
        monkeypatch.setattr("looper.trainers.lora_trainer.CacheDataset", MockCacheDataset)
        monkeypatch.setattr("looper.trainers.lora_trainer.save_adapters", mock_save)

        config = LoRAConfig(rank=8, num_layers=4)
        result = train_lora("test-model", data_dir, adapter_dir, config=config)

        # Verify load was called with correct model name
        assert len(load_calls) == 1
        assert load_calls[0][0] == "test-model"

        # Verify linear_to_lora_layers was called with correct config
        assert len(lora_layer_calls) == 1
        assert lora_layer_calls[0][1] == 4  # num_layers
        assert lora_layer_calls[0][2]["rank"] == 8  # rank from config

        # Verify return dict has expected keys
        assert "final_train_loss" in result
        assert "iters" in result


# ---------- load_model_with_adapter ----------


class TestLoadModelWithAdapter:
    def test_loads_without_adapter(self, monkeypatch):
        load_calls = []

        def mock_load(model_name, **kwargs):
            load_calls.append((model_name, kwargs))
            return object(), object()

        monkeypatch.setattr("looper.trainers.lora_trainer.load", mock_load)

        load_model_with_adapter("test-model")
        assert len(load_calls) == 1
        assert "adapter_path" not in load_calls[0][1]

    def test_loads_with_adapter(self, monkeypatch):
        load_calls = []

        def mock_load(model_name, **kwargs):
            load_calls.append((model_name, kwargs))
            return object(), object()

        monkeypatch.setattr("looper.trainers.lora_trainer.load", mock_load)

        load_model_with_adapter("test-model", adapter_path=Path("/tmp/adapter"))
        assert len(load_calls) == 1
        assert load_calls[0][1]["adapter_path"] == str(Path("/tmp/adapter"))


# ---------- full_replay_train ----------


class TestFullReplayTrain:
    def test_prepares_data_and_calls_train(self, tmp_path: Path, monkeypatch):
        examples = _make_examples(20)
        adapter_dir = tmp_path / "adapter"

        train_lora_calls = []

        def mock_train_lora(model_name, data_dir, adapter_output_dir, config=None):
            # Verify data files were created
            assert (data_dir / "train.jsonl").exists()
            assert (data_dir / "valid.jsonl").exists()
            train_lora_calls.append({
                "model_name": model_name,
                "data_dir": data_dir,
                "adapter_output_dir": adapter_output_dir,
                "config": config,
            })
            return {"final_train_loss": 0.5, "final_val_loss": 0.6, "iters": 100}

        monkeypatch.setattr("looper.trainers.full_replay.train_lora", mock_train_lora)

        config = LoRAConfig(rank=8)
        result = full_replay_train(examples, "test-model", adapter_dir, config=config)

        assert len(train_lora_calls) == 1
        assert train_lora_calls[0]["model_name"] == "test-model"
        assert train_lora_calls[0]["adapter_output_dir"] == adapter_dir
        assert train_lora_calls[0]["config"].rank == 8
        assert result["final_train_loss"] == 0.5
