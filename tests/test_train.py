"""Tests for the cellpose training pipeline.

test_unit_* tests validate Python logic only (no cellpose needed).
test_integration_training runs an actual 1-epoch cellpose training on CPU.
"""

import os
import shutil
import tempfile
from pathlib import Path

import pytest
import yaml

# Import from train.py at repo root
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from train import (
    parse_args,
    load_config,
    get_param,
    prepare_cellpose_dirs,
    generate_model_id,
    write_results,
    save_model,
    find_trained_model,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_data(tmp_path):
    """Create a minimal data directory structure with dummy TIF files."""
    for split in ["train", "validation", "test"]:
        (tmp_path / "in" / split).mkdir(parents=True)
        (tmp_path / "gt" / split).mkdir(parents=True)
        for i in range(3):
            (tmp_path / "in" / split / f"img_{i}.tif").write_bytes(b"fake")
            (tmp_path / "gt" / split / f"img_{i}.tif").write_bytes(b"fake")
    (tmp_path / "out").mkdir()

    config = {
        "training": {
            "pretrained_model": "cyto3",
            "n_epochs": 1,
            "learning_rate": 0.001,
            "model_name": "unit_test_model",
            "channels": [0, 0],
            "diameter": 30,
        },
        "metadata": {
            "source_datasets": [1, 2],
            "trained_by": "test_user",
            "workflow_name": "W_Segmentation-Cellpose4",
        },
    }
    with open(tmp_path / "config.yaml", "w") as f:
        yaml.dump(config, f)

    return tmp_path


@pytest.fixture
def real_test_data():
    """Path to the real test training data shipped with the repo."""
    data_dir = os.path.join(
        os.path.dirname(__file__), "..", "test_training_data", "data"
    )
    if not os.path.isdir(data_dir):
        pytest.skip("test_training_data not found")
    return data_dir


# ---------------------------------------------------------------------------
# Unit tests (no cellpose needed)
# ---------------------------------------------------------------------------

class TestParseArgs:
    def test_required_args(self):
        args = parse_args(["--infolder", "/in", "--outfolder", "/out",
                           "--gtfolder", "/gt"])
        assert args.infolder == "/in"
        assert args.outfolder == "/out"
        assert args.gtfolder == "/gt"

    def test_biaflows_args_accepted(self):
        args = parse_args(["--infolder", "/in", "--outfolder", "/out",
                           "--gtfolder", "/gt", "--local", "-nmc"])
        assert args.local is True

    def test_unknown_args_ignored(self):
        args = parse_args(["--infolder", "/in", "--outfolder", "/out",
                           "--gtfolder", "/gt", "--unknown", "value"])
        assert args.infolder == "/in"


class TestLoadConfig:
    def test_loads_config_from_parent_of_infolder(self, mock_data):
        config = load_config(str(mock_data / "in"))
        assert config["training"]["model_name"] == "unit_test_model"

    def test_returns_empty_dict_when_no_config(self, tmp_path):
        (tmp_path / "in").mkdir()
        config = load_config(str(tmp_path / "in"))
        assert config == {}


class TestGetParam:
    def test_config_takes_priority(self):
        config = {"training": {"n_epochs": 50}}
        os.environ["N_EPOCHS"] = "200"
        try:
            result = get_param(config, "training", "n_epochs",
                               "N_EPOCHS", 100, cast=int)
            assert result == 50
        finally:
            del os.environ["N_EPOCHS"]

    def test_env_var_fallback(self):
        os.environ["N_EPOCHS"] = "200"
        try:
            result = get_param({}, "training", "n_epochs",
                               "N_EPOCHS", 100, cast=int)
            assert result == 200
        finally:
            del os.environ["N_EPOCHS"]

    def test_default_fallback(self):
        result = get_param({}, "training", "n_epochs",
                           "NONEXISTENT_VAR", 100, cast=int)
        assert result == 100

    def test_no_cast(self):
        config = {"training": {"model": "cyto3"}}
        result = get_param(config, "training", "model", "MODEL", "default")
        assert result == "cyto3"


class TestPrepareCellposeDirs:
    def test_creates_symlinks_with_masks_suffix(self, mock_data):
        out_dir = prepare_cellpose_dirs(
            str(mock_data / "in"), str(mock_data / "gt"), "train"
        )
        try:
            assert out_dir is not None
            assert os.path.exists(os.path.join(out_dir, "img_0.tif"))
            assert os.path.exists(os.path.join(out_dir, "img_0_masks.tif"))
        finally:
            if out_dir and os.path.isdir(out_dir):
                shutil.rmtree(out_dir)

    def test_returns_none_for_missing_split(self, mock_data):
        result = prepare_cellpose_dirs(
            str(mock_data / "in"), str(mock_data / "gt"), "nonexistent"
        )
        assert result is None

    def test_returns_none_for_empty_dir(self, tmp_path):
        (tmp_path / "in" / "train").mkdir(parents=True)
        (tmp_path / "gt" / "train").mkdir(parents=True)
        result = prepare_cellpose_dirs(
            str(tmp_path / "in"), str(tmp_path / "gt"), "train"
        )
        assert result is None


class TestGenerateModelId:
    def test_uses_model_name_when_provided(self):
        config = {"training": {"model_name": "my_model"}}
        model_id, model_name = generate_model_id(config)
        assert model_name == "my_model"
        assert model_id.startswith("my_model_")
        assert len(model_id) > len("my_model_")  # has timestamp

    def test_fallback_without_model_name(self):
        config = {
            "metadata": {
                "workflow_name": "cellpose",
                "trained_by": "user1",
            }
        }
        model_id, model_name = generate_model_id(config)
        assert "cellpose" in model_id
        assert "user1" in model_id

    def test_fallback_with_empty_config(self):
        model_id, model_name = generate_model_id({})
        assert "cellpose" in model_id
        assert "unknown" in model_id


class TestWriteResults:
    def test_writes_yaml_with_expected_keys(self, tmp_path):
        config = {
            "training": {"pretrained_model": "cyto3", "n_epochs": 10},
            "metadata": {"source_datasets": [1], "trained_by": "user1"},
        }
        results = write_results(
            str(tmp_path), "model_123", "my_model", config
        )
        assert results["model_id"] == "model_123"
        assert results["model_name"] == "my_model"
        assert results["pretrained_model"] == "cyto3"

        results_path = tmp_path / "training_results.yaml"
        assert results_path.exists()
        with open(results_path) as f:
            loaded = yaml.safe_load(f)
        assert loaded["model_id"] == "model_123"


class TestSaveModel:
    def test_creates_zip_and_persistent_copy(self, tmp_path):
        # Create a fake model file
        model_file = tmp_path / "fake_model.pth"
        model_file.write_bytes(b"model data")

        outfolder = tmp_path / "out"
        outfolder.mkdir()

        zip_path = save_model(str(model_file), "test_model_123", str(outfolder))
        assert os.path.exists(zip_path)
        assert zip_path.endswith(".zip")

        # Check persistent copy
        persist_dir = "/tmp/models/test_model_123"
        try:
            assert os.path.isdir(persist_dir)
            assert os.path.exists(
                os.path.join(persist_dir, "fake_model.pth")
            )
        finally:
            shutil.rmtree(persist_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Integration test (requires cellpose installed)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestIntegrationTraining:
    def test_full_training_pipeline(self, real_test_data):
        """Run actual cellpose training with 1 epoch on test data."""
        with tempfile.TemporaryDirectory() as outdir:
            from train import main

            main([
                "--infolder", os.path.join(real_test_data, "in"),
                "--outfolder", outdir,
                "--gtfolder", os.path.join(real_test_data, "gt"),
                "--local", "-nmc",
            ])

            # Check training_results.yaml was created
            results_path = os.path.join(outdir, "training_results.yaml")
            assert os.path.exists(results_path), \
                "training_results.yaml not found"
            with open(results_path) as f:
                results = yaml.safe_load(f)
            assert results["model_id"].startswith("ci_test_model_")
            assert results["model_name"] == "ci_test_model"
            assert results["n_epochs"] == 1

            # Check model zip was created
            zips = [f for f in os.listdir(outdir) if f.endswith(".zip")]
            assert len(zips) == 1, f"Expected 1 zip, found: {zips}"

            # Clean up persistent model dir
            model_id = results["model_id"]
            shutil.rmtree(f"/tmp/models/{model_id}", ignore_errors=True)
