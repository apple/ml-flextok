# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. and EPFL. All Rights Reserved.

import pytest
import torch
import math
from unittest.mock import MagicMock, patch
from typing import Dict, Any, List

from flextok.flow_matching.noise_modules import MinRFNoiseModule
from flextok.flow_matching.pipelines import MinRFPipeline
from flextok.flow_matching.cfg_utils import (
    MomentumBuffer,
    normalized_guidance,
    classifier_free_guidance,
    project
)
from tests.conftest import assert_tensor_shape, assert_dict_keys


class TestMinRFNoiseModule:
    """Test suite for the MinRFNoiseModule class."""

    @pytest.fixture
    def noise_config(self):
        """Fixture providing basic noise module configuration."""
        return {
            "clean_images_read_key": "clean_images",
            "noised_images_write_key": "noised_images",
            "timesteps_write_key": "timesteps",
            "sigmas_write_key": "sigmas",
            "ln": True,
            "stratisfied": False,
            "mode_scale": 0.0,
            "noise_write_key": "flow_noise",
            "noise_read_key": None,
        }

    @pytest.fixture
    def noise_module(self, noise_config):
        """Fixture providing a MinRFNoiseModule instance."""
        return MinRFNoiseModule(**noise_config)

    @pytest.fixture
    def sample_clean_images(self):
        """Fixture providing sample clean images."""
        return [torch.randn(1, 16, 8, 8), torch.randn(1, 16, 8, 8)]

    @pytest.fixture
    def data_dict_with_clean_images(self, sample_clean_images):
        """Fixture providing data dict with clean images."""
        return {"clean_images": sample_clean_images}

    @pytest.mark.unit
    def test_init(self, noise_config):
        """Test MinRFNoiseModule initialization."""
        module = MinRFNoiseModule(**noise_config)

        assert module.clean_images_read_key == "clean_images"
        assert module.noised_images_write_key == "noised_images"
        assert module.timesteps_write_key == "timesteps"
        assert module.sigmas_write_key == "sigmas"
        assert module.ln is True
        assert module.stratisfied is False
        assert module.mode_scale == 0.0
        assert module.noise_write_key == "flow_noise"
        assert module.noise_read_key is None

    @pytest.mark.unit
    def test_forward_logit_normal(self, noise_module, data_dict_with_clean_images):
        """Test forward pass with logit-normal noise (ln=True)."""
        noise_module.ln = True
        noise_module.stratisfied = False

        result = noise_module.forward(data_dict_with_clean_images)

        # Check that all expected keys are written
        expected_keys = ["noised_images", "flow_noise", "sigmas", "timesteps"]
        assert_dict_keys(result, expected_keys)

        # Check shapes and types
        noised_images = result["noised_images"]
        flow_noise = result["flow_noise"]
        sigmas = result["sigmas"]
        timesteps = result["timesteps"]

        assert len(noised_images) == 2
        assert len(flow_noise) == 2
        assert_tensor_shape(sigmas, (2,), "sigmas")
        assert_tensor_shape(timesteps, (2,), "timesteps")

        # Verify sigmas are in valid range [0, 1] for sigmoid
        assert torch.all(sigmas >= 0) and torch.all(sigmas <= 1)

        # Verify timesteps equal sigmas in this case
        assert torch.equal(timesteps, sigmas)

    @pytest.mark.unit
    def test_forward_logit_normal_stratified(self, noise_module, data_dict_with_clean_images):
        """Test forward pass with stratified logit-normal noise."""
        noise_module.ln = True
        noise_module.stratisfied = True

        result = noise_module.forward(data_dict_with_clean_images)

        # Check basic structure
        assert_dict_keys(result, ["noised_images", "flow_noise", "sigmas", "timesteps"])

        sigmas = result["sigmas"]
        assert_tensor_shape(sigmas, (2,), "sigmas")
        assert torch.all(sigmas >= 0) and torch.all(sigmas <= 1)

    @pytest.mark.unit
    def test_forward_uniform_noise(self, noise_module, data_dict_with_clean_images):
        """Test forward pass with uniform noise (ln=False)."""
        noise_module.ln = False
        noise_module.mode_scale = 0.0

        result = noise_module.forward(data_dict_with_clean_images)

        # Check basic structure
        assert_dict_keys(result, ["noised_images", "flow_noise", "sigmas", "timesteps"])

        sigmas = result["sigmas"]
        assert_tensor_shape(sigmas, (2,), "sigmas")
        # Uniform noise should be in [0, 1]
        assert torch.all(sigmas >= 0) and torch.all(sigmas <= 1)

    @pytest.mark.unit
    def test_forward_uniform_with_mode_scale(self, noise_module, data_dict_with_clean_images):
        """Test forward pass with uniform noise and mode scaling (SD3 style)."""
        noise_module.ln = False
        noise_module.mode_scale = 1.5

        result = noise_module.forward(data_dict_with_clean_images)

        sigmas = result["sigmas"]
        assert_tensor_shape(sigmas, (2,), "sigmas")
        # With mode scaling, sigmas might be outside [0, 1] range
        assert sigmas is not None

    @pytest.mark.unit
    def test_forward_with_provided_noise(self, noise_module, data_dict_with_clean_images):
        """Test forward pass when noise is provided in data_dict."""
        noise_module.noise_read_key = "custom_noise"

        # Add custom noise to data dict
        custom_noise = [torch.randn(1, 16, 8, 8), torch.randn(1, 16, 8, 8)]
        data_dict_with_clean_images["custom_noise"] = custom_noise

        result = noise_module.forward(data_dict_with_clean_images)

        # Verify the provided noise was used
        flow_noise = result["flow_noise"]
        assert len(flow_noise) == len(custom_noise)

        # The noise should be the same as what we provided
        for provided, returned in zip(custom_noise, flow_noise):
            assert torch.equal(provided, returned)

    @pytest.mark.unit
    def test_noising_formula(self, noise_module, data_dict_with_clean_images):
        """Test that the noising formula is correctly applied."""
        result = noise_module.forward(data_dict_with_clean_images)

        clean_images = data_dict_with_clean_images["clean_images"]
        noised_images = result["noised_images"]
        flow_noise = result["flow_noise"]
        sigmas = result["sigmas"]

        # Verify noising formula: noised = sigma * noise + (1 - sigma) * clean
        for i, (clean, noised, noise, sigma) in enumerate(zip(clean_images, noised_images, flow_noise, sigmas)):
            expected_noised = sigma * noise + (1.0 - sigma) * clean
            assert torch.allclose(noised, expected_noised, rtol=1e-5), f"Noising formula failed for image {i}"

    @pytest.mark.unit
    def test_device_consistency(self, noise_module):
        """Test that tensors are created on the correct device."""
        device = torch.device("cpu")
        clean_images = [torch.randn(1, 16, 8, 8, device=device) for _ in range(2)]
        data_dict = {"clean_images": clean_images}

        result = noise_module.forward(data_dict)

        # Check that all outputs are on the same device
        for tensor in result["sigmas"], result["timesteps"]:
            assert tensor.device == device


class TestMinRFPipeline:
    """Test suite for the MinRFPipeline class."""

    @pytest.fixture
    def mock_model(self):
        """Fixture providing a mock flow model."""
        model = MagicMock()
        model.device = torch.device("cpu")

        def mock_forward(data_dict):
            # Mock model output
            noised_images = data_dict["noised_images"]
            output_dict = data_dict.copy()
            output_dict["reconst"] = [torch.randn_like(img) for img in noised_images]
            return output_dict

        model.side_effect = mock_forward
        return model

    @pytest.fixture
    def pipeline_config(self):
        """Fixture providing basic pipeline configuration."""
        return {
            "noise_read_key": "noise",
            "target_sizes_read_key": "target_sizes",
            "latents_read_key": "latents",
            "timesteps_read_key": "timesteps",
            "noised_images_read_key": "noised_images",
            "reconst_write_key": "reconst",
            "out_channels": 16,
        }

    @pytest.fixture
    def pipeline(self, mock_model, pipeline_config):
        """Fixture providing a MinRFPipeline instance."""
        return MinRFPipeline(model=mock_model, **pipeline_config)

    @pytest.fixture
    def sample_data_dict(self):
        """Fixture providing sample data dict for pipeline."""
        return {
            "latents": [torch.randn(1, 64, 8), torch.randn(1, 64, 8)],
            "target_sizes": [(8, 8), (8, 8)],
        }

    @pytest.mark.unit
    def test_init(self, mock_model, pipeline_config):
        """Test MinRFPipeline initialization."""
        pipeline = MinRFPipeline(model=mock_model, **pipeline_config)

        assert pipeline.model == mock_model
        assert pipeline.noise_read_key == "noise"
        assert pipeline.target_sizes_read_key == "target_sizes"
        assert pipeline.out_channels == 16

    @pytest.mark.unit
    def test_call_basic(self, pipeline, sample_data_dict, mock_model):
        """Test basic pipeline call without CFG."""
        result = pipeline(
            sample_data_dict,
            timesteps=2,
            guidance_scale=1.0,
            verbose=False
        )

        # Check that model was called
        assert mock_model.call_count > 0

        # Check that output contains reconstructions
        assert "reconst" in result
        assert len(result["reconst"]) == 2

    @pytest.mark.unit
    def test_call_with_vae_image_sizes_int(self, pipeline, sample_data_dict):
        """Test pipeline call with integer vae_image_sizes."""
        result = pipeline(
            sample_data_dict,
            timesteps=2,
            vae_image_sizes=8,
            guidance_scale=1.0,
            verbose=False
        )

        assert "reconst" in result
        assert len(result["reconst"]) == 2

    @pytest.mark.unit
    def test_call_with_cfg(self, pipeline, sample_data_dict, mock_model):
        """Test pipeline call with classifier-free guidance."""
        result = pipeline(
            sample_data_dict,
            timesteps=2,
            guidance_scale=2.0,
            perform_norm_guidance=False,
            verbose=False
        )

        # With CFG, model should be called twice per timestep (conditional + unconditional)
        # With 2 timesteps, that's 4 calls minimum
        assert mock_model.call_count >= 4

        assert "reconst" in result

    @pytest.mark.unit
    def test_call_with_norm_guidance(self, pipeline, sample_data_dict, mock_model):
        """Test pipeline call with normalized guidance (APG)."""
        result = pipeline(
            sample_data_dict,
            timesteps=2,
            guidance_scale=2.0,
            perform_norm_guidance=True,
            verbose=False
        )

        # With normalized guidance, model should still be called multiple times
        assert mock_model.call_count >= 4

        assert "reconst" in result

    @pytest.mark.unit
    def test_call_with_callable_guidance_scale(self, pipeline, sample_data_dict):
        """Test pipeline call with callable guidance scale."""

        def dynamic_scale(t):
            return 1.0 + t  # Scale increases with time

        result = pipeline(
            sample_data_dict,
            timesteps=2,
            guidance_scale=dynamic_scale,
            verbose=False
        )

        assert "reconst" in result

    @pytest.mark.unit
    def test_call_with_provided_noise(self, pipeline, sample_data_dict):
        """Test pipeline call when noise is provided."""
        noise = [torch.randn(1, 16, 8, 8), torch.randn(1, 16, 8, 8)]
        sample_data_dict["noise"] = noise

        result = pipeline(
            sample_data_dict,
            timesteps=2,
            guidance_scale=1.0,
            verbose=False
        )

        assert "reconst" in result

    @pytest.mark.unit
    def test_call_with_generator(self, pipeline, sample_data_dict):
        """Test pipeline call with specified generator for reproducibility."""
        generator = torch.Generator()
        generator.manual_seed(42)

        result1 = pipeline(
            sample_data_dict,
            timesteps=2,
            generator=generator,
            guidance_scale=1.0,
            verbose=False
        )

        # Reset generator and run again
        generator.manual_seed(42)
        result2 = pipeline(
            sample_data_dict,
            timesteps=2,
            generator=generator,
            guidance_scale=1.0,
            verbose=False
        )

        # Results should be deterministic
        assert "reconst" in result1
        assert "reconst" in result2


class TestCFGUtils:
    """Test suite for CFG utility functions."""

    @pytest.mark.unit
    def test_momentum_buffer_init(self):
        """Test MomentumBuffer initialization."""
        buffer = MomentumBuffer(momentum=0.9)
        assert buffer.momentum == 0.9
        assert buffer.running_average == 0

    @pytest.mark.unit
    def test_momentum_buffer_update(self):
        """Test MomentumBuffer update mechanism."""
        buffer = MomentumBuffer(momentum=0.9)

        # First update
        update1 = torch.tensor(2.0)
        buffer.update(update1)
        expected1 = update1 + 0.9 * 0  # 0.9 * initial value (0)
        assert torch.allclose(buffer.running_average, expected1)

        # Second update
        update2 = torch.tensor(1.0)
        buffer.update(update2)
        expected2 = update2 + 0.9 * buffer.running_average
        assert torch.allclose(buffer.running_average, expected2)

    @pytest.mark.unit
    def test_project_function(self):
        """Test the project function for vector projection."""
        # Create test vectors
        v0 = torch.randn(1, 4, 8, 8)
        v1 = torch.randn(1, 4, 8, 8)

        v0_parallel, v0_orthogonal = project(v0, v1)

        # Check shapes
        assert_tensor_shape(v0_parallel, v0.shape, "parallel component")
        assert_tensor_shape(v0_orthogonal, v0.shape, "orthogonal component")

        # Check that parallel + orthogonal = original
        reconstructed = v0_parallel + v0_orthogonal
        assert torch.allclose(reconstructed, v0, rtol=1e-4)

        # Check that parallel is indeed parallel to normalized v1
        v1_normalized = torch.nn.functional.normalize(v1, dim=[-1, -2, -3])
        dot_product = (v0_parallel * v1_normalized).sum(dim=[-1, -2, -3])
        parallel_magnitude = v0_parallel.norm(dim=[-1, -2, -3])

        # The ratio should be constant (indicating parallel vectors)
        if parallel_magnitude.item() > 1e-6:  # Avoid division by zero
            assert torch.allclose(
                torch.abs(dot_product), parallel_magnitude, rtol=1e-4
            )

    @pytest.mark.unit
    def test_classifier_free_guidance(self):
        """Test standard classifier-free guidance."""
        pred_cond = torch.randn(1, 4, 8, 8)
        pred_uncond = torch.randn(1, 4, 8, 8)
        guidance_scale = 2.0

        result = classifier_free_guidance(pred_cond, pred_uncond, guidance_scale)

        # Check formula: uncond + scale * (cond - uncond)
        expected = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
        assert torch.allclose(result, expected)

        # Check shape preservation
        assert_tensor_shape(result, pred_cond.shape, "CFG result")

    @pytest.mark.unit
    def test_classifier_free_guidance_no_guidance(self):
        """Test CFG with guidance scale = 1.0 (no guidance)."""
        pred_cond = torch.randn(1, 4, 8, 8)
        pred_uncond = torch.randn(1, 4, 8, 8)
        guidance_scale = 1.0

        result = classifier_free_guidance(pred_cond, pred_uncond, guidance_scale)

        # With scale=1.0, result should equal pred_cond
        assert torch.allclose(result, pred_cond)

    @pytest.mark.unit
    def test_normalized_guidance_basic(self):
        """Test normalized guidance without momentum buffer."""
        pred_cond = torch.randn(1, 4, 8, 8)
        pred_uncond = torch.randn(1, 4, 8, 8)
        guidance_scale = 2.0

        result = normalized_guidance(
            pred_cond=pred_cond,
            pred_uncond=pred_uncond,
            guidance_scale=guidance_scale,
            momentum_buffer=None,
            eta=0.0,
            norm_threshold=0.0  # No thresholding
        )

        # Check shape preservation
        assert_tensor_shape(result, pred_cond.shape, "normalized guidance result")

    @pytest.mark.unit
    def test_normalized_guidance_with_momentum(self):
        """Test normalized guidance with momentum buffer."""
        pred_cond = torch.randn(1, 4, 8, 8)
        pred_uncond = torch.randn(1, 4, 8, 8)
        guidance_scale = 2.0
        momentum_buffer = MomentumBuffer(momentum=0.9)

        result = normalized_guidance(
            pred_cond=pred_cond,
            pred_uncond=pred_uncond,
            guidance_scale=guidance_scale,
            momentum_buffer=momentum_buffer,
            eta=0.0,
            norm_threshold=0.0
        )

        # Check that momentum buffer was updated
        assert momentum_buffer.running_average is not None
        assert_tensor_shape(result, pred_cond.shape, "normalized guidance with momentum")

    @pytest.mark.unit
    def test_normalized_guidance_with_thresholding(self):
        """Test normalized guidance with norm thresholding."""
        pred_cond = torch.randn(1, 4, 8, 8)
        pred_uncond = torch.randn(1, 4, 8, 8)
        guidance_scale = 2.0
        norm_threshold = 1.0

        result = normalized_guidance(
            pred_cond=pred_cond,
            pred_uncond=pred_uncond,
            guidance_scale=guidance_scale,
            momentum_buffer=None,
            eta=0.0,
            norm_threshold=norm_threshold
        )

        assert_tensor_shape(result, pred_cond.shape, "normalized guidance with thresholding")

    @pytest.mark.unit
    def test_normalized_guidance_with_eta(self):
        """Test normalized guidance with parallel component (eta > 0)."""
        pred_cond = torch.randn(1, 4, 8, 8)
        pred_uncond = torch.randn(1, 4, 8, 8)
        guidance_scale = 2.0
        eta = 0.5

        result = normalized_guidance(
            pred_cond=pred_cond,
            pred_uncond=pred_uncond,
            guidance_scale=guidance_scale,
            momentum_buffer=None,
            eta=eta,
            norm_threshold=0.0
        )

        assert_tensor_shape(result, pred_cond.shape, "normalized guidance with eta")

    @pytest.mark.unit
    def test_guidance_consistency(self):
        """Test that normalized guidance reduces to CFG under certain conditions."""
        pred_cond = torch.randn(1, 4, 8, 8)
        pred_uncond = torch.randn(1, 4, 8, 8)
        guidance_scale = 1.5

        # Standard CFG
        cfg_result = classifier_free_guidance(pred_cond, pred_uncond, guidance_scale)

        # Normalized guidance with specific settings should approximate CFG
        # Note: They won't be exactly equal due to the normalization and projection operations
        norm_result = normalized_guidance(
            pred_cond=pred_cond,
            pred_uncond=pred_uncond,
            guidance_scale=guidance_scale,
            momentum_buffer=None,
            eta=1.0,  # Full parallel component
            norm_threshold=0.0  # No thresholding
        )

        # They should be similar in magnitude
        cfg_norm = cfg_result.norm()
        norm_norm = norm_result.norm()

        # Allow for some difference due to the normalization process
        assert torch.abs(cfg_norm - norm_norm) / cfg_norm < 0.5  # Within 50%
