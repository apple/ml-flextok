# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. and EPFL. All Rights Reserved.

import pytest
import torch
import numpy as np
from typing import Dict, Any, List
from unittest.mock import MagicMock, patch


@pytest.fixture
def device():
    """Fixture providing the device to use for testing."""
    return torch.device("cpu")  # Use CPU for testing to avoid GPU dependencies


@pytest.fixture
def sample_image_tensor():
    """Fixture providing a sample image tensor for testing."""
    # Create a small RGB image tensor [B=1, C=3, H=64, W=64]
    return torch.randn(1, 3, 64, 64)


@pytest.fixture
def sample_image_list():
    """Fixture providing a list of sample image tensors."""
    # Create list of small RGB image tensors [1, 3, 64, 64] each
    return [torch.randn(1, 3, 64, 64) for _ in range(2)]


@pytest.fixture
def sample_vae_latents():
    """Fixture providing sample VAE latents."""
    # VAE typically downsamples by factor of 8, so 64x64 -> 8x8
    return torch.randn(1, 16, 8, 8)


@pytest.fixture
def sample_token_ids():
    """Fixture providing sample token ID sequences."""
    # Sample token sequences of varying lengths
    return [
        torch.randint(0, 1000, (1, 50)),  # 50 tokens
        torch.randint(0, 1000, (1, 75)),  # 75 tokens
    ]


@pytest.fixture
def sample_data_dict(sample_image_list):
    """Fixture providing a sample data dictionary."""
    return {
        "rgb": sample_image_list,
    }


@pytest.fixture
def mock_vae():
    """Fixture providing a mock VAE module."""
    vae = MagicMock()
    vae.images_read_key = "rgb"
    vae.images_reconst_write_key = "images_reconst"
    vae.downsample_factor = 8
    vae.latent_dim = 16
    vae.encode.return_value = {"vae_latents": torch.randn(1, 16, 8, 8)}
    vae.decode.return_value = {"images_reconst": [torch.randn(1, 3, 64, 64)]}
    return vae


@pytest.fixture
def mock_encoder():
    """Fixture providing a mock encoder module."""
    encoder = MagicMock()
    encoder.return_value = {"encoder_output": torch.randn(1, 128, 64)}
    encoder.init_weights_muP = MagicMock()
    # Mock the register module for max tokens
    encoder.module_dict = {
        "enc_register_module": MagicMock(n_max=100)
    }
    return encoder


@pytest.fixture
def mock_decoder():
    """Fixture providing a mock decoder module."""
    decoder = MagicMock()
    decoder.return_value = {"decoder_output": torch.randn(1, 16, 8, 8)}
    decoder.init_weights_muP = MagicMock()
    # Mock the nested dropout module
    decoder.module_dict = {
        "dec_nested_dropout": MagicMock(eval_keep_k_read_key="eval_keep_k")
    }
    return decoder


@pytest.fixture
def mock_regularizer():
    """Fixture providing a mock regularizer module."""
    regularizer = MagicMock()
    regularizer.tokens_write_key = "token_ids"
    regularizer.quants_write_key = "quantized_tokens"
    regularizer.return_value = {
        "token_ids": [torch.randint(0, 1000, (1, 50))],
        "quantized_tokens": [torch.randn(1, 50, 512)]
    }
    regularizer.indices_to_embedding.return_value = torch.randn(1, 50, 512)
    return regularizer


@pytest.fixture
def mock_flow_matching():
    """Fixture providing a mock flow matching noise module."""
    flow_matching = MagicMock()
    flow_matching.return_value = {
        "noised_latents": torch.randn(1, 16, 8, 8),
        "timesteps": torch.tensor([0.5]),
        "sigmas": torch.tensor([1.0]),
    }
    return flow_matching


@pytest.fixture
def mock_pipeline():
    """Fixture providing a mock pipeline module."""

    def pipeline_init(model):
        pipeline = MagicMock()
        pipeline.return_value = {"denoised_latents": torch.randn(1, 16, 8, 8)}
        return pipeline

    return pipeline_init


@pytest.fixture
def generator():
    """Fixture providing a torch.Generator for reproducible testing."""
    gen = torch.Generator()
    gen.manual_seed(42)
    return gen


@pytest.fixture(autouse=True)
def set_random_seed():
    """Automatically set random seeds for reproducible tests."""
    torch.manual_seed(42)
    np.random.seed(42)


# Utility functions for testing
def assert_tensor_shape(tensor: torch.Tensor, expected_shape: tuple, name: str = "tensor"):
    """Helper function to assert tensor shapes."""
    assert tensor.shape == expected_shape, f"{name} shape {tensor.shape} != expected {expected_shape}"


def assert_tensor_dtype(tensor: torch.Tensor, expected_dtype: torch.dtype, name: str = "tensor"):
    """Helper function to assert tensor dtypes."""
    assert tensor.dtype == expected_dtype, f"{name} dtype {tensor.dtype} != expected {expected_dtype}"


def assert_dict_keys(data_dict: Dict[str, Any], expected_keys: List[str], name: str = "dict"):
    """Helper function to assert dictionary contains expected keys."""
    missing_keys = set(expected_keys) - set(data_dict.keys())
    assert not missing_keys, f"{name} missing keys: {missing_keys}"
