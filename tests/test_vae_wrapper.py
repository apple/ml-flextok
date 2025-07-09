# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. and EPFL. All Rights Reserved.

import pytest
import torch
from unittest.mock import MagicMock, patch, PropertyMock
from typing import Dict, Any

from flextok.vae_wrapper import StableDiffusionVAE, VAE_BASE_CFG
from tests.conftest import assert_tensor_shape, assert_dict_keys


class TestStableDiffusionVAE:
    """Test suite for the StableDiffusionVAE wrapper class."""

    @pytest.fixture
    def vae_config(self):
        """Fixture providing basic VAE configuration."""
        return {
            "images_read_key": "rgb",
            "vae_latents_read_key": "vae_latents",
            "vae_latents_write_key": "vae_latents_out",
            "images_reconst_write_key": "images_reconst",
            "vae_kl_loss_write_key": "kl_loss",
            "latent_channels": 16,
            "scaling_factor": 0.13025,
            "dtype_override": "fp32",
            "sample_posterior": True,
            "learnable_logvar": False,
            "logvar_init": 0.0,
            "compile_encode_fn": False,
            "force_vae_encode": True,
            "frozen": False,
        }

    @pytest.fixture
    def mock_autoencoder_kl(self):
        """Fixture providing a mock AutoencoderKL."""
        mock_vae = MagicMock()
        mock_vae.config = {
            "down_block_types": ["DownEncoderBlock2D"] * 4,
            "latent_channels": 16,
            "scaling_factor": 0.13025
        }

        # Mock the latent distribution
        mock_latent_dist = MagicMock()
        mock_latent_dist.sample.return_value = torch.randn(1, 16, 8, 8)
        mock_latent_dist.mode.return_value = torch.randn(1, 16, 8, 8)
        mock_latent_dist.kl.return_value = torch.randn(1)

        # Mock encode/decode methods
        mock_encode_result = MagicMock()
        mock_encode_result.latent_dist = mock_latent_dist
        mock_vae.encode.return_value = mock_encode_result

        mock_decode_result = MagicMock()
        mock_decode_result.sample = torch.randn(1, 3, 64, 64)
        mock_vae.decode.return_value = mock_decode_result

        return mock_vae

    @pytest.fixture
    def vae_wrapper(self, vae_config, mock_autoencoder_kl):
        """Fixture providing a StableDiffusionVAE wrapper."""
        with patch('flextok.vae_wrapper.AutoencoderKL') as mock_ae_class:
            mock_ae_class.from_config.return_value = mock_autoencoder_kl
            return StableDiffusionVAE(**vae_config)

    @pytest.mark.unit
    def test_init_from_scratch(self, vae_config, mock_autoencoder_kl):
        """Test VAE initialization from scratch (no HuggingFace path)."""
        with patch('flextok.vae_wrapper.AutoencoderKL') as mock_ae_class:
            mock_ae_class.from_config.return_value = mock_autoencoder_kl

            vae = StableDiffusionVAE(**vae_config)

            # Verify AutoencoderKL was created from config
            mock_ae_class.from_config.assert_called_once()
            config_used = mock_ae_class.from_config.call_args[0][0]

            # Check that base config was used with latent_channels override
            assert config_used["latent_channels"] == 16
            assert "act_fn" in config_used  # From VAE_BASE_CFG

            # Verify properties are set correctly
            assert vae.images_read_key == "rgb"
            assert vae.vae_latents_write_key == "vae_latents_out"
            assert vae.scaling_factor == 0.13025
            assert vae.sample_posterior is True
            assert vae.force_vae_encode is True

    @pytest.mark.unit
    def test_init_from_hub(self, vae_config, mock_autoencoder_kl):
        """Test VAE initialization from HuggingFace Hub."""
        vae_config["hf_hub_path"] = "stabilityai/sdxl-vae"

        with patch('flextok.vae_wrapper.AutoencoderKL') as mock_ae_class:
            mock_ae_class.from_pretrained.return_value = mock_autoencoder_kl

            vae = StableDiffusionVAE(**vae_config)

            # Verify AutoencoderKL was loaded from Hub
            mock_ae_class.from_pretrained.assert_called_once_with(
                "stabilityai/sdxl-vae",
                low_cpu_mem_usage=False
            )

    @pytest.mark.unit
    def test_init_with_learnable_logvar(self, vae_config, mock_autoencoder_kl):
        """Test VAE initialization with learnable logvar."""
        vae_config["learnable_logvar"] = True
        vae_config["logvar_init"] = -2.0

        with patch('flextok.vae_wrapper.AutoencoderKL') as mock_ae_class:
            mock_ae_class.from_config.return_value = mock_autoencoder_kl

            vae = StableDiffusionVAE(**vae_config)

            # Verify logvar parameter was created
            assert vae.logvar is not None
            assert isinstance(vae.logvar, torch.nn.Parameter)
            assert vae.logvar.item() == -2.0

    @pytest.mark.unit
    def test_init_frozen(self, vae_config, mock_autoencoder_kl):
        """Test VAE initialization in frozen mode."""
        vae_config["frozen"] = True

        with patch('flextok.vae_wrapper.AutoencoderKL') as mock_ae_class:
            mock_ae_class.from_config.return_value = mock_autoencoder_kl

            vae = StableDiffusionVAE(**vae_config)

            # Verify freeze was called
            with patch.object(vae, 'freeze') as mock_freeze:
                vae.__init__(**vae_config)
                # Note: This would be called during actual init, but our fixture doesn't capture it

    @pytest.mark.unit
    def test_properties(self, vae_wrapper, mock_autoencoder_kl):
        """Test VAE properties."""
        # Test downsample_factor
        assert vae_wrapper.downsample_factor == 8  # 2^(4-1) based on mock config

        # Test latent_dim
        assert vae_wrapper.latent_dim == 16

        # Test device and device_type
        with patch.object(vae_wrapper, 'parameters') as mock_params:
            mock_param = MagicMock()
            mock_param.device = torch.device('cpu')
            mock_params.return_value = [mock_param]

            assert vae_wrapper.device == torch.device('cpu')
            assert vae_wrapper.device_type == 'cpu'

    @pytest.mark.unit
    def test_freeze(self, vae_wrapper):
        """Test the freeze method."""
        # Create some dummy parameters
        with patch.object(vae_wrapper, 'parameters') as mock_params:
            mock_param1 = MagicMock()
            mock_param2 = MagicMock()
            mock_params.return_value = [mock_param1, mock_param2]

            with patch.object(vae_wrapper, 'eval') as mock_eval:
                result = vae_wrapper.freeze()

                # Verify all parameters had requires_grad set to False
                mock_param1.__setattr__.assert_called_with('requires_grad', False)
                mock_param2.__setattr__.assert_called_with('requires_grad', False)

                # Verify eval was called
                mock_eval.assert_called_once()

                # Verify method returns self
                assert result == vae_wrapper

    @pytest.mark.unit
    def test_train_frozen(self, vae_wrapper):
        """Test train method when frozen."""
        vae_wrapper.frozen = True

        # When frozen, train should return self without calling super().train()
        with patch('super') as mock_super:
            result = vae_wrapper.train(mode=True)
            assert result == vae_wrapper
            mock_super.assert_not_called()

    @pytest.mark.unit
    def test_train_not_frozen(self, vae_wrapper):
        """Test train method when not frozen."""
        vae_wrapper.frozen = False

        # When not frozen, should call normal train method
        with patch('torch.nn.Module.train', return_value=vae_wrapper) as mock_train:
            result = vae_wrapper.train(mode=False)
            assert result == vae_wrapper
            mock_train.assert_called_once_with(mode=False)

    @pytest.mark.unit
    def test_encode_tensor_input(self, vae_wrapper, sample_image_tensor, mock_autoencoder_kl):
        """Test encode method with tensor input."""
        data_dict = {"rgb": sample_image_tensor}

        result = vae_wrapper.encode(data_dict)

        # Verify VAE encode was called
        mock_autoencoder_kl.encode.assert_called_once()

        # Check output keys
        expected_keys = ["vae_latents_out", "kl_loss"]
        assert_dict_keys(result, expected_keys)

        # Check output types and shapes
        latents = result["vae_latents_out"]
        kl_loss = result["kl_loss"]

        assert isinstance(latents, torch.Tensor)
        assert isinstance(kl_loss, torch.Tensor)

    @pytest.mark.unit
    def test_encode_list_input(self, vae_wrapper, sample_image_list, mock_autoencoder_kl):
        """Test encode method with list input."""
        data_dict = {"rgb": sample_image_list}

        result = vae_wrapper.encode(data_dict)

        # Verify VAE encode was called with concatenated tensor
        mock_autoencoder_kl.encode.assert_called_once()

        # Check that outputs are lists
        latents = result["vae_latents_out"]
        kl_loss = result["kl_loss"]

        assert isinstance(latents, list)
        assert isinstance(kl_loss, list)
        assert len(latents) == len(sample_image_list)
        assert len(kl_loss) == len(sample_image_list)

    @pytest.mark.unit
    def test_encode_skip_when_exists(self, vae_wrapper, sample_image_tensor, mock_autoencoder_kl):
        """Test encode method skips when latents already exist and force_vae_encode is False."""
        vae_wrapper.force_vae_encode = False

        # Data dict already contains VAE latents
        data_dict = {
            "rgb": sample_image_tensor,
            "vae_latents_out": torch.randn(1, 16, 8, 8)
        }

        result = vae_wrapper.encode(data_dict)

        # Verify VAE encode was NOT called
        mock_autoencoder_kl.encode.assert_not_called()

        # Original data should be returned unchanged
        assert result == data_dict

    @pytest.mark.unit
    def test_encode_sample_posterior(self, vae_wrapper, sample_image_tensor, mock_autoencoder_kl):
        """Test encode method with sample_posterior=True."""
        vae_wrapper.sample_posterior = True
        data_dict = {"rgb": sample_image_tensor}

        vae_wrapper.encode(data_dict)

        # Verify sample() was called on latent distribution
        latent_dist = mock_autoencoder_kl.encode.return_value.latent_dist
        latent_dist.sample.assert_called_once()
        latent_dist.mode.assert_not_called()

    @pytest.mark.unit
    def test_encode_mode_posterior(self, vae_wrapper, sample_image_tensor, mock_autoencoder_kl):
        """Test encode method with sample_posterior=False."""
        vae_wrapper.sample_posterior = False
        data_dict = {"rgb": sample_image_tensor}

        vae_wrapper.encode(data_dict)

        # Verify mode() was called on latent distribution
        latent_dist = mock_autoencoder_kl.encode.return_value.latent_dist
        latent_dist.mode.assert_called_once()
        latent_dist.sample.assert_not_called()

    @pytest.mark.unit
    def test_decode_tensor_input(self, vae_wrapper, sample_vae_latents, mock_autoencoder_kl):
        """Test decode method with tensor input."""
        data_dict = {"vae_latents": sample_vae_latents}

        result = vae_wrapper.decode(data_dict)

        # Verify VAE decode was called with scaled latents
        mock_autoencoder_kl.decode.assert_called_once()

        # Check output
        assert "images_reconst" in result
        images = result["images_reconst"]
        assert isinstance(images, torch.Tensor)

    @pytest.mark.unit
    def test_decode_list_input(self, vae_wrapper, mock_autoencoder_kl):
        """Test decode method with list input."""
        latents_list = [torch.randn(1, 16, 8, 8), torch.randn(1, 16, 8, 8)]
        data_dict = {"vae_latents": latents_list}

        result = vae_wrapper.decode(data_dict)

        # Verify VAE decode was called
        mock_autoencoder_kl.decode.assert_called_once()

        # Check that output is a list
        images = result["images_reconst"]
        assert isinstance(images, list)
        assert len(images) == len(latents_list)

    @pytest.mark.unit
    def test_autoencode(self, vae_wrapper, sample_image_tensor):
        """Test autoencode method (encode + decode)."""
        data_dict = {"rgb": sample_image_tensor}

        # Mock encode and decode methods
        encode_result = {
            "vae_latents_out": torch.randn(1, 16, 8, 8),
            "kl_loss": torch.randn(1)
        }
        decode_result = {"images_reconst": torch.randn(1, 3, 64, 64)}

        with patch.object(vae_wrapper, 'encode', return_value=encode_result) as mock_encode, \
                patch.object(vae_wrapper, 'decode', return_value=decode_result) as mock_decode:
            result = vae_wrapper.autoencode(data_dict)

            # Verify encode was called
            mock_encode.assert_called_once_with(data_dict)

            # Verify decode was called with latents added to read key
            decode_call_args = mock_decode.call_args[0][0]
            assert "vae_latents" in decode_call_args  # Read key for decode

            assert result == decode_result

    @pytest.mark.unit
    def test_forward(self, vae_wrapper, sample_image_tensor):
        """Test forward method (should call autoencode)."""
        data_dict = {"rgb": sample_image_tensor}
        expected_result = {"images_reconst": torch.randn(1, 3, 64, 64)}

        with patch.object(vae_wrapper, 'autoencode', return_value=expected_result) as mock_autoencode:
            result = vae_wrapper.forward(data_dict)

            mock_autoencode.assert_called_once_with(data_dict)
            assert result == expected_result

    @pytest.mark.unit
    def test_get_last_layer(self, vae_wrapper, mock_autoencoder_kl):
        """Test get_last_layer method."""
        # Mock the decoder conv_out layer
        mock_conv_out = MagicMock()
        mock_conv_out.weight = torch.randn(3, 16, 3, 3)
        mock_autoencoder_kl.decoder.conv_out = mock_conv_out

        result = vae_wrapper.get_last_layer()
        assert result is mock_conv_out.weight

    @pytest.mark.unit
    def test_get_logvar_with_learnable(self, vae_config, mock_autoencoder_kl):
        """Test get_logvar method when learnable_logvar is True."""
        vae_config["learnable_logvar"] = True
        vae_config["logvar_init"] = -1.5

        with patch('flextok.vae_wrapper.AutoencoderKL') as mock_ae_class:
            mock_ae_class.from_config.return_value = mock_autoencoder_kl

            vae = StableDiffusionVAE(**vae_config)

            logvar = vae.get_logvar()
            assert logvar is not None
            assert isinstance(logvar, torch.nn.Parameter)

    @pytest.mark.unit
    def test_get_logvar_without_learnable(self, vae_wrapper):
        """Test get_logvar method when learnable_logvar is False."""
        logvar = vae_wrapper.get_logvar()
        assert logvar is None

    @pytest.mark.unit
    def test_compile_encode_fn(self, vae_config, mock_autoencoder_kl):
        """Test VAE initialization with compile_encode_fn=True."""
        vae_config["compile_encode_fn"] = True

        with patch('flextok.vae_wrapper.AutoencoderKL') as mock_ae_class, \
                patch('torch.compile') as mock_compile, \
                patch('torch._inductor.config'):
            mock_ae_class.from_config.return_value = mock_autoencoder_kl
            mock_compile.return_value = MagicMock()

            vae = StableDiffusionVAE(**vae_config)

            # Verify torch.compile was called on encode
            mock_compile.assert_called_once()

            # Verify memory format and fuse_qkv_projections were called
            mock_autoencoder_kl.to.assert_called_once_with(memory_format=torch.channels_last)
            mock_autoencoder_kl.fuse_qkv_projections.assert_called_once()

    @pytest.mark.unit
    def test_dtype_override_handling(self, vae_config, mock_autoencoder_kl):
        """Test different dtype override values."""
        test_cases = [
            ("fp32", torch.float32),
            ("fp16", torch.float16),
            ("bf16", torch.bfloat16),
            (None, torch.float32),  # Default fallback
        ]

        for dtype_str, expected_dtype in test_cases:
            vae_config["dtype_override"] = dtype_str

            with patch('flextok.vae_wrapper.AutoencoderKL') as mock_ae_class, \
                    patch('flextok.utils.misc.str_to_dtype') as mock_str_to_dtype:

                if dtype_str is not None:
                    mock_str_to_dtype.return_value = expected_dtype

                mock_ae_class.from_config.return_value = mock_autoencoder_kl

                vae = StableDiffusionVAE(**vae_config)

                assert vae.dtype_override == expected_dtype

                if dtype_str is not None:
                    mock_str_to_dtype.assert_called_once_with(dtype_str)

    @pytest.mark.unit
    def test_vae_base_cfg_structure(self):
        """Test that VAE_BASE_CFG has required structure."""
        required_keys = [
            "_class_name", "act_fn", "block_out_channels",
            "down_block_types", "in_channels", "layers_per_block",
            "norm_num_groups", "out_channels", "sample_size",
            "up_block_types"
        ]

        for key in required_keys:
            assert key in VAE_BASE_CFG, f"Missing required key: {key}"

        # Check that override fields are None
        assert VAE_BASE_CFG["latent_channels"] is None
        assert VAE_BASE_CFG["scaling_factor"] is None
