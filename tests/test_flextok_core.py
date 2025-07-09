# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. and EPFL. All Rights Reserved.

import pytest
import torch
from unittest.mock import MagicMock, patch
from typing import Dict, Any

from flextok.flextok_wrapper import FlexTok, FlexTokFromHub
from tests.conftest import assert_tensor_shape, assert_dict_keys


class TestFlexTok:
    """Test suite for the main FlexTok class."""

    @pytest.fixture
    def flextok_model(self, mock_vae, mock_encoder, mock_decoder, mock_regularizer,
                      mock_flow_matching, mock_pipeline):
        """Fixture providing a FlexTok model with mocked components."""
        return FlexTok(
            vae=mock_vae,
            encoder=mock_encoder,
            decoder=mock_decoder,
            regularizer=mock_regularizer,
            flow_matching_noise_module=mock_flow_matching,
            pipeline=mock_pipeline,
        )

    def test_init(self, flextok_model, mock_vae, mock_encoder, mock_decoder,
                  mock_regularizer, mock_flow_matching, mock_pipeline):
        """Test FlexTok initialization."""
        assert flextok_model.vae == mock_vae
        assert flextok_model.encoder == mock_encoder
        assert flextok_model.decoder == mock_decoder
        assert flextok_model.regularizer == mock_regularizer
        assert flextok_model.flow_matching_noise_module == mock_flow_matching

        # Test that keys are properly set from regularizer and VAE
        assert flextok_model.token_write_key == mock_regularizer.tokens_write_key
        assert flextok_model.quants_write_key == mock_regularizer.quants_write_key
        assert flextok_model.image_write_key == mock_vae.images_reconst_write_key

    def test_properties(self, flextok_model, mock_vae):
        """Test FlexTok properties."""
        assert flextok_model.downsample_factor == mock_vae.downsample_factor

        # Test device property (should return device of first parameter)
        with patch.object(flextok_model, 'parameters') as mock_params:
            mock_param = MagicMock()
            mock_param.device = torch.device('cpu')
            mock_params.return_value = [mock_param]
            assert flextok_model.device == torch.device('cpu')

    def test_init_weights_muP(self, flextok_model, mock_encoder, mock_decoder):
        """Test muP initialization."""
        flextok_model.init_weights_muP()
        mock_encoder.init_weights_muP.assert_called_once()
        mock_decoder.init_weights_muP.assert_called_once()

    @pytest.mark.unit
    def test_encode(self, flextok_model, sample_data_dict, mock_vae, mock_encoder, mock_regularizer):
        """Test the encode method."""
        # Mock the components to return expected data
        mock_vae.encode.return_value = {"vae_latents": torch.randn(1, 16, 8, 8)}
        mock_encoder.return_value = {"encoder_output": torch.randn(1, 128, 64)}
        mock_regularizer.return_value = {
            "token_ids": [torch.randint(0, 1000, (1, 50))],
            "quantized_tokens": [torch.randn(1, 50, 512)]
        }

        result = flextok_model.encode(sample_data_dict)

        # Verify the pipeline was called correctly
        mock_vae.encode.assert_called_once_with(sample_data_dict)
        mock_encoder.assert_called_once()
        mock_regularizer.assert_called_once()

        # Check that result contains expected keys
        assert_dict_keys(result, ["token_ids", "quantized_tokens"])

    @pytest.mark.unit
    def test_decode(self, flextok_model, mock_pipeline, mock_vae):
        """Test the decode method."""
        # Prepare test data
        data_dict = {
            "quantized_tokens": [torch.randn(1, 50, 512)],
        }

        # Mock pipeline and VAE returns
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.return_value = {"denoised_latents": torch.randn(1, 16, 8, 8)}
        flextok_model.pipeline = mock_pipeline_instance

        mock_vae.decode.return_value = {"images_reconst": [torch.randn(1, 3, 64, 64)]}

        result = flextok_model.decode(
            data_dict,
            timesteps=10,
            generator=None,
            vae_image_sizes=32,
            verbose=False,
            guidance_scale=1.0,
            perform_norm_guidance=False
        )

        # Verify pipeline was called with correct arguments
        mock_pipeline_instance.assert_called_once_with(
            data_dict,
            generator=None,
            timesteps=10,
            vae_image_sizes=32,
            verbose=False,
            guidance_scale=1.0,
            perform_norm_guidance=False
        )

        # Verify VAE decode was called
        mock_vae.decode.assert_called_once()

    @pytest.mark.unit
    def test_autoencode(self, flextok_model, sample_data_dict):
        """Test the autoencode method (encode + decode)."""
        # Mock encode and decode methods
        encoded_result = {
            "token_ids": [torch.randint(0, 1000, (1, 50))],
            "quantized_tokens": [torch.randn(1, 50, 512)]
        }
        decoded_result = {"images_reconst": [torch.randn(1, 3, 64, 64)]}

        with patch.object(flextok_model, 'encode', return_value=encoded_result) as mock_encode, \
                patch.object(flextok_model, 'decode', return_value=decoded_result) as mock_decode:
            result = flextok_model.autoencode(
                sample_data_dict,
                timesteps=10,
                guidance_scale=1.0
            )

            # Verify encode was called with original data
            mock_encode.assert_called_once_with(sample_data_dict)

            # Verify decode was called with encoded result and parameters
            mock_decode.assert_called_once_with(
                encoded_result,
                timesteps=10,
                generator=None,
                vae_image_sizes=None,
                verbose=True,
                guidance_scale=1.0,
                perform_norm_guidance=False
            )

    @pytest.mark.unit
    def test_tokenize(self, flextok_model, sample_image_tensor, mock_vae):
        """Test the tokenize method."""
        # Mock the encode method to return token IDs
        token_ids = [torch.randint(0, 1000, (1, 50)), torch.randint(0, 1000, (1, 75))]

        with patch.object(flextok_model, 'encode') as mock_encode:
            mock_encode.return_value = {flextok_model.token_write_key: token_ids}

            result = flextok_model.tokenize(sample_image_tensor)

            # Verify encode was called with properly formatted data dict
            expected_data_dict = {mock_vae.images_read_key: [sample_image_tensor]}
            mock_encode.assert_called_once_with(expected_data_dict)

            # Verify result is the token IDs list
            assert result == token_ids

    @pytest.mark.unit
    def test_get_padded_token_seq(self, flextok_model):
        """Test the _get_padded_token_seq method."""
        # Test padding a shorter sequence
        token_ids = torch.randint(0, 1000, (1, 30))
        max_seq_len = 50

        padded = flextok_model._get_padded_token_seq(token_ids, max_seq_len)

        assert_tensor_shape(padded, (1, 50), "padded tokens")

        # Verify original tokens are preserved
        assert torch.equal(padded[:, :30], token_ids)

        # Verify padding is zeros
        assert torch.all(padded[:, 30:] == 0)

    @pytest.mark.unit
    def test_prepare_data_dict_for_detokenization(self, flextok_model, sample_token_ids,
                                                  mock_encoder, mock_regularizer):
        """Test the _prepare_data_dict_for_detokenization method."""
        # Set up mocks
        mock_encoder.module_dict["enc_register_module"].n_max = 100

        # Mock regularizer's indices_to_embedding
        def mock_indices_to_embedding(token_ids):
            return torch.randn(1, token_ids.shape[1], 512)

        mock_regularizer.indices_to_embedding = mock_indices_to_embedding

        result = flextok_model._prepare_data_dict_for_detokenization(sample_token_ids)

        # Check that result contains expected keys
        expected_keys = [
            flextok_model.quants_write_key,
            mock_encoder.module_dict["dec_nested_dropout"].eval_keep_k_read_key
        ]
        assert_dict_keys(result, expected_keys)

        # Check that token lengths are preserved
        token_lens = result[mock_encoder.module_dict["dec_nested_dropout"].eval_keep_k_read_key]
        assert token_lens == [50, 75]  # Original lengths from sample_token_ids

    @pytest.mark.unit
    def test_detokenize(self, flextok_model, sample_token_ids, mock_vae):
        """Test the detokenize method."""
        # Mock the helper methods and decode
        mock_data_dict = {"prepared": "data"}
        decoded_images = [torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64)]

        with patch.object(flextok_model, '_prepare_data_dict_for_detokenization',
                          return_value=mock_data_dict) as mock_prepare, \
                patch.object(flextok_model, 'decode') as mock_decode:
            mock_decode.return_value = {mock_vae.images_reconst_write_key: decoded_images}

            result = flextok_model.detokenize(
                sample_token_ids,
                vae_image_sizes=32,
                timesteps=10
            )

            # Verify preparation was called with token IDs
            mock_prepare.assert_called_once_with(token_ids_list=sample_token_ids)

            # Verify decode was called with prepared data and parameters
            mock_decode.assert_called_once_with(
                data_dict=mock_data_dict,
                vae_image_sizes=32,
                timesteps=10
            )

            # Verify result is concatenated images
            expected_result = torch.cat(decoded_images, dim=0)
            assert torch.equal(result, expected_result)

    @pytest.mark.unit
    def test_forward(self, flextok_model, sample_data_dict, mock_flow_matching, mock_decoder):
        """Test the forward method (training pipeline)."""
        # Mock encode method
        encoded_data = {
            "token_ids": [torch.randint(0, 1000, (1, 50))],
            "quantized_tokens": [torch.randn(1, 50, 512)]
        }

        # Mock flow matching to add noise-related data
        noised_data = encoded_data.copy()
        noised_data.update({
            "noised_latents": torch.randn(1, 16, 8, 8),
            "timesteps": torch.tensor([0.5]),
            "sigmas": torch.tensor([1.0]),
        })

        # Mock decoder output
        decoder_output = {"decoder_output": torch.randn(1, 16, 8, 8)}

        with patch.object(flextok_model, 'encode', return_value=encoded_data) as mock_encode:
            mock_flow_matching.return_value = noised_data
            mock_decoder.return_value = decoder_output

            result = flextok_model.forward(sample_data_dict)

            # Verify the full pipeline was executed
            mock_encode.assert_called_once_with(sample_data_dict)
            mock_flow_matching.assert_called_once_with(encoded_data)
            mock_decoder.assert_called_once_with(noised_data)

            assert result == decoder_output


class TestFlexTokFromHub:
    """Test suite for the FlexTokFromHub class."""

    @pytest.mark.unit
    def test_init_with_config(self):
        """Test FlexTokFromHub initialization with config."""
        # Mock config
        mock_config = {
            "vae": {"_target_": "flextok.vae_wrapper.StableDiffusionVAE"},
            "encoder": {"_target_": "mock.encoder"},
            "decoder": {"_target_": "mock.decoder"},
            "regularizer": {"_target_": "mock.regularizer"},
            "flow_matching_noise_module": {"_target_": "mock.flow_matching"},
            "pipeline": {"_target_": "mock.pipeline"},
        }

        # Mock the instantiate function and sanitize config
        with patch('flextok.flextok_wrapper.instantiate') as mock_instantiate, \
                patch('flextok.flextok_wrapper._sanitize_hydra_config') as mock_sanitize:
            # Mock instantiate to return mock objects
            mock_components = {
                "vae": MagicMock(),
                "encoder": MagicMock(),
                "decoder": MagicMock(),
                "regularizer": MagicMock(),
                "flow_matching_noise_module": MagicMock(),
                "pipeline": MagicMock(),
            }
            mock_instantiate.side_effect = lambda config: mock_components[config["_target_"].split(".")[-1]]

            # Create FlexTokFromHub instance
            model = FlexTokFromHub(mock_config)

            # Verify config was sanitized
            mock_sanitize.assert_called_once()

            # Verify all components were instantiated
            assert mock_instantiate.call_count == 6

    @pytest.mark.unit
    def test_inheritance(self):
        """Test that FlexTokFromHub properly inherits from both FlexTok and PyTorchModelHubMixin."""
        from flextok.flextok_wrapper import FlexTok
        from huggingface_hub import PyTorchModelHubMixin

        assert issubclass(FlexTokFromHub, FlexTok)
        assert issubclass(FlexTokFromHub, PyTorchModelHubMixin)
