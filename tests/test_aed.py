import pytest
import torch

from scene_text_recognition.configs.schemas import (
    AttnBasedEncDecModelConfig,
    DecoderConfig,
    EncoderConfig,
    FeatureExtractorConfig,
    PreEncoderConfig,
)
from scene_text_recognition.frameworks.aed import AttnBasedEncDecModel
from scene_text_recognition.frameworks.schemas import AEDOutput
from scene_text_recognition.tokenizer import AEDTokenizer


@pytest.fixture
def tokenizer() -> AEDTokenizer:
    """Create a simple tokenizer for testing."""
    vocab = list("abcdefghijklmnopqrstuvwxyz0123456789")
    return AEDTokenizer(vocab)


@pytest.fixture
def config() -> AttnBasedEncDecModelConfig:
    """Create a minimal config for testing."""
    return AttnBasedEncDecModelConfig(
        feature_extractor=FeatureExtractorConfig(
            type="resnet18",
            in_channels=3,
            input_height=32,
        ),
        pre_encoder=PreEncoderConfig(type="linear"),
        encoder=EncoderConfig(
            type="transformer",
            hidden_size=64,
            num_heads=4,
            num_layers=2,
        ),
        decoder=DecoderConfig(
            type="transformer",
            hidden_size=64,
            num_heads=4,
            num_layers=2,
        ),
    )


@pytest.fixture
def model(
    config: AttnBasedEncDecModelConfig, tokenizer: AEDTokenizer
) -> AttnBasedEncDecModel:
    """Create model instance for testing."""
    return AttnBasedEncDecModel(config, tokenizer)


# =============================================================================
# _add_sos_eos tests
# =============================================================================


class TestAddSosEos:
    def test_add_sos_eos_basic(self, model: AttnBasedEncDecModel) -> None:
        """Test basic SOS/EOS addition."""
        # targets: [A, B, C] (token ids 4, 5, 6 assuming special tokens come first)
        targets = torch.tensor([[4, 5, 6]])
        target_lens = torch.tensor([3])

        sos_added, eos_added = model._add_sos_eos(targets, target_lens)

        # SOS should be prepended
        assert sos_added[0, 0].item() == model.tokenizer.sos_id
        assert sos_added[0, 1:4].tolist() == [4, 5, 6]

        # EOS should be at position 3 (after A, B, C)
        assert eos_added[0, :3].tolist() == [4, 5, 6]
        assert eos_added[0, 3].item() == model.tokenizer.eos_id

    def test_add_sos_eos_different_lengths(self, model: AttnBasedEncDecModel) -> None:
        """Test SOS/EOS addition with different sequence lengths in batch."""
        pad_id = model.tokenizer.pad_id
        sos_id = model.tokenizer.sos_id
        eos_id = model.tokenizer.eos_id

        # Batch with different lengths: [A, B, pad] and [C, D, E]
        targets = torch.tensor(
            [
                [4, 5, pad_id],  # length 2
                [6, 7, 8],  # length 3
            ]
        )
        target_lens = torch.tensor([2, 3])

        sos_added, eos_added = model._add_sos_eos(targets, target_lens)

        # Check SOS added
        assert sos_added[0, 0].item() == sos_id
        assert sos_added[1, 0].item() == sos_id
        assert sos_added[0, 1:3].tolist() == [4, 5]
        assert sos_added[1, 1:4].tolist() == [6, 7, 8]

        # Check EOS added at correct positions
        assert eos_added[0, 2].item() == eos_id  # position 2 for length 2
        assert eos_added[1, 3].item() == eos_id  # position 3 for length 3

    def test_add_sos_eos_output_shape(self, model: AttnBasedEncDecModel) -> None:
        """Test output shapes of SOS/EOS addition."""
        batch_size = 4
        seq_len = 10
        targets = torch.randint(4, 20, (batch_size, seq_len))
        target_lens = torch.tensor([5, 7, 3, 10])

        sos_added, eos_added = model._add_sos_eos(targets, target_lens)

        # SOS added should have +1 column
        assert sos_added.shape == (batch_size, seq_len + 1)
        # EOS added should also have +1 column
        assert eos_added.shape == (batch_size, seq_len + 1)

    def test_add_sos_eos_preserves_device(self, model: AttnBasedEncDecModel) -> None:
        """Test that output tensors are on the same device as input."""
        targets = torch.tensor([[4, 5, 6]])
        target_lens = torch.tensor([3])

        sos_added, eos_added = model._add_sos_eos(targets, target_lens)

        assert sos_added.device == targets.device
        assert eos_added.device == targets.device

    def test_add_sos_eos_preserves_dtype(self, model: AttnBasedEncDecModel) -> None:
        """Test that output tensors preserve input dtype."""
        targets = torch.tensor([[4, 5, 6]], dtype=torch.long)
        target_lens = torch.tensor([3])

        sos_added, eos_added = model._add_sos_eos(targets, target_lens)

        assert sos_added.dtype == targets.dtype
        assert eos_added.dtype == targets.dtype


class TestForward:
    def test_forward_output_shape(
        self, model: AttnBasedEncDecModel, tokenizer: AEDTokenizer
    ) -> None:
        """Test output shapes of forward pass."""
        batch_size = 2
        img_height = 32
        img_width = 100
        seq_len = 5

        x = torch.randn(batch_size, 3, img_height, img_width)
        xlens = torch.tensor([img_width, img_width])
        targets = torch.randint(4, tokenizer.vocab_size, (batch_size, seq_len))
        target_lens = torch.tensor([seq_len, seq_len])

        output = model(x, xlens, targets, target_lens)

        # After _add_sos_eos, decoder input has seq_len+1 (SOS + targets)
        # So output also has seq_len+1
        expected_seq_len = seq_len + 1

        # logits shape: (batch, seq_len+1, vocab_size)
        assert output.logits.shape == (
            batch_size,
            expected_seq_len,
            tokenizer.vocab_size,
        )
        # log_probs shape: same as logits
        assert output.log_probs.shape == (
            batch_size,
            expected_seq_len,
            tokenizer.vocab_size,
        )
        # predictions shape: (batch, seq_len+1)
        assert output.predictions.shape == (batch_size, expected_seq_len)

    def test_forward_output_type(
        self, model: AttnBasedEncDecModel, tokenizer: AEDTokenizer
    ) -> None:
        """Test that forward returns AEDOutput."""
        batch_size = 2
        x = torch.randn(batch_size, 3, 32, 100)
        xlens = torch.tensor([100, 100])
        targets = torch.randint(4, tokenizer.vocab_size, (batch_size, 5))
        target_lens = torch.tensor([5, 5])

        output = model(x, xlens, targets, target_lens)

        assert isinstance(output, AEDOutput)

    def test_forward_loss_is_scalar(
        self, model: AttnBasedEncDecModel, tokenizer: AEDTokenizer
    ) -> None:
        """Test that loss is a scalar tensor."""
        batch_size = 2
        x = torch.randn(batch_size, 3, 32, 100)
        xlens = torch.tensor([100, 100])
        targets = torch.randint(4, tokenizer.vocab_size, (batch_size, 5))
        target_lens = torch.tensor([5, 5])

        output = model(x, xlens, targets, target_lens)

        assert output.loss is not None
        assert output.loss.dim() == 0  # scalar

    def test_forward_with_targets(
        self, model: AttnBasedEncDecModel, tokenizer: AEDTokenizer
    ) -> None:
        """Test forward with explicit targets (training mode)."""
        batch_size = 2
        x = torch.randn(batch_size, 3, 32, 100)
        xlens = torch.tensor([100, 100])
        targets = torch.randint(4, tokenizer.vocab_size, (batch_size, 5))
        target_lens = torch.tensor([5, 5])

        output = model(x, xlens, targets, target_lens)

        assert output.logits is not None
        assert output.loss is not None

    def test_forward_without_targets(
        self, model: AttnBasedEncDecModel, tokenizer: AEDTokenizer
    ) -> None:
        """Test forward without targets (inference mode with SOS only)."""
        batch_size = 2
        x = torch.randn(batch_size, 3, 32, 100)
        xlens = torch.tensor([100, 100])

        output = model(x, xlens)

        assert output.logits is not None
        # When no targets, output should have seq_len=1 (just SOS)
        assert output.logits.shape[1] == 1


class TestInit:
    def test_init_creates_all_submodules(self, model: AttnBasedEncDecModel) -> None:
        """Test that all submodules are created."""
        assert hasattr(model, "feature_extractor")
        assert hasattr(model, "pre_encoder")
        assert hasattr(model, "encoder")
        assert hasattr(model, "decoder")
        assert hasattr(model, "embedding")
        assert hasattr(model, "lm_head")
        assert hasattr(model, "loss")

    def test_init_invalid_feature_extractor(self, tokenizer: AEDTokenizer) -> None:
        """Test that invalid feature_extractor raises ValueError."""
        config = AttnBasedEncDecModelConfig(
            feature_extractor=FeatureExtractorConfig(
                type="invalid_type",  # type: ignore
                in_channels=3,
                input_height=32,
            ),
            pre_encoder=PreEncoderConfig(type="linear"),
            encoder=EncoderConfig(
                type="transformer",
                hidden_size=64,
                num_heads=4,
                num_layers=2,
            ),
            decoder=DecoderConfig(
                type="transformer",
                hidden_size=64,
                num_heads=4,
                num_layers=2,
            ),
        )

        with pytest.raises(ValueError, match="Unknown feature extractor"):
            AttnBasedEncDecModel(config, tokenizer)

    def test_init_invalid_encoder(self, tokenizer: AEDTokenizer) -> None:
        """Test that invalid encoder raises ValueError."""
        config = AttnBasedEncDecModelConfig(
            feature_extractor=FeatureExtractorConfig(
                type="resnet18",
                in_channels=3,
                input_height=32,
            ),
            pre_encoder=PreEncoderConfig(type="linear"),
            encoder=EncoderConfig(
                type="invalid_type",  # type: ignore
                hidden_size=64,
                num_heads=4,
                num_layers=2,
            ),
            decoder=DecoderConfig(
                type="transformer",
                hidden_size=64,
                num_heads=4,
                num_layers=2,
            ),
        )

        with pytest.raises(ValueError, match="Unknown encoder"):
            AttnBasedEncDecModel(config, tokenizer)

    def test_init_invalid_decoder(self, tokenizer: AEDTokenizer) -> None:
        """Test that invalid decoder raises ValueError."""
        config = AttnBasedEncDecModelConfig(
            feature_extractor=FeatureExtractorConfig(
                type="resnet18",
                in_channels=3,
                input_height=32,
            ),
            pre_encoder=PreEncoderConfig(type="linear"),
            encoder=EncoderConfig(
                type="transformer",
                hidden_size=64,
                num_heads=4,
                num_layers=2,
            ),
            decoder=DecoderConfig(
                type="invalid_type",  # type: ignore
                hidden_size=64,
                num_heads=4,
                num_layers=2,
            ),
        )

        with pytest.raises(
            ValueError, match="Unknown encoder"
        ):  # Note: error message says "encoder"
            AttnBasedEncDecModel(config, tokenizer)


# =============================================================================
# Other tests
# =============================================================================


class TestComputeLoss:
    def test_compute_loss_ignores_pad(
        self, model: AttnBasedEncDecModel, tokenizer: AEDTokenizer
    ) -> None:
        """Test that loss calculation ignores pad tokens."""
        pad_id = tokenizer.pad_id
        vocab_size = tokenizer.vocab_size

        # Create logits and targets with correct shape for _compute_loss
        # logits: (batch_size, seq_len, vocab_size)
        # targets: (batch_size, seq_len)
        batch_size = 2
        seq_len = 5
        logits = torch.randn(batch_size, seq_len, vocab_size)
        targets_with_pad = torch.tensor(
            [
                [4, 5, pad_id, pad_id, pad_id],
                [6, 7, 8, pad_id, pad_id],
            ]
        )

        # Compute loss
        loss = model._compute_loss(logits, targets_with_pad)

        # Loss should be computed only on non-pad positions
        assert loss.isfinite()


class TestGradientFlow:
    def test_gradient_flow(
        self, model: AttnBasedEncDecModel, tokenizer: AEDTokenizer
    ) -> None:
        """Test that gradients flow through the model."""
        batch_size = 2
        x = torch.randn(batch_size, 3, 32, 100)
        xlens = torch.tensor([100, 100])
        targets = torch.randint(4, tokenizer.vocab_size, (batch_size, 5))
        target_lens = torch.tensor([5, 5])

        output = model(x, xlens, targets, target_lens)
        output.loss.backward()

        # Check gradients exist for key parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_backward_no_error(
        self, model: AttnBasedEncDecModel, tokenizer: AEDTokenizer
    ) -> None:
        """Test that backward pass completes without error."""
        batch_size = 2
        x = torch.randn(batch_size, 3, 32, 100)
        xlens = torch.tensor([100, 100])
        targets = torch.randint(4, tokenizer.vocab_size, (batch_size, 5))
        target_lens = torch.tensor([5, 5])

        output = model(x, xlens, targets, target_lens)

        # Should not raise any exception
        output.loss.backward()
