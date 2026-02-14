"""Test AutoName."""

from structcast_model.builders.auto_name import AutoName


def test_auto_name_first_call() -> None:
    """Test AutoName generates base name on first call."""
    auto_name = AutoName()
    assert auto_name("layer") == "layer"


def test_auto_name_subsequent_calls() -> None:
    """Test AutoName generates unique names on subsequent calls."""
    auto_name = AutoName(infix="_")
    assert auto_name("layer") == "layer"
    assert auto_name("layer") == "layer_1"
    assert auto_name("layer") == "layer_2"


def test_auto_name_custom_infix() -> None:
    """Test AutoName with custom infix."""
    auto_name = AutoName(infix="@")
    assert auto_name("layer") == "layer"
    assert auto_name("layer") == "layer@1"
    assert auto_name("layer") == "layer@2"


def test_auto_name_different_values() -> None:
    """Test AutoName with different values."""
    auto_name = AutoName(infix="_")
    assert auto_name("conv") == "conv"
    assert auto_name("relu") == "relu"
    assert auto_name("conv") == "conv_1"
    assert auto_name("relu") == "relu_1"
    assert auto_name("conv") == "conv_2"


def test_auto_name_reset() -> None:
    """Test AutoName reset functionality."""
    auto_name = AutoName(infix="_")
    assert auto_name("layer") == "layer"
    assert auto_name("layer") == "layer_1"
    assert auto_name("layer") == "layer_2"

    auto_name.reset()

    assert auto_name("layer") == "layer"
    assert auto_name("layer") == "layer_1"


def test_auto_name_empty_infix() -> None:
    """Test AutoName with empty infix (default)."""
    auto_name = AutoName(infix="")
    assert auto_name("layer") == "layer"
    assert auto_name("layer") == "layer1"
    assert auto_name("layer") == "layer2"
