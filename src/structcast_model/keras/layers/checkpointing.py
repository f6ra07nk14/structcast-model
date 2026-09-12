"""The backend guard a generated Keras layer with `GRADIENT_CHECKPOINTING` enabled runs first."""

from logging import getLogger

import keras

logger = getLogger(__name__)


def disable_flash_attention_for_remat() -> None:
    """Turn Keras' flash attention dispatch off for this process, on the JAX backend alone.

    Called by the `__init__` of every generated layer that checkpoints, before that layer builds its
    own sub-layers: `keras.layers.MultiHeadAttention.__init__` caches the dispatch decision as
    `flash_attention or keras.config.is_flash_attention_enabled()`, so a flip made after the
    sub-layer exists -- at training time, say -- would never reach it.

    Attention inside a `keras.remat` body is where the JAX cuDNN fused kernel turns fatal: it refuses
    sequence lengths it cannot serve (ViT-B/16 at 224px asks it for 197) and outside rematerialization
    Keras catches that refusal and falls back to the unfused path, while inside the recomputation
    nothing catches it and the training step dies with a `NotImplementedError`. The fallback is what
    those shapes get anyway, so disabling the dispatch costs them nothing and is what makes the
    recomputation survive; refusing the combination instead would take activation checkpointing away
    from every attention model on this backend, and the memory it saves is the point of the field.

    The switch is process-global and one-way here, which is why it is gated on the backend: a
    TensorFlow or torch run routes its own attention and is left alone. A layer built with an
    explicit `flash_attention=True` is left alone too -- that argument wins over the global switch.
    """
    # `is False`, not falsy: the default and the enabled state are both `None` ("attempt it"), and
    # only an explicit disable reads back as False -- which is also what keeps the warning to one
    # line however many checkpointed layers a model builds.
    if keras.backend.backend() != "jax" or keras.config.is_flash_attention_enabled() is False:
        return
    keras.config.disable_flash_attention()
    logger.warning(
        "Disabled Keras flash attention for this process: a JAX-backend layer with "
        "GRADIENT_CHECKPOINTING enabled recomputes its body inside keras.remat, where the cuDNN "
        "fused attention kernel raises on the sequence lengths it cannot serve instead of falling "
        "back. Attention now always runs on the unfused path, which is slower."
    )
