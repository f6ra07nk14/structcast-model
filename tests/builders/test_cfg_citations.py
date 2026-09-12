"""The shipped templates, and what a builder makes of them, cite no repository document.

A `cfg/` template is documentation the user reads and copies: it arrives in an editor, often outside
a checkout of this repository, so `docs/adr/...` and `CONTEXT.md` name nothing the reader can open
and a comment leaning on one leaves a rule without its reason. `REFERENCE.md` is the exception the
templates may point at, since it ships with them. The same rule already holds for the emitted code
of one fixture per framework; what is asserted here is that it holds for the templates themselves
and for what the builders render them into.
"""

from pathlib import Path
from typing import Any

from structcast_model.builders.flax import FlaxBuilder, FlaxLearnerBuilder
from structcast_model.builders.keras import KerasBuilder, KerasLearnerBuilder
from structcast_model.builders.torch import TorchBuilder, TorchLearnerBuilder
from tests import CFG_DIR

CITATIONS = ("docs/adr", "CONTEXT.md")
"""The two repository documents a reader of a template or a generated file cannot open."""

RENDERED: tuple[tuple[Any, str], ...] = (
    (TorchBuilder, "torch/models/VisionTransformer.yaml"),
    (TorchLearnerBuilder, "torch/learners/ImageClassifierShowcase.yaml"),
    (FlaxBuilder, "flax/models/VisionTransformer.yaml"),
    (FlaxLearnerBuilder, "flax/learners/ImageClassifierShowcase.yaml"),
    (KerasBuilder, "keras/models/VisionTransformer.yaml"),
    (KerasLearnerBuilder, "keras/learners/ImageClassifierShowcase.yaml"),
)
"""One model and one learner per framework: the showcase templates carry the densest commentary."""


def test_templates_and_their_generated_code_cite_no_repository_document(tmp_path: Path) -> None:
    """Every `cfg/` template, and the module each builder writes from one, stands on its own prose.

    A citation is not a formatting nit here: it is the difference between a comment that explains a
    constraint and one that defers the explanation to a file the reader does not have. The whole
    written module is read, not just the class script, because the layer sources travel in it.
    """
    for template in sorted(CFG_DIR.rglob("*.yaml")):
        text = template.read_text(encoding="utf-8")
        for citation in CITATIONS:
            assert citation not in text, f"{template} cites {citation}"

    for index, (builder, relative) in enumerate(RENDERED):
        builder.from_path(CFG_DIR / relative)()(module := tmp_path / f"{index}.py")
        text = module.read_text(encoding="utf-8")
        for citation in CITATIONS:
            assert citation not in text, f"{relative} renders a file citing {citation}"
