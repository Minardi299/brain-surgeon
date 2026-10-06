import torch

from danger_tracker.refusal_direction import (
    project_out,
    HARMFUL_PROMPTS,
    HARMLESS_PROMPTS,
    DEFAULT_SOURCE_LAYER,
)


def test_project_out_removes_the_direction_component():
    # A residual that is exactly along the direction should be zeroed out.
    d = torch.tensor([3.0, 4.0, 0.0])
    d = d / d.norm()
    resid = 5.0 * d
    out = project_out(resid, d)
    assert torch.allclose(out, torch.zeros(3), atol=1e-5)


def test_project_out_preserves_the_orthogonal_component():
    d = torch.tensor([1.0, 0.0, 0.0])
    # resid = 2*d + orthogonal part; only the 2*d part should be removed.
    resid = torch.tensor([2.0, 7.0, -3.0])
    out = project_out(resid, d)
    assert torch.allclose(out, torch.tensor([0.0, 7.0, -3.0]), atol=1e-5)


def test_project_out_is_batched_and_shape_preserving():
    d = torch.tensor([0.0, 1.0, 0.0])
    resid = torch.randn(2, 5, 3)  # [batch, seq, d_model]
    out = project_out(resid, d)
    assert out.shape == resid.shape
    # The component along d (index 1) must be ~0 at every position.
    assert torch.allclose(out[..., 1], torch.zeros(2, 5), atol=1e-5)


def test_default_prompt_sets_are_balanced_and_nonempty():
    assert len(HARMFUL_PROMPTS) == len(HARMLESS_PROMPTS)
    assert len(HARMFUL_PROMPTS) >= 4
    assert isinstance(DEFAULT_SOURCE_LAYER, int)
