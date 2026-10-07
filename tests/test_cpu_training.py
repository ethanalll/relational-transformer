"""CPU readiness: the attention math and a real model optimizer step."""

import pytest
import torch

from rt import RelationalTransformer
from rt.model import MaskedAttention


def test_cpu_attention_matches_masked_equation():
    torch.manual_seed(7)
    attn = MaskedAttention(d_model=8, num_heads=2)
    with torch.no_grad():
        attn.wo.weight.copy_(torch.eye(8))
    x = torch.randn(1, 4, 8, requires_grad=True)
    mask = torch.tensor(
        [
            [
                [True, True, False, False],
                [False, False, False, False],
                [False, True, True, False],
                [True, False, False, True],
            ]
        ]
    )
    counts = mask.sum(-1, keepdim=True).float()
    actual = attn(x, mask, counts)

    q = attn.wq(x).reshape(1, 4, 2, 4).transpose(1, 2)
    k = attn.wk(x).reshape(1, 4, 2, 4).transpose(1, 2)
    v = attn.wv(x).reshape(1, 4, 2, 4).transpose(1, 2)
    q = attn.q_norm(q) * attn.scale * counts.clamp_min(1).log().unsqueeze(1)
    k = attn.k_norm(k)
    scores = q @ k.transpose(-1, -2) / 4
    scores = scores.masked_fill(~mask[:, None], -torch.inf)
    # Empty neighborhoods contribute exactly zero rather than NaN.
    probabilities = torch.nan_to_num(scores.softmax(-1))
    expected = (probabilities @ v).transpose(1, 2).reshape(1, 4, 8)
    expected = attn.wo(expected * (2 * torch.sigmoid(attn.wg(x))))
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert torch.equal(actual[:, 1], torch.zeros_like(actual[:, 1]))
    actual.square().sum().backward()
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("materialize", [False, True])
def test_cpu_model_forward_backward_and_update(materialize):
    torch.manual_seed(0)
    model = RelationalTransformer(2, 16, 8, 2, 32, False, materialize)
    b, s = 2, 8
    batch = {
        "node_idxs": torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3]]).expand(b, -1),
        "f2p_nbr_idxs": torch.full((b, s, 5), -1, dtype=torch.long),
        "col_name_idxs": torch.tensor([[0, 1] * 4]).expand(b, -1),
        "table_name_idxs": torch.zeros(b, s, dtype=torch.long),
        "is_padding": torch.tensor([[False] * 7 + [True]]).expand(b, -1),
        "is_targets": torch.tensor([[True] + [False] * 7]).expand(b, -1),
        "sem_types": torch.zeros(b, s, dtype=torch.long),
        "col_name_values": torch.randn(b, s, 8),
        "number_values": torch.randn(b, s, 1),
        "text_values": torch.zeros(b, s, 8),
        "datetime_values": torch.zeros(b, s, 1),
        "boolean_values": torch.zeros(b, s, 1),
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = model.dec_dict["number"].weight.detach().clone()
    loss = model(batch, return_embeddings=False)[0]
    assert torch.isfinite(loss)
    loss.backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
    optimizer.step()
    assert not torch.equal(before, model.dec_dict["number"].weight)
