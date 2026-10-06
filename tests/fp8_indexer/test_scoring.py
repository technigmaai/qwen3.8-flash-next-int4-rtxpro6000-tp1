"""Preview scorer tests against quantized-input FP32 reference."""
import pytest
import torch
from vllm.models.qwen3_8_flash_next.nvidia.ops.qsa import qsa_mqa_paged, qsa_select_paged_tokens


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize('query_lens', [[1], [3, 1, 5], [37, 1, 3]])
def test_paged_scores_and_topk(dtype, query_lens):
    torch.manual_seed(20261006)
    device = 'cuda'
    # The retained SM120 deterministic kernel supports k=512/1024/2048.
    # Match the real model's 2048-token budget (512 compressed groups).
    heads, width, cr, page, per_req = 4, 128, 4, 8, 128
    columns = page * per_req
    requests = len(query_lens)
    rows = sum(query_lens)
    q = torch.randn(rows, heads, width, device=device, dtype=torch.bfloat16).to(dtype)
    cache = torch.randn(requests * per_req, page, 1, width,
                        device=device, dtype=torch.bfloat16).to(dtype)
    table = torch.randperm(requests * per_req, device=device, dtype=torch.int32).reshape(requests, per_req)
    mapping = torch.repeat_interleave(torch.arange(requests, device=device, dtype=torch.int32),
                                     torch.tensor(query_lens, device=device))
    lengths = torch.tensor([columns * cr - 1 - 7 * i for i in range(requests)], device=device, dtype=torch.int32)
    positions = torch.cat([torch.arange(int(lengths[i]) - n, int(lengths[i]), device=device, dtype=torch.int64)
                           for i, n in enumerate(query_lens)])
    logits, visible = qsa_mqa_paged(q, cache, table, mapping, positions, lengths, cr)
    reference = torch.full_like(logits, -float('inf'))
    for row in range(rows):
        req = int(mapping[row])
        count = min((int(positions[row]) + 1) // cr, int(lengths[req]) // cr)
        assert int(visible[row]) == count
        keys = cache[table[req].long()].reshape(columns, width)[:count].float()
        scores = torch.relu(keys @ q[row].float().T).sum(-1) / width**0.5
        reference[row, :count] = scores
        torch.testing.assert_close(logits[row, :count], scores, rtol=1e-3, atol=1e-3)
    topk_tokens = 2048
    result = qsa_select_paged_tokens(q, cache, table, mapping, positions, lengths, topk_tokens, cr)
    for row in range(rows):
        selected = result[row, :topk_tokens].reshape(-1, cr)[:, 0].long() // cr
        wanted = reference[row].topk(topk_tokens // cr).values.sort().values
        actual = reference[row, selected].sort().values
        torch.testing.assert_close(actual, wanted, rtol=1e-3, atol=1e-3)
    torch.cuda.synchronize()


def test_reject_mismatched_query_cache_dtype():
    q = torch.empty((1, 4, 128), device='cuda', dtype=torch.bfloat16)
    cache = torch.empty((1, 8, 1, 128), device='cuda', dtype=torch.float8_e4m3fn)
    with pytest.raises(ValueError, match='dtypes must match'):
        qsa_mqa_paged(q, cache, torch.zeros((1, 1), device='cuda', dtype=torch.int32),
                      torch.zeros(1, device='cuda', dtype=torch.int32),
                      torch.zeros(1, device='cuda', dtype=torch.int64),
                      torch.ones(1, device='cuda', dtype=torch.int32), 4)
