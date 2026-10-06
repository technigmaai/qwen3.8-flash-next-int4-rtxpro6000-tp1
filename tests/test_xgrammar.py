"""Real XGrammar stop/rollback regression checks for upstream vLLM PR 52805."""
import xgrammar as xgr
from vllm.v1.structured_output.backend_xgrammar import XgrammarGrammar

vocab = ['{', '}', '"', 'ready', ':', 'true', ' ', '<eos>', '\n']
info = xgr.TokenizerInfo(vocab, xgr.VocabType.RAW, stop_token_ids=[7])
ctx = xgr.GrammarCompiler(info).compile_json_schema(
    '{"type":"object","properties":{"ready":{"type":"boolean"}},'
    '"required":["ready"],"additionalProperties":false}')
matcher = xgr.GrammarMatcher(ctx, max_rollback_tokens=16)
grammar = XgrammarGrammar(vocab_size=len(vocab), matcher=matcher, ctx=ctx)
prefix = [0, 2, 3, 2, 4, 5, 1]
assert grammar.accept_tokens('test', prefix)
assert grammar.validate_tokens([7, 8]) == [7]
assert not matcher.is_terminated()  # speculative validation rolled back
assert grammar.accept_tokens('test', [7, 8])
assert grammar.is_terminated() and grammar.num_processed_tokens == 8
assert grammar.accept_tokens('test', [8])  # post-EOS MTP artifact is ignored
assert grammar.validate_tokens([8]) == []
grammar.reset()
assert not grammar.is_terminated() and grammar.num_processed_tokens == 0
assert grammar.accept_tokens('test', prefix + [7, 8])
assert grammar.is_terminated() and grammar.num_processed_tokens == 8
print('Real XGrammar terminal token, validation rollback and reset: PASS')
