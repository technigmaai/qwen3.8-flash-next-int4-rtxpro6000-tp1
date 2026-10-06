"""Offline checks with the actual model tokenizer; no GPU/server/tool execution."""
import json
import sys

import xgrammar as xgr
from transformers import AutoTokenizer
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser import ParserManager
from vllm.parser.abstract_parser import DelegatingParser

tokenizer = AutoTokenizer.from_pretrained(sys.argv[1], local_files_only=True)
compiler = xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(tokenizer))
parser_cls = ParserManager.get_parser('qwen3_coder', 'qwen3', True)
assert issubclass(parser_cls, DelegatingParser)
tools = [{'type': 'function', 'function': {'name': name,
    'description': 'Offline mock; never executed.', 'parameters': {
        'type': 'object', 'properties': {'value': {'type': 'string'}},
        'required': ['value'], 'additionalProperties': False}}}
    for name in ['probe_ping', 'other_tool']]
valid = '<tool_call>\n<function=probe_ping>\n<parameter=value>OK</parameter>\n</function>\n</tool_call>'
for thinking in [False, True]:
    for choice in ['required', {'type': 'function', 'function': {'name': 'probe_ping'}}]:
        request = ChatCompletionRequest(model='rtx', tools=tools, tool_choice=choice,
            messages=[{'role': 'user', 'content': 'Reply OK. Do not call any tools.'}],
            chat_template_kwargs={'enable_thinking': thinking})
        parser = parser_cls(tokenizer, request.tools,
            chat_template_kwargs=request.chat_template_kwargs)
        assert parser.reasoning_parser is not None and parser.tool_parser is not None
        assert parser.reasoning_parser._parser_engine.thinking_enabled is thinking
        adjusted = parser.adjust_request(request)
        assert adjusted.structured_outputs and adjusted.structured_outputs.structural_tag
        tag = adjusted.structured_outputs.structural_tag
        grammar = compiler.compile_structural_tag(tag)
        assert not xgr.GrammarMatcher(grammar).accept_string('OK'), tag
        assert xgr.GrammarMatcher(grammar).accept_string(valid), tag
        assert not xgr.GrammarMatcher(grammar).accept_string(valid.replace('probe_ping', 'invented')), tag
        if choice != 'required':
            assert not xgr.GrammarMatcher(grammar).accept_string(valid.replace('probe_ping', 'other_tool'))
        source = ('Testing.\n</think>\n' if thinking else '') + valid
        _, content, calls = parser.parse(source, request, enable_auto_tools=True,
            model_output_token_ids=tokenizer.encode(source, add_special_tokens=False))
        assert calls and calls[0].name == 'probe_ping', (content, calls)
        assert json.loads(calls[0].arguments) == {'value': 'OK'}, calls
        print(f'OFFLINE thinking={thinking} choice={choice}: adapters, grammar rejection/acceptance, parsing PASS', flush=True)
print('ALL OFFLINE PARSER CHECKS PASSED', flush=True)
