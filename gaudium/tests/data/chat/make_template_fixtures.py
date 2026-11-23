#!/usr/bin/env python3
"""Reference renderings of chat templates by Jinja2, set up the way chat templates are rendered
for a model: `trim_blocks` and `lstrip_blocks` on, loop controls, a `{% generation %}` block that
renders its body unchanged, the `tojson` filter as json.dumps(x, ensure_ascii=False, indent=None,
separators=None, sort_keys=False) with each argument settable, and the globals
`raise_exception` and `strftime_now`.

usage: make_template_fixtures.py <out dir> [<Inkling 2 model dir> [<GGUF chat template file>]]

Writes into <out dir>:
  jinja_features.json   this script's own template (FEATURES below) and its renderings: checked
                        by the tracked test `server::jinja::tests::features_render_like_jinja2`;
  inkling2_renders.json (with a model dir) the model's `chat_template.jinja` rendered with each
                        context of `inkling2_contexts()`: checked by the ignored test
                        `server::chat::tests::inkling2_template_renders_like_jinja2` (set
                        GAUDIUM_INKLING2_DIR to the model directory). Given the template of the
                        GGUF build too, every context must render to the same text with it.

Each case is {"name", "context", "text"} or {"name", "context", "error"} (the message of
`raise_exception`). Needs jinja2.
"""
import json
import os
import sys
from datetime import datetime

import jinja2
from jinja2 import nodes
from jinja2.ext import Extension
from jinja2.sandbox import ImmutableSandboxedEnvironment


class Raised(Exception):
    pass


class Generation(Extension):
    tags = {"generation"}

    def parse(self, parser):
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
        return nodes.CallBlock(self.call_method("_render", []), [], [], body).set_lineno(lineno)

    def _render(self, caller):
        return caller()


def environment():
    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=[Generation, jinja2.ext.loopcontrols]
    )

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    def raise_exception(message):
        raise Raised(message)

    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
    return env


def render(env, source, cases):
    tpl = env.from_string(source)
    out = []
    for name, ctx in cases:
        try:
            out.append({"name": name, "context": ctx, "text": tpl.render(**ctx)})
        except Raised as e:
            out.append({"name": name, "context": ctx, "error": str(e)})
    return out


# The feature template: what the Inkling 2 template and similar ones use, written for this test.
FEATURES = r"""{%- macro block(role, kind, text) -%}<{{ role }}:{{ kind }}>{{ text }}</>{%- endmacro -%}
{%- set kinds = {"user": "U", "assistant": "A", "system": "S"} -%}
{%- set st = namespace(n=0, seen=[]) -%}
{%- for m in messages -%}
{%- if m.role not in kinds -%}{{ raise_exception("Unknown role: " ~ m.role) }}{%- endif -%}
{%- set st.n = st.n + 1 -%}
{%- set st.seen = st.seen + [m.role] -%}
{%- set body -%}
{%- if m.content is string -%}{{ block(kinds[m.role], "t", m.content) }}
{%- elif m.content is sequence -%}{%- for p in m.content -%}{{ block(kinds[m.role], p.type, p.text | default("")) }}{%- endfor -%}
{%- endif -%}
{%- endset -%}
{%- if m.role == "assistant" -%}{%- generation -%}{{ body }}{%- endgeneration -%}{%- else -%}{{ body }}{%- endif -%}
{%- if m.args is defined -%}
{%- if m.args is not mapping -%}{{ raise_exception("args must be an object") }}{%- endif -%}
[{{ m.args | tojson }}|{{ m.args | tojson(sort_keys=true) }}|{{ m.args | tojson(separators=(",", ":")) }}|{{ m.args | tojson(ensure_ascii=true, sort_keys=true, separators=(",", ":")) }}|{{ m.args | tojson(indent=2) }}]
{%- for k, v in m.args | dictsort(case_sensitive=true) -%}{{ k | tojson }}={{ v | tojson(separators=(",", ":")) }}{%- if not loop.last -%};{%- endif -%}{%- endfor -%}
{%- for k, v in m.args | dictsort -%}({{ k }}){%- endfor -%}
{%- endif -%}
{%- endfor -%}
{{- "#" ~ st.n ~ ":" ~ (st.seen | join(",")) -}}
{%- if level is defined -%}
{%- set num = level | float if level is not string else {"low": 0.2, "high": 0.9}[level | trim] -%}
{%- if num is undefined -%}{{ raise_exception("Unknown level: " ~ level) }}{%- endif -%}
{%- if num < 0.0 or num > 0.99 -%}{{ raise_exception("level must be in [0.0, 0.99]") }}{%- endif -%}
{{- " L=" -}}{%- if num == 0.0 -%}0{%- else -%}{{ num }}{%- endif -%}
{%- endif -%}
{{- " " ~ (title | default("none"))[1:4] ~ (title | default("none"))[-2:] ~ (title | default("x"))[::-1] -}}
"""


def feature_cases():
    args = {"b": 1, "a": [1, 2.5, None, True], "B": {"z": "é\U0001f600", "y": "line\n\"q\""}, "_": {}}
    return [
        ("plain", {"messages": [{"role": "user", "content": "Hi"}]}),
        ("parts", {"messages": [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "image"}]}]}),
        ("args", {"messages": [{"role": "assistant", "content": "ok", "args": args}]}),
        ("args_empty", {"messages": [{"role": "assistant", "content": "", "args": {}}]}),
        ("args_not_mapping", {"messages": [{"role": "assistant", "content": "", "args": [1]}]}),
        ("bad_role", {"messages": [{"role": "robot", "content": "x"}]}),
        ("multi", {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]}),
        ("level_name", {"messages": [{"role": "user", "content": "x"}], "level": " high "}),
        ("level_number", {"messages": [{"role": "user", "content": "x"}], "level": 0.5}),
        ("level_zero", {"messages": [{"role": "user", "content": "x"}], "level": 0}),
        ("level_high_number", {"messages": [{"role": "user", "content": "x"}], "level": 0.99}),
        ("level_out_of_range", {"messages": [{"role": "user", "content": "x"}], "level": 1.5}),
        ("title_slices", {"messages": [{"role": "user", "content": "x"}], "title": "Inkling été"}),
    ]


def inkling2_contexts():
    base = {"add_generation_prompt": True, "eos_token": "<|content_model_end_sampling|>", "pad_token": "<|endoftext|>"}
    weather = {"type": "function", "function": {"name": "get_weather", "description": "Weather for a city.",
               "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City é"},
                              "days": {"type": "integer"}}, "required": ["city"]}}}
    bare = {"name": "lookup", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}
    nodesc = {"type": "function", "function": {"name": "now"}}
    call = {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Oslo", "days": 2}}}
    call2 = {"id": "call_2", "type": "function", "function": {"name": "lookup",
             "arguments": {"q": "café \"quoted\"", "Z": [1, 2.5, None, True, {"b": 1, "a": 2}], "a": {"y": 1, "x": "\n"}}}}
    u = lambda t: {"role": "user", "content": t}
    a = lambda t: {"role": "assistant", "content": t}
    cases = []

    def add(name, msgs, **kw):
        ctx = dict(base)
        ctx["messages"] = msgs
        ctx.update(kw)
        cases.append((name, ctx))

    add("plain", [u("Hi")])
    add("system", [{"role": "system", "content": "Be brief."}, u("Hi")])
    add("two_systems", [{"role": "system", "content": "A"}, {"role": "system", "content": "B"}, u("Hi")])
    add("only_system", [{"role": "system", "content": "Alone."}])
    add("multi_turn", [u("Hi"), a("Hello!"), u("How are you?")])
    add("reasoning_kept", [u("2+2?"), {"role": "assistant", "content": "4", "reasoning_content": "Add them."}, u("And 3+3?")])
    add("reasoning_only", [u("x"), {"role": "assistant", "content": "", "reasoning_content": "hmm"}])
    add("empty_assistant", [u("x"), a("")])
    add("tools_declared", [u("Weather in Oslo?")], tools=[weather, bare, nodesc])
    add("tool_call_and_result", [u("Weather in Oslo?"), {"role": "assistant", "content": "", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"}], tools=[weather])
    add("tool_call_reasoning_text", [u("Weather?"), {"role": "assistant", "content": "Checking.", "reasoning_content": "Use the tool.",
        "tool_calls": [call, call2]}, {"role": "tool", "name": "get_weather", "content": "sunny"},
        {"role": "tool", "tool_call_id": "call_2", "content": "found"}], tools=[weather, bare])
    add("tool_result_unnamed", [u("x"), {"role": "assistant", "content": None, "tool_calls": [call]},
        {"role": "tool", "content": "no id"}, {"role": "tool", "tool_call_id": "call_9", "content": "unknown id"}], tools=[weather])
    add("tool_call_no_args", [u("x"), {"role": "assistant", "content": None,
        "tool_calls": [{"id": "c", "function": {"name": "now"}}, {"id": "d", "function": {"name": "now", "arguments": {}}}]}], tools=[nodesc])
    add("assistant_named", [u("x"), {"role": "assistant", "name": "get_weather", "content": "", "tool_calls": [call]}])
    add("assistant_name_mismatch", [u("x"), {"role": "assistant", "name": "other", "content": "", "tool_calls": [call]}])
    add("args_as_string", [u("x"), {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c", "function": {"name": "f", "arguments": "{\"a\": 1}"}}]}])
    add("args_not_object", [u("x"), {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c", "function": {"name": "f", "arguments": [1, 2]}}]}])
    add("call_name_not_string", [u("x"), {"role": "assistant", "content": "", "tool_calls": [{"id": "c", "function": {"arguments": {}}}]}])
    add("image_and_text", [{"role": "user", "content": [{"type": "text", "text": "What is this?"}, {"type": "image"}]}])
    add("images_only", [{"role": "user", "content": [{"type": "image"}, {"type": "image"}]}])
    add("text_parts", [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}])
    add("tool_image", [u("x"), {"role": "assistant", "content": "", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "call_1", "content": [{"type": "text", "text": "chart:"}, {"type": "image"}]}])
    add("assistant_image", [u("x"), {"role": "assistant", "content": [{"type": "image"}]}])
    add("audio", [{"role": "user", "content": [{"type": "audio"}]}])
    add("video", [{"role": "user", "content": [{"type": "video"}]}])
    add("unknown_role", [{"role": "narrator", "content": "x"}])
    add("unicode", [u("你好 \U0001f600 <|end_message|> é")])
    for lvl in ["none", "minimal", "low", "medium", "high", "xhigh", "max", " high ", "turbo", None]:
        add("effort_" + str(lvl).strip().replace(" ", "_"), [u("Hi")], reasoning_effort=lvl)
    for num in [0, 0.0, 0.5, 0.25, 0.99, 1, 1.0, -0.1]:
        add("effort_num_" + str(num).replace(".", "_").replace("-", "neg"), [u("Hi")], reasoning_effort=num)
    add("effort_with_system", [{"role": "system", "content": "S"}, u("Hi")], reasoning_effort="low")
    add("effort_and_flag", [u("Hi")], reasoning_effort="none", enable_thinking=False)
    return cases


def main():
    out = sys.argv[1]
    env = environment()
    with open(os.path.join(out, "jinja_features.json"), "w", encoding="utf-8") as f:
        json.dump({"template": FEATURES, "cases": render(env, FEATURES, feature_cases())}, f, ensure_ascii=False, indent=1)
        f.write("\n")
    if len(sys.argv) > 2:
        source = open(os.path.join(sys.argv[2], "chat_template.jinja"), encoding="utf-8").read()
        cases = render(env, source, inkling2_contexts())
        if len(sys.argv) > 3:
            gguf = open(sys.argv[3], encoding="utf-8").read()
            again = render(env, gguf, inkling2_contexts())
            assert [c.get("text", c.get("error")) for c in cases] == [c.get("text", c.get("error")) for c in again]
            print("the GGUF template renders every context the same")
        with open(os.path.join(out, "inkling2_renders.json"), "w", encoding="utf-8") as f:
            json.dump({"cases": cases}, f, ensure_ascii=False, indent=1)
            f.write("\n")
        print(len(cases), "Inkling 2 cases,", sum("error" in c for c in cases), "raise")


if __name__ == "__main__":
    main()
