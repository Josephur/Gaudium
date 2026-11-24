#!/usr/bin/env python3
"""Reference cases of the Inkling 2 tokenizer: text -> pre-tokenizer pieces -> token ids, and
token ids -> decoded text.

usage: make_inkling2_fixture.py <model dir with tokenizer.json> <out .json>

Writes one JSON object:
  "encode": [{"text", "pieces", "ids"}]  pieces: the pre-tokenizer's split of each text segment
            between special tokens (no model files needed to check them); ids: the encoding
            with special tokens matched in the text;
  "decode": [{"ids", "skip_special", "text"}]  including id runs that cut a character's bytes
            (the decoder writes U+FFFD for an incomplete or invalid sequence).

The tracked test `tokenizer::tests::letter_case_pieces_match_the_reference` checks the pieces;
the ignored `tokenizer::tests::inkling2_ids_match_the_reference` checks ids and decoding against
the model's own `tokenizer.json` (set GAUDIUM_INKLING2_DIR to the model directory). Needs the
reference `tokenizers` package, CPU only.
"""
import json
import random
import sys

from tokenizers import Tokenizer

HAND = [
    "",
    "Hello, world!",
    "hello world",
    "Hello World HELLO world hELLO",
    "iPhone McDonald's NASA's JavaScript XMLHttpRequest",
    "I'm you're we've they'll she'd it's IT'S WE'RE YOU'VE THEY'LL I'D I'M",
    "don't DON'T Don'T don\u2019t",
    "'s 's 'S 'll 'LL 'Re 'vE x'\u017f x'\u017ft",
    "1 12 123 1234 12345 123456 1234567 3.14159 1,000,000 -42 +7 0x1F 1e10",
    "\u0661\u0662\u0663\u0664 \u096c\u096d \u2160\u2161\u2162 \u00bd \u2460\u2461",
    "    leading spaces",
    "trailing spaces    ",
    "tabs\there\t\tand\t\t\tthere",
    "line1\nline2\n\nline4\n\n\n",
    "crlf\r\nline\r\n\r\nend\r\n",
    "mixed \t \n \r\n  \n\t x",
    " \n",
    "\n\n\n",
    "   ",
    "a  b   c    d",
    "x\u00a0y\u2009z\u3000w\u200bv",
    "punctuation!!! ??? ... ;;; ::: --- ___ ***",
    "path/to/file.txt and C:\\Windows\\System32 and a//b///c",
    "end with slash/\nnext/\r\n//",
    "url: https://example.com/a/b?c=d&e=f#g",
    "email: someone@example.org",
    "def f(x):\n    return x * 2  # double\n",
    "fn main() {\n    let v: Vec<u8> = vec![1, 2, 3];\n    println!(\"{:?}\", v);\n}\n",
    "#include <stdio.h>\nint main(void) { printf(\"hi\\n\"); return 0; }",
    "{\"name\":\"get_weather\",\"args\":{\"city\":\"Oslo\",\"n\":2}}",
    "<html><body class=\"x\">Text &amp; more</body></html>",
    "SELECT * FROM users WHERE id = 42;",
    "\u4f60\u597d\uff0c\u4e16\u754c\uff01\u8fd9\u662f\u4e00\u4e2a\u6d4b\u8bd5\u3002",
    "\u65e5\u672c\u8a9e\u306e\u30c6\u30ad\u30b9\u30c8\u3067\u3059\u3002\u30ab\u30bf\u30ab\u30ca",
    "\ud55c\uad6d\uc5b4 \ud14d\uc2a4\ud2b8\uc785\ub2c8\ub2e4.",
    "\u041f\u0440\u0438\u0432\u0435\u0442, \u043c\u0438\u0440! \u041c\u041e\u0421\u041a\u0412\u0410 \u041c\u043e\u0441\u043a\u0432\u0430",
    "\u0393\u03b5\u03b9\u03ac \u03c3\u03bf\u03c5 \u039a\u03cc\u03c3\u03bc\u03b5 \u0391\u0398\u0397\u039d\u0391",
    "\u0645\u0631\u062d\u0628\u0627 \u0628\u0627\u0644\u0639\u0627\u0644\u0645",
    "\u05e9\u05dc\u05d5\u05dd \u05e2\u05d5\u05dc\u05dd",
    "\u0928\u092e\u0938\u094d\u0924\u0947 \u0926\u0941\u0928\u093f\u092f\u093e \u0939\u093f\u0928\u094d\u0926\u0940",
    "\u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e35\u0e0a\u0e32\u0e27\u0e42\u0e25\u0e01",
    "\u0bb5\u0ba3\u0b95\u0bcd\u0b95\u0bae\u0bcd \u0ba4\u0bae\u0bbf\u0bb4\u0bcd",
    "Stra\u00dfe \u00c4rger \u00f6ffnen \u00dcBER \u1e9e",
    "caf\u00e9 cafe\u0301 na\u00efve \u00c9COLE E\u0301COLE",
    "\u01c5ungla \u01c8ubljana \u1f88\u1f89 Title\u01c5Case",
    "\u02b0\u02b1 a\u02b0b A\u02b0B \u3005\u3005 \u30fc\u30fc",
    "z\u0301\u0302\u0303 \u0301\u0302 !\u0301! 1\u0301 A\u0301a",
    "emoji \U0001f600\U0001f601 \U0001f44d\U0001f3fd \U0001f468\u200d\U0001f469\u200d\U0001f467 \u2764\ufe0f \U0001f1f3\U0001f1f4",
    "\U0001f600\U0001f600\U0001f600text\U0001f600",
    "math: \u2200x\u2208\u211d, x\u00b2 \u2265 0 \u2227 \u221a4 = 2 \u2260 3",
    "currency: $100 \u20ac50 \u00a330 \u00a51000 \u20b9200",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",
    "0000000000000000000000000000000000000000000000000000000000000000",
    "                                                                    x",
    "\n" * 40 + "x",
    "\u4e00" * 50,
    "\U0001f600" * 20,
    "ab" * 60,
    "<|message_user|><|content_text|>Hi<|end_message|>",
    "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>What is 2+2?<|end_message|>",
    "<|message_model|><|content_thinking|>Let me think.<|end_message|><|message_model|><|content_text|>4<|end_message|><|content_model_end_sampling|>",
    "<|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":{\"city\":\"Oslo\"}}<|end_message|>",
    "<|message_tool|>get_weather<|content_text|>sunny, 21 C<|end_message|>",
    "<|message_user|><|content_image|><|unused_200054|><|end_message|>",
    "text<|end_message|>text<|endoftext|>more",
    "<|end_message|<|end_message|>|>",
    "<|message_user|",
    "<| message_user |>",
    "a<|unused|>b<|begin_of_text|>c",
    "\u0000\u0001\u0007\u001b[31mred\u001b[0m\u007f",
    "\ufeffBOM at start",
    "\ufffd replacement \ufffd",
    "Ω≈ç√∫˜µ≤≥÷ åß∂ƒ©˙∆˚¬…æ",
    "Ⓐⓑ ⒜ ㎒ ℌ ℵ ℓ №",
    "\U0002ebf0\U0002ebf1 \U00031350 new CJK",
    "a\u1c89b A\u1c89b \u1c8a",
    "x\u0085y x\u2028y x\u2029y x\u180ey x\u200by",
    "Ж'S ж'll Σ's",
]


def segments(text, specials):
    """The text between special tokens (what the pre-tokenizer sees), in order."""
    out, i, start = [], 0, 0
    while i < len(text):
        hit = None
        for s in specials:
            if text.startswith(s, i) and (hit is None or len(s) > len(hit)):
                hit = s
        if hit:
            if i > start:
                out.append(text[start:i])
            i += len(hit)
            start = i
        else:
            i += 1
    if start < len(text):
        out.append(text[start:])
    return out


def main():
    tok = Tokenizer.from_file(sys.argv[1] + "/tokenizer.json")
    specials = [t.content for t in tok.get_added_tokens_decoder().values()]
    rng = random.Random(20261123)
    pools = [
        "abcdefghijklmnopqrstuvwxyz", "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "0123456789",
        " \t\n\r", "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~", "\u00e9\u00df\u00c4\u0153\u0141\u017f",
        "\u4e00\u4e8c\u4e09\u56db\u4f60\u597d", "\u0430\u0431\u0432\u0410\u0411\u0412",
        "\u0301\u0308\u093e\u093f\u094d", "\u02b0\u3005\u30fc", "\U0001f600\U0001f44d\u2764",
        "\u00a0\u2009\u3000\u200b", "\u0661\u0662\u2160\u00bd",
    ]
    texts = list(HAND)
    for _ in range(260):
        n = rng.randint(1, 40)
        texts.append("".join(rng.choice(rng.choice(pools)) for _ in range(n)))
    for _ in range(24):
        words = []
        for _ in range(rng.randint(2, 12)):
            w = "".join(rng.choice(pools[rng.choice([0, 0, 0, 1, 2, 5, 6, 7])]) for _ in range(rng.randint(1, 8)))
            words.append(w)
        sep = rng.choice([" ", "  ", "\n", "'", "-", "/", " \n", "<|end_message|>"])
        texts.append(sep.join(words))

    pre = tok.pre_tokenizer
    enc = []
    for t in texts:
        pieces = [[p for p, _ in pre.pre_tokenize_str(seg)] for seg in segments(t, specials)]
        ids = tok.encode(t, add_special_tokens=False).ids
        assert tok.decode(ids, skip_special_tokens=False) == t, repr(t)
        enc.append({"text": t, "pieces": pieces, "ids": ids})

    dec = []
    multi = [e for e in enc if any(ord(c) > 0x7F for c in e["text"]) and len(e["ids"]) > 1]
    for e in multi[:40]:
        ids = e["ids"]
        cut = rng.randint(1, len(ids) - 1)
        for part in (ids[:cut], ids[cut:]):
            dec.append({"ids": part, "skip_special": False, "text": tok.decode(part, skip_special_tokens=False)})
    # single byte-level tokens that are a lone lead or continuation byte
    for b in (0x80, 0xBF, 0xC3, 0xE4, 0xF0, 0xFF):
        ch = bytes([b]).decode("latin-1")
        # the byte-level alphabet: bytes 0x80..0xA0 and 0xAD map above U+00FF
        alpha = {}
        n = 0
        for x in range(256):
            if 0x21 <= x <= 0x7E or 0xA1 <= x <= 0xAC or 0xAE <= x <= 0xFF:
                alpha[x] = chr(x)
            else:
                alpha[x] = chr(256 + n)
                n += 1
        tid = tok.token_to_id(alpha[b])
        dec.append({"ids": [tid], "skip_special": False, "text": tok.decode([tid], skip_special_tokens=False)})
        dec.append({"ids": [tid, tid], "skip_special": False, "text": tok.decode([tid, tid], skip_special_tokens=False)})
    for e in enc:
        if "<|" in e["text"]:
            dec.append({"ids": e["ids"], "skip_special": True, "text": tok.decode(e["ids"], skip_special_tokens=True)})

    with open(sys.argv[2], "w", encoding="utf-8") as f:
        json.dump({"encode": enc, "decode": dec}, f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")
    print(len(enc), "encode cases,", len(dec), "decode cases")


if __name__ == "__main__":
    main()
