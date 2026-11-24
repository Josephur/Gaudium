# Vision regression suite

Every speculative method must keep working with images: its output must be token-identical to
the engine's own non-speculative K=0 run, greedy and sampled with the same seed, and still
answer the picture correctly. `gaudium --vision-suite` checks exactly that on the card, for every
model and method listed in a suite file. A drafter is served beside a model's vision tower only
once the suite has passed for that pairing (the table `vis_drafter_ok` in
`gaudium/verified/vision_budget.bend`; any other pairing is served text only).

## What one row checks

A row is one model with one method (K=0 or a drafter). The suite runs each row as its own
`gaudium --generate` process, one after another; the suite process itself never opens the card.
One engine load serves, in this order:

1. a text request, on an engine that has not seen an image yet;
2. every image case of the suite file (each request with its own images);
3. the text request again.

Every request runs under every sampler setting of the suite file (greedy, and sampled with a fixed
seed), each run right after its K=0 reference on the same engine (`--compare-k0`). The row passes
when all of these hold:

- **identity**: every run is token-identical to its K=0 reference. An opt-in method (marked
  `"opt_in_forks"` in the suite file AND selecting an opt-in form in its own options: a draft
  tree, which may differ from K=0 in the last bit of a leaf row, or `--gdn-verify mme`) may fork
  only at ties: the run's whole stream is replayed on the K=0 loop, and at every position where
  K=0 would have chosen otherwise the two tokens' BF16 logits must be at most one unit in the
  last place apart;
- **answers**: every answer, decoded with the model's tokenizer, contains one accepted word of
  every `expect` group of its case;
- **no state left behind**: the text request after the images gives the tokens it gave before
  them, under every sampler setting.

The verdict itself is verified code (`gaudium/verified/vision_suite.bend`): no fork passes a
default method, an untraced fork or a missing report never passes, a diverged run passes an
opt-in method only when its replay completed, traced the first divergence and found only ties,
and no failed check is outweighed by the others.

## Running it

The suite file and its fixed images are in `gaudium/tests/vision_suite/`
(`make_images.py` regenerates the images). Model names in the suite file and `{models}` in its
options resolve in the `--models` directory.

```bash
# compile every row's recipes into the recipe cache first (no card needed)
gaudium --vision-suite gaudium/tests/vision_suite/suite.json --models <models dir> \
        --kernels <kernel build dir> --cache-dir <recipe cache> --build-cache
# then run it (one card job; nothing else may hold the card meanwhile)
gaudium --vision-suite gaudium/tests/vision_suite/suite.json --models <models dir> \
        --kernels <kernel build dir> --cache-dir <recipe cache> --out <dir>
```

It prints one line per row (identity runs passed, answers right, after-request, draft acceptance,
tokens per launch, time) followed by every failure, and writes `report.json` and each row's
command line, prompts and logs under `--out`. The exit status is 0 only when every row passed.
`--only <model>` or `--only <model>/<method>` runs a subset; options after `--` are added to
every row (for example `-- --max-ctx 16384`).

## Adding a model family or a drafter

A new family is one more entry of `models` in the suite file, a new drafter one more entry of a
model's `methods`, with the options its `gaudium --generate` run takes. Nothing in the runner is
specific to a model family. Once the rows of a pairing pass, its case of `vis_drafter_ok` may
become `True{}` (Gemma 4's drafters, for example, through their own rows).

## How the drafters see an image

- **DSpine and DFlash2** never embed prompt tokens: their context is the target's own residual
  stream (the tapped layers), so an image position reaches them as what the target computed from
  the image. Their block rows are generated tokens and masks. Their RoPE rows are addressed and
  rotated by KV slot on both sides of their attention.
- **MTP** (the model's own multi-token-prediction head) embeds the token after each position. For
  a position followed by an image token, the prompt pass reads that image's own embedding row
  instead of the image-pad token's embedding (`engine/mtp.rs`, the image form of the context
  pass).
- Prompt-lookup drafting, when ported, must never match inside an image span
  (`PromptLayout::lookup_history`).
