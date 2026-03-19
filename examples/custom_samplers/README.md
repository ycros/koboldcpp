# Custom Samplers

Request-provided JavaScript samplers are disabled by default.
Launch KoboldCpp with:

```bash
python koboldcpp.py --allowcustomsamplers ...
```

## Request Fields

- `custom_sampler`: JavaScript source code
- `custom_sampler_params`: optional JSON value passed to `init()`
- `custom_sampler_debug`: optional boolean that enables per-token pre/post custom sampler traces and JS log capture
- `sampler_order`: optional sampler order array; custom sampler slot is `7`

If you are sending raw HTTP JSON, prefer passing `custom_sampler_params` as an actual JSON value rather than a pre-serialized string.

If `custom_sampler` is provided and `sampler_order` is omitted, the default becomes:

```json
[6, 0, 1, 3, 4, 2, 7, 5]
```

This places the custom sampler just before temperature.

## JavaScript Entry Points

```js
function init(params, s) {
    // optional, runs once per request
}

function sample(s) {
    // runs once per token
    // return undefined to continue native sampling pipeline
    // return a token id to select it immediately
    return s.pick();
}
```

## Available Methods

Candidate transforms:

- `softmax()`
- `topK(k)`
- `topA(a, minKeep)`
- `topP(p, minKeep)`
- `minP(p, minKeep)`
- `tailFree(z, minKeep)`
- `typical(p, minKeep)`
- `topNSigma(nsigma)`
- `temperature(temp, smoothingFactor, smoothingCurve)`
- `entropy(minTemp, maxTemp, exponent, smoothingFactor, smoothingCurve)`
- `repPenalty(range, penalty, slope, presencePenalty)`
- `xtc(threshold, probability)`

Candidate inspection and mutation:

- `size()`
- `id(i)`
- `logit(i)`
- `prob(i)`
- `setLogit(i, value)`
- `addLogit(i, delta)`

For pruning transforms with `minKeep`, values below `1` are clamped to `1` so the sampler cannot empty the candidate set by accident.

Selection helpers:

- `pick()`
- `pickGreedy()`
- `pickIndex(i)`
- `pickToken(id)`

Session helpers:

- `recentCount()`
- `recentToken(back)`
- `log(...values)`
- `random()`
- `tokenText(id)`
- `vocabSize()`
- `contextSize()`

If `custom_sampler_debug` is enabled, `log(...values)` messages from `init()` are attached to the request-level init log section, and messages from `sample()` are attached to the corresponding generated token step.

## Current Limitations

- `custom_sampler` is not supported together with `mirostat`
- returning a token id not present in the current candidate set is an error
- custom sampler JS runs inside a small, request-local Duktape heap with a short execution timeout
- stateful native samplers like DRY and adaptive-p are not exposed to JS yet

## Examples

- `top_k_top_p.js`
- `manual_repeat_penalty.js`
- `boost_ids_from_params.js`

## Regression Tests

Run both the C++ bridge tests and Python API marshalling tests with:

```bash
make test_custom_samplers
```
