/**
 * Result returned from `sample(s)`.
 *
 * - Return `undefined` to let KoboldCpp continue running the rest of the native sampler pipeline.
 * - Return a token id to immediately select that token for the current decode step.
 */
type KoboldSamplerResult = number | void;

/**
 * One token alternative from a logprob list.
 *
 * This type is mainly useful in helper code you write around sampler traces.
 */
interface KoboldSamplerAlternative {
  /** Decoded token text as a JavaScript string. */
  token: string;

  /** Numeric token id in the current model vocabulary. */
  token_id: number;

  /** Natural-log probability for this token. */
  logprob: number;

  /** UTF-8 bytes for the decoded token text. */
  bytes: number[];
}

/**
 * Native-backed API object passed into `init(params, s)` and `sample(s)`.
 *
 * The object exposes the current candidate set in-place. Most mutating methods return `s`
 * so you can chain transforms without allocating extra JavaScript objects.
 */
interface KoboldSamplerApi {
  /**
   * Returns the number of candidates currently retained in the active set.
   */
  size(): number;

  /**
   * Returns the token id at the given candidate index.
   *
   * @param index Zero-based candidate index.
   */
  id(index: number): number;

  /**
   * Returns the current logit for the candidate at the given index.
   *
   * @param index Zero-based candidate index.
   */
  logit(index: number): number;

  /**
   * Returns the current probability for the candidate at the given index.
   *
   * If probabilities are stale, KoboldCpp recomputes them before returning the value.
   *
   * @param index Zero-based candidate index.
   */
  prob(index: number): number;

  /**
   * Replaces the candidate logit at `index`.
   *
   * @param index Zero-based candidate index.
   * @param value New logit value.
   * @returns The same sampler API object for chaining.
   */
  setLogit(index: number, value: number): KoboldSamplerApi;

  /**
   * Adds `delta` to the candidate logit at `index`.
   *
   * @param index Zero-based candidate index.
   * @param delta Amount to add to the existing logit.
   * @returns The same sampler API object for chaining.
   */
  addLogit(index: number, delta: number): KoboldSamplerApi;

  /**
   * Applies softmax to the current candidate set.
   *
   * @returns The same sampler API object for chaining.
   */
  softmax(): KoboldSamplerApi;

  /**
   * Applies top-k pruning.
   *
   * @param k Number of strongest candidates to keep.
   * @returns The same sampler API object for chaining.
   */
  topK(k: number): KoboldSamplerApi;

  /**
   * Applies top-a pruning.
   *
   * Candidates below `a * best_prob^2` are removed after softmax.
   *
   * @param a Top-a threshold.
   * @param minKeep Minimum number of candidates to keep.
   * @returns The same sampler API object for chaining.
   */
  topA(a: number, minKeep?: number): KoboldSamplerApi;

  /**
   * Applies top-p pruning.
   *
   * @param p Cumulative probability threshold.
   * @param minKeep Minimum number of candidates to keep.
   * @returns The same sampler API object for chaining.
   */
  topP(p: number, minKeep?: number): KoboldSamplerApi;

  /**
   * Applies min-p pruning.
   *
   * @param p Relative probability floor.
   * @param minKeep Minimum number of candidates to keep.
   * @returns The same sampler API object for chaining.
   */
  minP(p: number, minKeep?: number): KoboldSamplerApi;

  /**
   * Applies tail-free sampling.
   *
   * @param z Tail-free cutoff value.
   * @param minKeep Minimum number of candidates to keep.
   * @returns The same sampler API object for chaining.
   */
  tailFree(z: number, minKeep?: number): KoboldSamplerApi;

  /**
   * Applies typical sampling.
   *
   * @param p Typical sampling threshold.
   * @param minKeep Minimum number of candidates to keep.
   * @returns The same sampler API object for chaining.
   */
  typical(p: number, minKeep?: number): KoboldSamplerApi;

  /**
   * Applies top-n-sigma filtering.
   *
   * @param nsigma Sigma threshold.
   * @returns The same sampler API object for chaining.
   */
  topNSigma(nsigma: number): KoboldSamplerApi;

  /**
   * Applies temperature and optional smoothing.
   *
   * @param temp Base temperature.
   * @param smoothingFactor Optional smoothing factor.
   * @param smoothingCurve Optional smoothing curve.
   * @returns The same sampler API object for chaining.
   */
  temperature(temp: number, smoothingFactor?: number, smoothingCurve?: number): KoboldSamplerApi;

  /**
   * Applies entropy-based dynamic temperature.
   *
   * @param minTemp Lower temperature bound.
   * @param maxTemp Upper temperature bound.
   * @param exponent Entropy exponent.
   * @param smoothingFactor Optional smoothing factor.
   * @param smoothingCurve Optional smoothing curve.
   * @returns The same sampler API object for chaining.
   */
  entropy(minTemp: number, maxTemp: number, exponent: number, smoothingFactor?: number, smoothingCurve?: number): KoboldSamplerApi;

  /**
   * Applies the native repetition penalty transform using current session history.
   *
   * @param range Number of recent tokens to inspect.
   * @param penalty Repetition penalty value.
   * @param slope Repetition penalty slope.
   * @param presencePenalty Additional presence penalty.
   * @returns The same sampler API object for chaining.
   */
  repPenalty(range: number, penalty: number, slope: number, presencePenalty: number): KoboldSamplerApi;

  /**
   * Applies the native XTC transform.
   *
   * @param threshold XTC threshold.
   * @param probability XTC probability.
   * @returns The same sampler API object for chaining.
   */
  xtc(threshold: number, probability: number): KoboldSamplerApi;

  /**
   * Samples one token id from the current candidate set using KoboldCpp's RNG.
   */
  pick(): number;

  /**
   * Returns the token id with the strongest current logit.
   */
  pickGreedy(): number;

  /**
   * Returns the token id currently stored at the given candidate index.
   *
   * This is useful when you want to choose by index after your own scoring logic.
   *
   * @param index Zero-based candidate index.
   */
  pickIndex(index: number): number;

  /**
   * Returns `tokenId` if it is still present in the current candidate set.
   * Throws if the token is not available.
   *
   * @param tokenId Token id to validate and return.
   */
  pickToken(tokenId: number): number;

  /**
   * Returns the number of recent tokens available through `recentToken(back)`.
   */
  recentCount(): number;

  /**
   * Returns a token id from recent generation history.
   *
   * `back = 0` is the most recent token, `back = 1` is the token before that, and so on.
   *
   * @param back How far back from the current end of history to read.
   */
  recentToken(back: number): number;

  /**
   * Returns a random floating-point value in the range `[0, 1)` using the current request RNG.
   */
  random(): number;

  /**
   * Appends a debug log line for the current request.
   *
   * - When called inside `init(params, s)`, the message is attached to the request-level init log section.
   * - When called inside `sample(s)`, the message is attached to the current generated token step.
   *
   * @param values Values to stringify and join with spaces.
   * @returns The same sampler API object for chaining.
   */
  log(...values: unknown[]): KoboldSamplerApi;

  /**
   * Decodes a token id into token text using the active model tokenizer.
   *
   * @param tokenId Token id to decode.
   */
  tokenText(tokenId: number): string;

  /**
   * Returns the current vocabulary size.
   */
  vocabSize(): number;

  /**
   * Returns the current context size used for sampling.
   */
  contextSize(): number;
}

/**
 * Function signature for the hot-path sampler callback.
 *
 * Use this with JSDoc in your sampler source, for example:
 *
 * `/** @type {KoboldSamplerFunction} *\/`
 * `function sample(s) { ... }`
 */
type KoboldSamplerFunction = (s: KoboldSamplerApi) => KoboldSamplerResult;

/**
 * Function signature for the optional one-time initialization callback.
 *
 * `params` is whatever JSON value the request sent as `custom_sampler_params`.
 */
type KoboldSamplerInitFunction = (params: unknown, s: KoboldSamplerApi) => void;

/**
 * Optional one-time setup callback.
 *
 * KoboldCpp calls `init(params, s)` once per request before the first `sample(s)` call.
 * If you do not need setup work, you can omit this function entirely.
 *
 * @param params Parsed `custom_sampler_params` JSON value, if provided.
 * @param s Native-backed sampler API object.
 */
declare function init(params: unknown, s: KoboldSamplerApi): void;

/**
 * Required sampler callback.
 *
 * Return `undefined` to continue the native sampler pipeline.
 * Return a token id to immediately select that token for the current decode step.
 *
 * @param s Native-backed sampler API object for the current decode step.
 */
declare function sample(s: KoboldSamplerApi): KoboldSamplerResult;
