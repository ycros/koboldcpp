#include "llama.h"

#include "ggml.h"

#include "seqrep-sampler.h"

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cctype>
#include <ctime>
#include <initializer_list>
#include <map>
#include <vector>
#include <queue>
#include <random>
#include <unordered_map>

// FIXME: The UTF8 decoding stuff needs to get moved or become public or I need to find
// a different approach.
struct llama_partial_utf8 {
    uint32_t value;    // bit value so far (unshifted)
    int      n_remain; // num bytes remaining; -1 indicates invalid sequence
};
std::pair<std::vector<uint32_t>, llama_partial_utf8> decode_utf8(
        const char         * src,
        llama_partial_utf8   partial_start);


void seqrep_sampler_params_init(llama_sampler_seqrep_params * params) {
    assert(params != NULL);
    memset(params, 0, sizeof(llama_sampler_seqrep_params));
    params->last_n = 256;
    params->mid_word_scale = 0.1f;
    params->tolerance_half_step_cost = 1.0f;
    params->rewind_max_visits = 2;
    params->rewind_ban_length = 1;
}

void seqrep_sampler_params_dump(const llama_sampler_seqrep_params * params) {
    assert(params != NULL);
    printf("seqrep(last_n = %d, min_length = %zd, start_offset = %zd, presence_penalty = %.4f, length_penalty = %.4f, tolerance = %.4f, mid_word_scale = %.4f, tolerance_match_credit = %.4f, tolerance_half_step_cost = %.4f, rewind_min_length = %zd, rewind_seek_word_boundary = %zd, flags = %d)\n",
        params->last_n, params->min_length, params->start_offset, params->presence_penalty,
        params->length_penalty, params->tolerance, params->mid_word_scale, params->tolerance_match_credit,
        params->tolerance_half_step_cost, params->rewind_min_length, params->rewind_seek_word_boundary,
        params->flags);
}

void seqrep_sampler_help() {
    llama_sampler_seqrep_params p;

    seqrep_sampler_params_init(&p);
    printf("==== Sequence Repetition Sampler Help ====\n\n");
    printf("  The sequence repetition sampler takes a configuration string in the format:\n");
    printf("  arg1:arg2:argN\n");
    printf("  A colon separated argument can be a key value pair like xyz=1 or flag like xyz\n");
    printf("\n- Available key/value arguments\n");
    printf("  * repetition_mode=REPEAT_PENALTY\n    emulates the repetition penalty sampler. warning: 1.0 disables penalties since this preset enables flag_divide_by_penalty. using 0.0 is probably not what you want\n");
    printf("  * presence_mode=PRESENCE_PENALTY\n    emulates the presence penalty sampler\n");
    printf("  * frequency_mode=FREQUENCY_PENALTY\n    Emulates the repetition penalty sampler\n");
    printf("  * rewind_mode\n    Enables rewind mode and sets skip_ws_punct, require_wbound and persist_require_wbound flags\n");
    printf("  * last_n\n    last n tokens to consider for sequence penalizing (default: %d, 0 = disabled, -1 = ctx_size)\n", p.last_n);
    printf("  * min_length\n    minimum matching sequence length (default: %zd, < 2 = disabled)\n", p.min_length);
    printf("  * presence_penalty\n    presence penalty for tokens that can continue a sequence (default: %f, 0.0 = disabled)\n", p.presence_penalty);
    printf("  * length_penalty\n    penalty for tokens that can continue a sequence, multiplied by length (default: %f, 0.0 = disabled)\n", p.length_penalty);
    printf("  * tolerance\n    tolerance for fuzzy matching sequences (default: %f, 0 = disabled)\n", p.tolerance);
    printf("  * mid_word_scale\n    scale penalty when for mid-word tokens. 1.0 would mean apply the full penalty (default: %f, 1.0 = disabled)\n", p.mid_word_scale);
    printf("  * tolerance_match_credit\n    credit tolerance on matched tokens (default: %f, 0.0 = disabled)\n", p.tolerance_match_credit);
    printf("  * tolerance_half_step_cost\n    advanced option to adjust tolerance cost for failed matches within a half step of a match (default: %f, 1.0 = normal)\n", p.tolerance_half_step_cost);
    printf("  * start_offset\n    advanced option to set the initial offset for pattern matching. This is relative to the start of last_n. For example, you can set last_n=-1:start_offset=NUM_PROMPT_TOKENS to limit sequence matching to the prompt (default: %zu)\n", p.start_offset);
    printf("  * rewind_min_length\n    Ensure the sequence is at least the specified length in rewind mode after whitespace skipping and other modifications (default: %zu)\n", p.rewind_min_length);
    printf("  * rewind_max_visits\n    A position is limited to the specified number of rewinds. When the limit is exceeded, future rewinds cannot target it or earlier tokens. (default: %zu)\n", p.rewind_max_visits);
    printf("  * rewind_persist_bans\n    Tokens banned by rewind remain banned for an additional number of positions equal to the value. i.e. setting this to 1 would mean the token is banned for 2 positions. (default: %zu)\n", p.rewind_persist_bans);
    printf("  * rewind_ban_length\n    Number of tokens from the sequence to ban when rewinding. (default: %zu)\n", p.rewind_ban_length);
    printf("\n- Available flags arguments (currently all default to disabled)\n");
    printf("  * flag_immediate_wildcard\n    when tolerance is consumed, by default it doesn't count as a match until a real match is found\n");
    printf("  * flag_tolerance_no_consecutive\n    do not allow using tolerance consecutively\n");
    printf("  * flag_tolerance_no_first\n    do not allow using tolerance before the first match\n");
    printf("  * flag_tolerance_cap_initial\n    only meaningful with match credit, prevents match credit adjusting tolerance higher than the initial value\n");
    printf("  * flag_penalize_length_max_seen\n    when applying length_penalty, use the maximum seen sequence length rather than the total length of seen sequences\n");
    printf("  * flag_divide_by_penalty\n    divide the logit when applying a penalty rather than subtracting it. warning: when this flag is enabled, 1.0 disables penalties not 0.0. 0.0 is probably not what you want\n");
    printf("  * flag_rewind_mode\n    Rather than penalizing tokens that can continue a sequence, this mode will actually rewind and ban the token that _starts_ the sequence. Note: Requires support in the caller. Also only applies when min_length is at least 2. Most other settings will be ignored in this mode\n");
    printf("  * flag_rewind_skip_ws_punct\n    When rewinding, skip past whitespace and punctuation. For example, if the matched sequence was \"<NL>'hello\" then we will rewind to the token starting with 'h' and ban it.\n");
    printf("  * flag_rewind_use_shortest_match\n    Rewind to the shortest matching sequence of at least min_length rather than the longest. Only meaningful when multiple rewind seqrep samplers are defined.\n");
    printf("  * flag_rewind_require_wbound\n    Rewinding requires a word boundary. Only has an effect when rewind_seek_word_boundary isn't 0.\n");
    printf("  * flag_rewind_persist_require_wbound\n    Persisted bans are only applied if at a word bound.\n");
    printf("\n- Examples:\n");
    printf("  * repetition_mode=1.2:last_n=32\n    same as --repeat-last-n 32 --repeat-penalty 1.2\n");
    printf("  * presence_mode=.2:last_n=32\n    same as --repeat-last-n 32 --presence-penalty .2\n");
    printf("  * frequency_mode=.2:last_n=32\n    same as --repeat-last-n 32 --frequency-penalty .2\n");
    printf("  * min_length=3:tolerance=1:length_penalty=.2:last_n=-1\n    match repeated sequences of at least 3 tokens within the entire context and apply a penalty of 0.2*total_length to the token that would continue the sequence. allow one non-matching token in matched sequences.\n");
}

bool seqrep_sampler_params_parse(const char * s, llama_sampler_seqrep_params * params) {
    assert(params != NULL);
    assert(s != NULL);
    size_t offset = 0;
    std::string sparams = s;
    size_t slen = sparams.size();

    while (offset < slen) {
        size_t argsep = sparams.find_first_of(':', offset);
        std::string argchunk;
        if (argsep == std::string::npos) {
            argchunk = sparams.substr(offset);
        } else if (argsep > offset) {
            argchunk = sparams.substr(offset, argsep - offset);
        }
        std::string argval;
        size_t valsep = argchunk.find_first_of('=');
        if (valsep != std::string::npos && valsep < argchunk.size()) {
            argval = argchunk.substr(valsep + 1);
            argchunk.resize(valsep);
        }
        if (argchunk.empty() && argval.empty()) {
            // pass
        } else if (argchunk == "repetition_mode") {
            params->last_n = 64;
            params->min_length = 1;
            params->mid_word_scale = 1.0f;
            params->flags = LLAMA_SEQREP_DIVIDE_BY_PENALTY;
            params->length_penalty = 1.0f;
            params->presence_penalty = argval.empty() ? 1.1f : std::atof(argval.c_str());
        } else if (argchunk == "presence_mode") {
            params->last_n = 64;
            params->min_length = 1;
            params->mid_word_scale = 1.0f;
            params->flags = 0;
            params->length_penalty = 0.0f;
            params->presence_penalty = std::atof(argval.c_str());
        } else if (argchunk == "frequency_mode") {
            params->last_n = 64;
            params->min_length = 1;
            params->mid_word_scale = 1.0f;
            params->flags = 0;
            params->length_penalty = std::atof(argval.c_str());
            params->presence_penalty = 0.0f;
        } else if (argchunk == "rewind_mode") {
            params->flags = LLAMA_SEQREP_REWIND_REQUIRE_WBOUND
                | LLAMA_SEQREP_REWIND_PERSIST_REQUIRE_WBOUND
                | LLAMA_SEQREP_REWIND_SKIP_WS_PUNCT
                | LLAMA_SEQREP_REWIND_MODE;
        } else if (argchunk == "flag_immediate_wildcard") {
            params->flags |= LLAMA_SEQREP_IMMEDIATE_WILDCARD;
        } else if (argchunk == "flag_tolerance_no_consecutive") {
            params->flags |= LLAMA_SEQREP_TOLERANCE_NO_CONSECUTIVE;
        } else if (argchunk == "flag_tolerance_no_first") {
            params->flags |= LLAMA_SEQREP_TOLERANCE_NO_FIRST;
        } else if (argchunk == "flag_tolerance_cap_initial") {
            params->flags |= LLAMA_SEQREP_TOLERANCE_CAP_INITIAL;
        } else if (argchunk == "flag_penalize_length_max_seen") {
            params->flags |= LLAMA_SEQREP_PENALIZE_LENGTH_MAX_SEEN;
        } else if (argchunk == "flag_divide_by_penalty") {
            params->flags |= LLAMA_SEQREP_DIVIDE_BY_PENALTY;
        } else if (argchunk == "flag_rewind_mode") {
            params->flags |= LLAMA_SEQREP_REWIND_MODE;
        } else if (argchunk == "flag_rewind_skip_ws_punct") {
            params->flags |= LLAMA_SEQREP_REWIND_SKIP_WS_PUNCT | LLAMA_SEQREP_REWIND_MODE;
        } else if (argchunk == "flag_rewind_use_shortest_match") {
            params->flags |= LLAMA_SEQREP_REWIND_USE_SHORTEST_MATCH | LLAMA_SEQREP_REWIND_MODE;
        } else if (argchunk == "flag_rewind_require_wbound") {
            params->flags |= LLAMA_SEQREP_REWIND_REQUIRE_WBOUND | LLAMA_SEQREP_REWIND_MODE;
        } else if (argchunk == "flag_rewind_persist_require_wbound") {
            params->flags |= LLAMA_SEQREP_REWIND_PERSIST_REQUIRE_WBOUND | LLAMA_SEQREP_REWIND_MODE;
        } else if (argchunk == "min_length") {
            params->min_length = std::atoi(argval.c_str());
        } else if (argchunk == "rewind_min_length") {
            params->rewind_min_length = std::atoi(argval.c_str());
        } else if (argchunk == "rewind_seek_word_boundary") {
            params->rewind_seek_word_boundary = std::atoi(argval.c_str());
        } else if (argchunk == "rewind_max_visits") {
            params->rewind_max_visits = std::atoi(argval.c_str());
        } else if (argchunk == "rewind_persist_bans") {
            params->rewind_persist_bans = std::atoi(argval.c_str());
        } else if (argchunk == "rewind_ban_length") {
            params->rewind_ban_length = std::atoi(argval.c_str());
        } else if (argchunk == "start_offset") {
            params->start_offset = std::atoi(argval.c_str());
        } else if (argchunk == "last_n") {
            params->last_n = std::atoi(argval.c_str());
        } else if (argchunk == "tolerance") {
            params->tolerance = std::atof(argval.c_str());
        } else if (argchunk == "presence_penalty") {
            params->presence_penalty = std::atof(argval.c_str());
        } else if (argchunk == "length_penalty") {
            params->length_penalty = std::atof(argval.c_str());
        } else if (argchunk == "mid_word_scale") {
            params->mid_word_scale = std::atof(argval.c_str());
        } else if (argchunk == "tolerance_match_credit") {
            params->tolerance_match_credit = std::atof(argval.c_str());
        } else if (argchunk == "tolerance_half_step_cost") {
            params->tolerance_half_step_cost = std::atof(argval.c_str());
        } else {
            fprintf(stderr, "seqrep: Bad argument [%s]=[%s]!\n", argchunk.c_str(), argval.c_str());
            return false;
        }
        if (argsep != std::string::npos) {
            offset = argsep + 1;
        } else {
            break;
        }
    }
    return true;
}


// Internal helper function for sequence matching.
static size_t llama_seqrep_find_match(const llama_token * last_tokens_p, const size_t last_tokens_size, const int initial_offset, const llama_sampler_seqrep_params *params) {

    if (params->min_length < 2 || last_tokens_size <= params->min_length
            || size_t(initial_offset) < params->min_length - 1) {
        return 0;
    }

    int flags = params->flags;
    float tolerance = params->tolerance;
    int tail_offset = last_tokens_size - 1, offset = initial_offset;

    // If offset == tail_offset every step will match, obviously.
    // if tail_offset < offset... that would just be really weird.
    if (offset >= tail_offset) {
        return 0;
    }
    int matches = 0, pending_matches = 0;
    bool last_matched = true;

    // tail_offset must always be > offset so no need to check that condition here.
    while (offset >= 0) {
        if (last_tokens_p[offset] == last_tokens_p[tail_offset]) {
            offset--;
            tail_offset--;
            matches += 1 + pending_matches;
            pending_matches = 0;
            tolerance += params->tolerance_match_credit;
            if ((flags & LLAMA_SEQREP_TOLERANCE_CAP_INITIAL) != 0) {
                tolerance = std::min(params->tolerance, tolerance);
            }
            last_matched = true;
            continue;
        }

        if (offset == 0
                || ((flags & LLAMA_SEQREP_TOLERANCE_NO_FIRST) != 0 && offset == initial_offset)
                || ((flags & LLAMA_SEQREP_TOLERANCE_NO_CONSECUTIVE) != 0 && last_matched == false)) {
            break;
        }
        last_matched = false;

        if (tolerance >= params->tolerance_half_step_cost) {
            if (offset > 0 && last_tokens_p[offset - 1] == last_tokens_p[tail_offset]) {
                offset--;
                tolerance -= params->tolerance_half_step_cost;
                continue;
            } else if (tail_offset > offset + 1 && last_tokens_p[offset] == last_tokens_p[tail_offset - 1]) {
                // The first part of the condition above is to preserve the invariant tail_offset > offset
                tail_offset--;
                tolerance -= params->tolerance_half_step_cost;
                continue;
            }
        }
        if (tolerance < 1.0f) {
            break;
        }
        tolerance -= 1.0f;

        if ((flags & LLAMA_SEQREP_IMMEDIATE_WILDCARD) != 0) {
            matches++;
        } else {
            // A tolerance charge can count as a match, but only if we can find a
            // real match before the search is terminated.
            pending_matches++;
        }

        offset--;
        tail_offset--;
    }
    return matches;
}

// Internal helper macro for sequence matching, used to determine if a CP is a word boundary.
// 0x2000 through is 0x206f is standard unicode punctuation. The CP is considered a word bound
// if it falls in that range but is _not_ RIGHT SINGLE QUOTATION MARK (0x2019) or if it's in
// low ASCII range, not alphanumeric and also not a single quote.
#define LLAMA_SEQREP_IS_WBOUND(cp) ( (cp < 127 && cp != 39 && !std::isalnum((int(cp)))) || (cp != 0x2019 && cp >= 0x2000 && cp <= 0x206f) )


// Helper function for sequence matching.
// Bit 1 set indicates token is a word boundary. NL, " blah", "," - word boundary. "blah", "blah:" - not a word boundary.
// Bit 2 set indicates token ends on a word boundary. NL, "blah:", "blah " - ends on word boundary. " blah", "blah" - doesn't end on word boundary.
// Bit 3 set indicates all codepoints in the character count as boundary.
// Errata: Special cases apostrophe and only has partial support for unicode punctuation as word boundaries.
int llama_seqrep_check_word(struct llama_context * ctx, const llama_token token, std::vector<char> & buf) {
    if (token == llama_token_bos(ctx) || token == llama_token_eos(ctx) || token == llama_token_nl(ctx)) {
        // BOS, EOS, NL are always a boundary.
        return SEQREP_CW_START_IS_WBOUND | SEQREP_CW_END_IS_WBOUND | SEQREP_CW_ALL_WS_PUNCT;
    }

    if (buf.size() < 128) {
        buf.resize(128);
    }
    int n_tokens = llama_token_to_piece(ctx, token, buf.data(), buf.size() - 1);
    if (n_tokens < 0) {
        buf.resize(size_t(-n_tokens) + 128);
        const int check = llama_token_to_piece(ctx, token, buf.data(), buf.size() - 1);
        GGML_ASSERT(check == -n_tokens);
        n_tokens = check;
    }
    buf[n_tokens] = 0;
    auto decoded = decode_utf8(buf.data(), llama_partial_utf8{ 0, 0 });
    std::vector<uint32_t> & token_cps = decoded.first;
    const size_t token_cps_len = token_cps.size();

    if (token_cps_len < 2) {
        // < 2 here because decode_utf8 terminates with a 0 entry.
        if (decoded.second.n_remain != 0) {
            // Partial or invalid UTF8 sequence. Guess this is a boundary on both sides?
            return SEQREP_CW_START_IS_WBOUND | SEQREP_CW_END_IS_WBOUND;
        }
        // Token has no codepoints, can't be a boundary.
        return 0;
    }

    token_cps.resize(token_cps_len - 1);
    int result = SEQREP_CW_ALL_WS_PUNCT;

    for (size_t i = 0; i < token_cps_len - 1; i++) {
        if (!LLAMA_SEQREP_IS_WBOUND(token_cps[i])) {
            result &= ~SEQREP_CW_ALL_WS_PUNCT;
            continue;
        }
        if (i == 0)
            result |= SEQREP_CW_START_IS_WBOUND;
        if (i == token_cps_len - 2)
            result |= SEQREP_CW_END_IS_WBOUND;
    }
    return result;
}


size_t llama_sample_seqrep_penalty(struct llama_context * ctx, llama_token_data_array * candidates, const llama_token * last_tokens_p, size_t last_tokens_size, const llama_sampler_seqrep_params * params) {
    const size_t min_length = params->min_length;
    const int flags = params->flags;

    if (params->last_n == 0 || params->min_length < 1) {
        return 0;
    } else if (params->last_n > 0) {
        size_t window_offset = last_tokens_size - std::min(size_t(params->last_n), last_tokens_size);

        last_tokens_size -= window_offset;
        last_tokens_p += window_offset;
    }

    if (last_tokens_size < 1 || (min_length > 1 && last_tokens_size <= min_length)) {
        return 0;
    } else if ((params->flags & LLAMA_SEQREP_REWIND_MODE) == 0) {
        const float disabled = ((params->flags & LLAMA_SEQREP_DIVIDE_BY_PENALTY) == 0) ? 0.0f : 1.0f;
        if (params->presence_penalty == disabled && params->length_penalty == disabled) {
            return 0;
        }
    }

    if (params->mid_word_scale != 1.0f || (params->flags & LLAMA_SEQREP_REWIND_SKIP_WS_PUNCT) != 0) {
        // Only need ctx when mid_word_scale or REWIND_SKIP_WS_PUNCT flag is in effect.
        assert(ctx);
    }

    // const int64_t t_start_sample_us = ggml_time_us();

    // This will hold a map of token ids that can continue the sequence with its sequence length.
    std::unordered_map<llama_token, size_t> penalize_tokens;

    if (min_length > 1) {
        // Normal sequence matching mode.
        size_t start_offset       = params->start_offset;
        size_t max_matched_length = 0;
        size_t min_matched_length = last_tokens_size;

        if (start_offset == 0 || start_offset >= last_tokens_size - 1) {
            start_offset = last_tokens_size - 2;
        }
        for (size_t offset = start_offset; offset >= min_length - 1; offset--) {
            const size_t matched_length =
                llama_seqrep_find_match(last_tokens_p, last_tokens_size, offset, params);
            if (matched_length < min_length) {
                continue;
            }
            max_matched_length = std::max(max_matched_length, matched_length);
            min_matched_length = std::min(min_matched_length, matched_length);

            // The token one past where we started trying to match is the one that could continue
            // the previously observed sequence.
            llama_token penalize_token = last_tokens_p[offset + 1];

            auto pt_iter = penalize_tokens.find(penalize_token);
            if (pt_iter == penalize_tokens.end() || (flags & LLAMA_SEQREP_PENALIZE_LENGTH_MAX_SEEN) == 0) {
                penalize_tokens[penalize_token] += matched_length;
            } else {
                penalize_tokens[penalize_token] = std::max(pt_iter->second, matched_length);
            }
        }

        if ((flags & LLAMA_SEQREP_REWIND_MODE) != 0) {
            size_t result = ((flags & LLAMA_SEQREP_REWIND_USE_SHORTEST_MATCH) == 0 || max_matched_length < min_length
                ? max_matched_length
                : min_matched_length);
            if (max_matched_length > 0 && (params->flags & LLAMA_SEQREP_REWIND_SKIP_WS_PUNCT) != 0) {
                std::vector<char> buf(128, 0);
                for (size_t i = last_tokens_size - result; i < last_tokens_size; i++) {
                    if ((llama_seqrep_check_word(ctx, last_tokens_p[i], buf) & SEQREP_CW_ALL_WS_PUNCT) != 0) {
                        result--;
                    } else {
                        break;
                    }
                }
            }
            return result;
        }
    } else {
        // Single token matching mode. Can emulate existing repetition, presence and frequency samplers.
        size_t start_offset = params->start_offset;
        if (start_offset == 0 || start_offset >= last_tokens_size) {
            start_offset = last_tokens_size - 1;
        }
        for (int i = int(start_offset); i >= 0; i--) {
            llama_token penalize_token = last_tokens_p[i];

            if ((flags & LLAMA_SEQREP_PENALIZE_LENGTH_MAX_SEEN) != 0) {
                penalize_tokens[penalize_token] = 1;
            } else {
                penalize_tokens[penalize_token]++;
            }
        }
    }

    std::vector<char> buf(128, 0);
    const bool ends_on_word = params->mid_word_scale == 1.0f
        || (llama_seqrep_check_word(ctx, last_tokens_p[last_tokens_size - 1], buf) & SEQREP_CW_END_IS_WBOUND) != 0;

    for (size_t i = 0; i < candidates->size; ++i) {
        auto pt_iter = penalize_tokens.find(candidates->data[i].id);
        if (pt_iter == penalize_tokens.end()) {
            continue;
        }

        const size_t count = pt_iter->second;
        const bool pt_starts_word = params->mid_word_scale == 1.0f ||
            (llama_seqrep_check_word(ctx, candidates->data[i].id, buf) & SEQREP_CW_START_IS_WBOUND) != 0;
        float penalty_scale = ends_on_word || pt_starts_word ? 1.0f : params->mid_word_scale;
        float logit = candidates->data[i].logit;

        if ((flags & LLAMA_SEQREP_DIVIDE_BY_PENALTY) == 0) {
            float penalty =
                ( float(count) * params->length_penalty
                + float(count > 0) * params->presence_penalty );
            logit -= penalty * penalty_scale;
        } else {
            // This looks complicated. The point is to scale be able to scale penalties like
            // 1.2. For example, suppose length penalty is 1.2 and length is 3. 1.2 * 3 = 3.6
            // would be ridiculous. What we actually want is more like 1.6.
            // An alternative approach would be to iteratively apply the scale.
            // 10.0 / 1.6 == 6.25, however ((10.0 / 1.2) / 1.2) / 1.2 == 5.787
            float penalty =
                ( (float(count) * (params->length_penalty - 1.0f))
                + (float(count > 0) * (params->presence_penalty - 1.0f)) ) * penalty_scale
                + 1.0f;
            if (logit <= 0) {
                logit *= penalty;
            } else if (penalty != 0.0f) {
                logit /= penalty;
            } else {
                // Should we do something else here if penalty == 0?
                // This is consistent with logit * 0.0 at least.
                logit = 0.0f;
            }
        }
        candidates->data[i].logit = logit;
    }

    candidates->sorted = false;

    // FIXME: Find a way to set stuff in ctx
    // if (ctx) {
    //     ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    // }
    return 0;
}

seqrep_logit_info::seqrep_logit_info(llama_context * ctx, const size_t k)
  : n_vocab(llama_n_vocab(ctx))
  , token_data(top_k(llama_get_logits(ctx), k))
  { }

const std::vector<llama_token_data> & seqrep_logit_info:: get_token_data(void) {
    return token_data;
}

llama_token_data seqrep_logit_info::get_token_id(const llama_token token_id) const {
    for (const llama_token_data & td : token_data) {
        if (td.id == token_id)
            return td;
    }
    return {-1, 0, 0};
}

void seqrep_logit_info::rebuild(llama_context *ctx, const size_t k) {
    token_data = top_k(llama_get_logits(ctx), k);
}

void seqrep_logit_info::populate_logits(float * logits) {
    const float neginf = std::numeric_limits<float>::infinity() * -1;
    for (int i = 0; i < n_vocab; i++) {
        logits[i] = neginf;
    }
    for (const llama_token_data & td : token_data) {
        logits[td.id] = td.logit;
    }
}

// Yoinked from beam search code.
// Return top k token_data by logit.
std::vector<llama_token_data> seqrep_logit_info::top_k(const float * const logits, const size_t k) {
    std::vector<llama_token_data> min_heap;  // min-heap by logit
    const llama_token k_min = std::min(static_cast<llama_token>(k), n_vocab);
    min_heap.reserve(k_min);
    constexpr auto p = std::numeric_limits<float>::quiet_NaN();  // never used
    for (llama_token token_id = 0 ; token_id < k_min ; ++token_id) {
        const llama_token_data td = {token_id, logits[token_id], p};
        min_heap.push_back(td);
    }
    auto comp = [](const llama_token_data & a, const llama_token_data & b) { return a.logit > b.logit; };
    std::make_heap(min_heap.begin(), min_heap.end(), comp);
    for (llama_token token_id = k_min ; token_id < n_vocab ; ++token_id) {
        if (min_heap.front().logit < logits[token_id]) {
            std::pop_heap(min_heap.begin(), min_heap.end(), comp);
            min_heap.back().id = token_id;
            min_heap.back().logit = logits[token_id];
            std::push_heap(min_heap.begin(), min_heap.end(), comp);
        }
    }
    return min_heap;
}


seqrep_rewind_state::seqrep_rewind_state(const size_t n_vocab, const size_t n_ctx, const size_t k = 2000, const size_t high_water_mark = 0)
  : n_vocab(n_vocab)
  , n_ctx(n_ctx)
  , k(k)
  , high_water_mark(high_water_mark)
{
    logit_slots.reserve(n_ctx);
    rewind_slots.resize(n_ctx);
}

void seqrep_rewind_state::set_logits_slot(llama_context * ctx, const size_t orig_idx) {
    // printf("\n-- %zu, %zu, %zu\n", orig_idx, high_water_mark, logit_slots.size());
    GGML_ASSERT(orig_idx >= high_water_mark);
    const size_t idx = orig_idx - high_water_mark;
    GGML_ASSERT(idx <= logit_slots.size());
    if (idx == logit_slots.size()) {
        logit_slots.emplace_back(ctx, k);
    } else {
        logit_slots[idx].rebuild(ctx, k);
    }
}

struct seqrep_rewind_slot & seqrep_rewind_state::get_rewind_slot(const size_t orig_idx) {
    GGML_ASSERT(orig_idx >= high_water_mark);
    const size_t idx = orig_idx - high_water_mark;
    GGML_ASSERT(idx <= rewind_slots.size());
    return rewind_slots[idx];
}

void seqrep_rewind_state::populate_logits(llama_context * ctx, const size_t orig_idx) {
    GGML_ASSERT(orig_idx >= high_water_mark);
    const size_t idx = orig_idx - high_water_mark;
    logit_slots[idx].populate_logits(llama_get_logits(ctx));
}

void seqrep_rewind_state::set_high_water_mark(const size_t val) {
    high_water_mark = val;
}

static size_t llama_seqrep_check_rewind_internal(
        struct llama_context * ctx,
        const std::vector<llama_token> & last_tokens,
        const std::vector<llama_sampler_seqrep_params> & params_list,
        const llama_sampler_seqrep_params & merged_params,
        size_t * high_water_mark) {
    const size_t last_tokens_size = last_tokens.size();

    size_t min_matched_len = 0, max_matched_len = 0;

    for (auto & sr_params : params_list) {
        if ((sr_params.flags & LLAMA_SEQREP_REWIND_MODE) == 0) continue;
        const size_t matched_len = llama_sample_seqrep_penalty(ctx, NULL, last_tokens.data(), last_tokens_size, &sr_params);
        max_matched_len = std::max(max_matched_len, matched_len);
        min_matched_len = min_matched_len == 0
            ? matched_len
            : std::min(min_matched_len, matched_len);
    }
    if (max_matched_len < 2 || max_matched_len >= last_tokens_size) {
        return 0;
    }

    const size_t matched_len = ((merged_params.flags & LLAMA_SEQREP_REWIND_USE_SHORTEST_MATCH) == 0
        ? max_matched_len
        : min_matched_len);
    size_t idx = last_tokens_size - matched_len;

    if (idx < *high_water_mark) {
        // if ((merged_params.flags & LLAMA_SEQREP_REWIND_NO_HW_MARK_FIXUP) != 0) {
        //     return 0;
        // }
        if (*high_water_mark >= last_tokens_size - 2) {
            return 0;
        }
        idx = *high_water_mark;
    }
    if (merged_params.rewind_seek_word_boundary > 0) {
        std::vector<char> buf(128, 0);
        const size_t orig_idx = idx;
        bool found_idx = false;

        for (size_t steps = merged_params.rewind_seek_word_boundary + 1; idx >= *high_water_mark && steps > 0; idx--, steps--) {
            if ((llama_seqrep_check_word(ctx, last_tokens[idx], buf) & SEQREP_CW_START_IS_WBOUND) != 0
                || (llama_seqrep_check_word(ctx, last_tokens[idx - 1], buf) & SEQREP_CW_END_IS_WBOUND) != 0) {
                found_idx = true;
                break;
            }
        }
        if (!found_idx) {
            idx = orig_idx;
            for (size_t steps = merged_params.rewind_seek_word_boundary + 1; idx < last_tokens_size && steps > 0; idx++, steps--) {
                if ((llama_seqrep_check_word(ctx, last_tokens[idx], buf) & SEQREP_CW_START_IS_WBOUND) != 0
                    || (llama_seqrep_check_word(ctx, last_tokens[idx - 1], buf) & SEQREP_CW_END_IS_WBOUND) != 0) {
                    found_idx = true;
                    break;
                }
            }
            if (!found_idx || last_tokens_size - idx < merged_params.rewind_min_length) {
                if ((merged_params.flags & LLAMA_SEQREP_REWIND_REQUIRE_WBOUND) != 0) {
                    return 0;
                }
                idx = orig_idx;
            }
        }
    }

    const size_t rewind_distance = last_tokens.size() - idx;
    if (merged_params.rewind_min_length != 0 && rewind_distance < merged_params.rewind_min_length) {
        return 0;
    }

    return rewind_distance;
}

size_t llama_seqrep_handle_rewind(
        struct llama_context * ctx,
        struct seqrep_rewind_state & rewind_state,
        const std::vector<llama_token> & last_tokens,
        const size_t prompt_tokens_size,
        const std::vector<llama_sampler_seqrep_params> & params_list,
        size_t * high_water_mark) {
    const size_t last_tokens_size = last_tokens.size();

    if (last_tokens_size - prompt_tokens_size < 3) {
        return 0;
    }

    // llama_sampler_seqrep_params merged_params = llama_seqrep_merge_params(params_list, LLAMA_SEQREP_REWIND_MODE, 0);
    // printf("\nparams_list.size() = %zu\n", params_list.size());
    // seqrep_sampler_params_dump(&merged_params);
    GGML_ASSERT(params_list.size() > 0);
    llama_sampler_seqrep_params merged_params = params_list[0];
    // seqrep_sampler_params_dump(&merged_params);
    size_t rewind_distance = 0;
    size_t idx = last_tokens.size();
    std::vector<char> rewind_token_text_buf(128, 0);

    while (true) {
        rewind_distance = llama_seqrep_check_rewind_internal(
            ctx, last_tokens, params_list, merged_params, high_water_mark );

        if (rewind_distance == 0) break;

        idx = last_tokens.size() - rewind_distance;

        const size_t ban_length = std::min(rewind_distance, merged_params.rewind_ban_length);
        struct seqrep_rewind_slot &rw_slot = rewind_state.get_rewind_slot(idx);
        const bool at_wbound = idx == 0 ||
            (llama_seqrep_check_word(ctx, last_tokens[idx - 1], rewind_token_text_buf) & SEQREP_CW_END_IS_WBOUND) != 0;

        for (size_t i = idx; i < idx + ban_length; i++) {
            const llama_token penalize_token = last_tokens[i];
            if (i > idx && !at_wbound && (llama_seqrep_check_word(ctx, penalize_token, rewind_token_text_buf) & SEQREP_CW_START_IS_WBOUND) == 0) {
                continue;
            }
            if (std::find(rw_slot.tokens.begin(), rw_slot.tokens.end(), penalize_token) == rw_slot.tokens.end()) {
                rw_slot.tokens.push_back(penalize_token);
            }
        }

        if (++rw_slot.count >= merged_params.rewind_max_visits) {
            // This slot already hit max visits so we can set the HWM to the index one past it.
            *high_water_mark = idx + 1;
        }

        // last_tokens.resize(idx);
        GGML_ASSERT(idx > 0);
        rewind_state.populate_logits(ctx, idx - 1);

        break;
    }

    float * logits = llama_get_logits(ctx);
    const float neg_infinity = std::numeric_limits<float>::infinity() * -1;
    const size_t target_idx = idx;
    const bool at_wbound = target_idx == 0 ||
        (llama_seqrep_check_word(ctx, last_tokens[target_idx - 1], rewind_token_text_buf) & SEQREP_CW_END_IS_WBOUND) != 0;
    const bool persist_require_wbound = (merged_params.flags & LLAMA_SEQREP_REWIND_PERSIST_REQUIRE_WBOUND) != 0;
    const size_t persist_count = std::min(prompt_tokens_size - target_idx, merged_params.rewind_persist_bans);

    for (size_t i = target_idx - persist_count; i <= target_idx; i++) {
        // FIXME: There's a better way to calculate this.
        if (i <= prompt_tokens_size) {
            continue;
        }
        if (persist_require_wbound && i != target_idx && !at_wbound) {
            // We don't apply this logic when i == target_idx because the previous
            // checks should have taken it into account when the specific ban was applied
            // initially.
            continue;
        }
        for (const llama_token token_id : rewind_state.get_rewind_slot(i).tokens) {
           logits[token_id] = neg_infinity;
        }
    }

    return rewind_distance;
}


// Note: Doesn't merge presence or length penalties because of the divide_by_penalty flag.
struct llama_sampler_seqrep_params llama_seqrep_merge_params(const std::vector<llama_sampler_seqrep_params> & params_list, const int and_flags, const int not_flags) {
    struct llama_sampler_seqrep_params result;
    memset(&result, 0, sizeof(struct llama_sampler_seqrep_params));

    for (auto & sr_params : params_list) {
        if ((sr_params.flags & and_flags) != and_flags || (sr_params.flags & not_flags) != 0) {
            continue;
        }
        result.flags |= sr_params.flags;
        result.min_length = std::max(result.min_length, sr_params.min_length);
        result.last_n = sr_params.last_n < 0 || result.last_n < 0
            ? -1
            : std::max(result.last_n, sr_params.last_n);
        result.tolerance = std::max(result.tolerance, sr_params.tolerance);
        result.tolerance_half_step_cost = result.tolerance_half_step_cost == 0.0f
            ? sr_params.tolerance_half_step_cost
            : std::min(result.tolerance_half_step_cost, sr_params.tolerance_half_step_cost);
        result.mid_word_scale = std::max(result.mid_word_scale, sr_params.mid_word_scale);
        result.tolerance_match_credit = std::max(result.tolerance_match_credit, sr_params.tolerance_match_credit);
        result.rewind_min_length = std::max(result.rewind_min_length, sr_params.rewind_min_length);
        result.rewind_seek_word_boundary = std::max(result.rewind_seek_word_boundary, sr_params.rewind_seek_word_boundary);
        result.rewind_max_visits = std::max(result.rewind_max_visits, sr_params.rewind_max_visits);
        result.rewind_persist_bans = std::max(result.rewind_persist_bans, sr_params.rewind_persist_bans);
        result.rewind_ban_length = std::max(result.rewind_ban_length, sr_params.rewind_ban_length);
    }
    return result;
}
