#pragma once

#include <stddef.h>

#include <vector>

#include "llama.h"

enum llama_sampler_seqrep_flags {
    // Tolerance counting as a match is normally delayed until an actual match.
    // Setting this flag causes it to be applied immediately.
    LLAMA_SEQREP_IMMEDIATE_WILDCARD        = 1 << 0,

    // Tolerance charges can't be used consecutively.
    LLAMA_SEQREP_TOLERANCE_NO_CONSECUTIVE  = 1 << 2,

    // Tolerance charges can't be used before the first actual match.
    LLAMA_SEQREP_TOLERANCE_NO_FIRST        = 1 << 3,

    // Caps tolerance at the initial value. Only meaningful
    // when tolerance_match_credit > 0
    LLAMA_SEQREP_TOLERANCE_CAP_INITIAL     = 1 << 4,

    // When applying the length penalty, use the length of the longest observed
    // sequence matching the token rather than the total length of
    // sequences matching the token. In other words, if we find a sequence
    // of length 3 and a sequence of length 4 continued by token 69 then
    // with this flag on we penalize based on length 4, with it off we
    // penalize based on length 7 (3 + 4).
    LLAMA_SEQREP_PENALIZE_LENGTH_MAX_SEEN  = 1 << 5,

    // Divide the logit value by the penalty rather than subtracting.
    LLAMA_SEQREP_DIVIDE_BY_PENALTY         = 1 << 6,

    // Rewind to cut off the head of sequences rather than the end.
    // Ignored when min_length < 2.
    // Since it wouldn't make sense to rewind and then let sampling pick
    // the same token again, penalty values and mid_word_scale have no
    // effect.
    LLAMA_SEQREP_REWIND_MODE               = 1 << 7,

    // When rewinding, skip past whitespace and punctuation. For example,
    // if the matched sequence was "<NL>'hello" then we will rewind to the
    // token starting with 'h' and ban it.
    LLAMA_SEQREP_REWIND_SKIP_WS_PUNCT      = 1 << 8,

    // By default, rewinding sets a "high water mark" and that future rewinds cannot
    // pass. This is to prevent the sampler from getting stuck in a loop. This option
    // disables that behavior.
    // LLAMA_SEQREP_REWIND_NO_HW_MARK         = 1 << 9,

    // By default when rewinding, if we want to rewind to a point earlier than the HW mark
    // we rewind to the HW mark instead. This option disables that behavior and just doesn't
    // rewind when it would be to a position before the HW mark.
    // LLAMA_SEQREP_REWIND_NO_HW_MARK_FIXUP   = 1 << 10,

    // Rewind to the shortest matching sequence of at least min_length rather than the longest.
    LLAMA_SEQREP_REWIND_USE_SHORTEST_MATCH = 1 << 11,

    // Rewinding requires a word boundary. Only has an effect when rewind_seek_word_boundary isn't 0.
    LLAMA_SEQREP_REWIND_REQUIRE_WBOUND     = 1 << 12,

    // Persisted bans are only applied if at a word bound.
    LLAMA_SEQREP_REWIND_PERSIST_REQUIRE_WBOUND = 1 << 13,
};

typedef struct llama_sampler_seqrep_params {
    // The minimum length of a matching sequence of tokens. When this is < 2 then
    // the sampler works in single token mode and tolerance is ignored.
    size_t min_length;

    // Starting offset for matching against the end of the sequence. This can be used
    // to only match against sequences in the initial prompt, for example. Matching
    // starts at the offset and moves toward the beginning of the list.
    // Use 0 for penultimate token when min_length > 1 otherwise 0 for last token.
    size_t start_offset;

    // Window of last tokens to consider, starting from the end. < 0 means
    // the whole list.
    int    last_n;

    // Flags based on llama_sampler_seqrep_flags enum values ORed together.
    int    flags;

    // Tolerance for non-matching tokens in a sequence.
    float  tolerance;

    // Flat penalty applied to the token that can continue a repeated sequence.
    float  presence_penalty;

    // Scaling penalty applied to the token that can continue a repeated sequence.
    // The penalty is multiplied by the total length of sequences that are continued by this token unless
    // the PENALIZE_LENGTH_MAX_SEEN is set.
    float  length_penalty;

    // Scale for penalizing tokens from repeated sequences that aren't at/form a word boundary.
    float  mid_word_scale;

    // Tolerance credit per real match. I.E. .5 means +1 tolerance per 2 matched tokens.
    float  tolerance_match_credit;

    // Matching proceeds step by step comparing tokens starting from the tail of last_tokens
    // with tokens starting from some position closer to the list.
    // 1 2 3 9 1 2 3 7
    //       ^       ^
    // We'd have to move both positions to find a match.
    // 1 2 3 9 1 2 3
    //       ^     ^
    // We can move just the first position and immediately find a match. A half-step.
    // This can be set to a high value to disable half steps, setting it to 0 will make
    // half steps free.
    float  tolerance_half_step_cost;

    // Ensure the sequence is at least the specified length in rewind mode after
    // whitespace skipping and other modifications.
    size_t rewind_min_length;

    // When rewinding, try to find a word boundary within the specified distance, starting with tokens earlier than the rewind point.
    size_t rewind_seek_word_boundary;

    // A position is limited to the specified number of rewinds. When the limit is exceeded, future rewinds cannot target it or earlier tokens.
    size_t rewind_max_visits;

    // Tokens banned by rewind remain banned for an additional number of positions equal to the value. i.e. setting this to 1 would mean the token is banned for 2 positions.
    size_t rewind_persist_bans;

    // Number of tokens from the sequence to ban when rewinding.
    size_t rewind_ban_length;
} llama_sampler_seqrep_params;

enum seqrep_check_word_flags {
    SEQREP_CW_START_IS_WBOUND = 1 << 0,
    SEQREP_CW_END_IS_WBOUND   = 1 << 1,
    SEQREP_CW_ALL_WS_PUNCT    = 1 << 2,
};


struct seqrep_logit_info {
    const int n_vocab;
    std::vector<llama_token_data> token_data;

    seqrep_logit_info(llama_context * ctx, const size_t k);

    const std::vector<llama_token_data> & get_token_data(void);

    llama_token_data get_token_id(const llama_token token_id) const;

    void rebuild(llama_context *ctx, const size_t k);

    void populate_logits(float * logits);

    // Yoinked from beam search code.
    // Return top k token_data by logit.
    std::vector<llama_token_data> top_k(const float * const logits, const size_t k);

};

struct seqrep_rewind_slot {
  size_t count;
  std::vector<llama_token> tokens;
};

struct seqrep_rewind_state {
    const size_t n_vocab;
    const size_t n_ctx;
    const size_t k;

    size_t high_water_mark;
    std::vector<seqrep_logit_info>  logit_slots;
    std::vector<seqrep_rewind_slot> rewind_slots;

    seqrep_rewind_state(const size_t n_vocab, const size_t n_ctx, const size_t k, const size_t high_water_mark);

    struct seqrep_rewind_slot & get_rewind_slot(const size_t orig_idx);

    void set_logits_slot(llama_context * ctx, const size_t orig_idx);

    void populate_logits(llama_context * ctx, const size_t orig_idx);

    void set_high_water_mark(const size_t val);

};

// Sequence repetition penalty with semi-fuzzy matching. Note: Handles the last_n window itself.
size_t llama_sample_seqrep_penalty(
    struct llama_context * ctx,
    llama_token_data_array * candidates,
    const llama_token * last_tokens_p, size_t last_tokens_size,
    const llama_sampler_seqrep_params * params);

int llama_seqrep_check_word(struct llama_context * ctx, const llama_token token, std::vector<char> & buf);

size_t llama_seqrep_handle_rewind(
        struct llama_context * ctx,
        struct seqrep_rewind_state & rewind_state,
        const std::vector<llama_token> & last_tokens,
        const size_t prompt_tokens_size,
        const std::vector<llama_sampler_seqrep_params> & params_list,
        size_t * high_water_mark);

void seqrep_sampler_help();
void seqrep_sampler_params_init(llama_sampler_seqrep_params * params);
void seqrep_sampler_params_dump(const llama_sampler_seqrep_params * params);
bool seqrep_sampler_params_parse(const char * s, llama_sampler_seqrep_params * params);
struct llama_sampler_seqrep_params llama_seqrep_merge_params(
    const std::vector<llama_sampler_seqrep_params> & params_list,
    const int and_flags,
    const int not_flags);
