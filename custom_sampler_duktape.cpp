#include "custom_sampler_duktape.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <random>
#include <string>
#include <vector>

extern "C" {
#include "vendor/duktape/duktape.h"
}

#include "custom_sampler_duktape_runtime.h"
#include "src/unicode.h"
#include "vendor/nlohmann/json.hpp"

void sample_softmax(llama_token_data_array * cur_p, bool do_sort);
void sample_top_k(llama_token_data_array * cur_p, int32_t k);
void sample_top_a(llama_token_data_array * candidates, float a, size_t min_keep);
void sample_top_p(llama_token_data_array * cur_p, float p, size_t min_keep);
void sample_min_p(llama_token_data_array * cur_p, float p, size_t min_keep);
void sample_tail_free(llama_token_data_array * cur_p, float z, size_t min_keep);
void sampler_typical(llama_token_data_array * cur_p, float p, size_t min_keep);
void sample_top_n_sigma(llama_token_data_array * cur_p, float nsigma);
void sample_entropy(llama_token_data_array * cur_p, float min_temp, float max_temp, float exponent_val, float smoothing_factor, float smoothing_curve);
void sample_temperature(llama_token_data_array * candidates_p, float temp, float smoothing_factor, float smoothing_curve);
void sample_rep_pen(int n_ctx, int rep_pen_range, float rep_pen, float rep_pen_slope, float presence_penalty, llama_token_data_array * candidates_p);
void sample_xtc(llama_token_data_array * candidates, float xtc_threshold, float xtc_probability, std::mt19937 & rng);
size_t custom_sampler_recent_token_count();
llama_token custom_sampler_recent_token_from_back(size_t back);
std::string custom_sampler_token_text(llama_token token);

struct custom_sampler_call_state {
    llama_token_data_array * candidates = nullptr;
    int n_ctx = 0;
    int n_vocab = 0;
    std::mt19937 * rng = nullptr;
    bool probabilities_dirty = true;
};

enum class custom_sampler_debug_phase {
    none,
    init,
    sample,
};

struct duktape_custom_sampler::impl {
    duk_context * ctx = nullptr;
    custom_sampler_call_state current;
    size_t heap_bytes = 0;
    size_t heap_limit_bytes = 8u * 1024u * 1024u;
    bool memory_limit_triggered = false;
    bool exec_timeout_active = false;
    bool exec_timeout_triggered = false;
    std::chrono::steady_clock::time_point exec_deadline = {};
    bool enabled = false;
    bool debug_enabled = false;
    custom_sampler_debug_phase debug_phase = custom_sampler_debug_phase::none;
    nlohmann::json debug_payload;
};

namespace {

using json = nlohmann::json;
using steady_clock = std::chrono::steady_clock;

constexpr const char * KCPP_DUK_API_OBJECT = DUK_HIDDEN_SYMBOL("kcpp_api_object");
constexpr const char * KCPP_DUK_RUNTIME = DUK_HIDDEN_SYMBOL("kcpp_runtime");
constexpr auto KCPP_DUK_INIT_TIMEOUT = std::chrono::milliseconds(250);
constexpr auto KCPP_DUK_SAMPLE_TIMEOUT = std::chrono::milliseconds(50);

using sampler_method = duk_ret_t (*)(duk_context *);

struct sampler_method_def {
    const char * name;
    sampler_method method;
    duk_idx_t nargs;
};

union tracked_allocation_header {
    size_t size;
    std::max_align_t align;
};

duktape_custom_sampler::impl * runtime_from_udata(void * udata) {
    return static_cast<duktape_custom_sampler::impl *>(udata);
}

size_t tracked_total_size(size_t payload_size) {
    if (payload_size > std::numeric_limits<size_t>::max() - sizeof(tracked_allocation_header)) {
        return 0;
    }
    return sizeof(tracked_allocation_header) + payload_size;
}

tracked_allocation_header * tracked_header_from_ptr(void * ptr) {
    if (ptr == nullptr) {
        return nullptr;
    }
    return static_cast<tracked_allocation_header *>(ptr) - 1;
}

void begin_exec_budget(duktape_custom_sampler::impl * runtime, std::chrono::milliseconds timeout) {
    runtime->memory_limit_triggered = false;
    runtime->exec_timeout_active = true;
    runtime->exec_timeout_triggered = false;
    runtime->exec_deadline = steady_clock::now() + timeout;
}

void end_exec_budget(duktape_custom_sampler::impl * runtime) {
    runtime->exec_timeout_active = false;
    runtime->exec_timeout_triggered = false;
}

void * limited_alloc(void * udata, duk_size_t size) {
    auto * runtime = runtime_from_udata(udata);
    if (runtime == nullptr) {
        return nullptr;
    }

    const size_t payload_size = static_cast<size_t>(size);
    const size_t total_size = tracked_total_size(payload_size);
    if (total_size == 0 || payload_size > runtime->heap_limit_bytes || runtime->heap_bytes > runtime->heap_limit_bytes - payload_size) {
        runtime->memory_limit_triggered = true;
        return nullptr;
    }

    auto * header = static_cast<tracked_allocation_header *>(std::malloc(total_size));
    if (header == nullptr) {
        return nullptr;
    }

    header->size = payload_size;
    runtime->heap_bytes += payload_size;
    return header + 1;
}

void * limited_realloc(void * udata, void * ptr, duk_size_t size) {
    auto * runtime = runtime_from_udata(udata);
    if (runtime == nullptr) {
        return nullptr;
    }

    if (ptr == nullptr) {
        return limited_alloc(udata, size);
    }

    if (size == 0) {
        auto * header = tracked_header_from_ptr(ptr);
        runtime->heap_bytes -= header->size;
        std::free(header);
        return nullptr;
    }

    auto * header = tracked_header_from_ptr(ptr);
    const size_t old_payload_size = header->size;
    const size_t new_payload_size = static_cast<size_t>(size);
    const size_t total_size = tracked_total_size(new_payload_size);
    if (total_size == 0) {
        runtime->memory_limit_triggered = true;
        return nullptr;
    }

    if (new_payload_size > old_payload_size) {
        const size_t growth = new_payload_size - old_payload_size;
        if (growth > runtime->heap_limit_bytes || runtime->heap_bytes > runtime->heap_limit_bytes - growth) {
            runtime->memory_limit_triggered = true;
            return nullptr;
        }
    }

    auto * new_header = static_cast<tracked_allocation_header *>(std::realloc(header, total_size));
    if (new_header == nullptr) {
        return nullptr;
    }

    runtime->heap_bytes = runtime->heap_bytes - old_payload_size + new_payload_size;
    new_header->size = new_payload_size;
    return new_header + 1;
}

void limited_free(void * udata, void * ptr) {
    auto * runtime = runtime_from_udata(udata);
    if (ptr == nullptr) {
        return;
    }

    auto * header = tracked_header_from_ptr(ptr);
    if (runtime != nullptr && runtime->heap_bytes >= header->size) {
        runtime->heap_bytes -= header->size;
    }
    std::free(header);
}

void reset_debug_state(duktape_custom_sampler::impl * runtime, bool enabled) {
    runtime->debug_enabled = enabled;
    runtime->debug_phase = custom_sampler_debug_phase::none;
    runtime->debug_payload = json::object();
    if (enabled) {
        runtime->debug_payload["init_logs"] = json::array();
        runtime->debug_payload["steps"] = json::array();
    }
}

std::string normalize_utf8_for_json(const std::string & raw) {
    std::string normalized;
    for (const auto cpt : unicode_cpts_from_utf8(raw)) {
        normalized += unicode_cpt_to_utf8(cpt);
    }
    return normalized;
}

json bytes_for_json(const std::string & raw) {
    json bytes = json::array();
    for (const unsigned char ch : raw) {
        bytes.push_back(json(static_cast<int>(ch)));
    }
    return bytes;
}

json token_json_value(llama_token token_id) {
    const std::string raw = custom_sampler_token_text(token_id);
    json value = json::object();
    value["token"] = normalize_utf8_for_json(raw);
    value["bytes"] = bytes_for_json(raw);
    return value;
}

json make_candidate_snapshot(llama_token_data_array * candidates) {
    json snapshot = json::object();
    snapshot["candidate_count"] = candidates != nullptr ? candidates->size : 0;
    snapshot["candidates"] = json::array();

    if (candidates == nullptr || candidates->size == 0) {
        return snapshot;
    }

    std::vector<llama_token_data> copied(candidates->data, candidates->data + candidates->size);
    llama_token_data_array copied_array = { copied.data(), copied.size(), -1, candidates->sorted };
    sample_softmax(&copied_array, true);

    for (size_t i = 0; i < copied_array.size; ++i) {
        const auto & candidate = copied_array.data[i];
        json item = json::object();
        item["token_id"] = candidate.id;
        const json token_value = token_json_value(candidate.id);
        item["token"] = token_value["token"];
        item["bytes"] = token_value["bytes"];
        item["logit"] = candidate.logit;
        item["prob"] = candidate.p;
        snapshot["candidates"].push_back(std::move(item));
    }

    return snapshot;
}

json * get_current_step_json(duktape_custom_sampler::impl * runtime) {
    if (!runtime->debug_enabled || !runtime->debug_payload.contains("steps") || runtime->debug_payload["steps"].empty()) {
        return nullptr;
    }
    return &runtime->debug_payload["steps"].back();
}

void note_debug_error(duktape_custom_sampler::impl * runtime, const std::string & error_message) {
    if (auto * step = get_current_step_json(runtime)) {
        (*step)["error"] = normalize_utf8_for_json(error_message);
    }
}

void append_debug_log(duktape_custom_sampler::impl * runtime, const std::string & message) {
    if (!runtime->debug_enabled) {
        return;
    }

    const std::string safe_message = normalize_utf8_for_json(message);

    if (runtime->debug_phase == custom_sampler_debug_phase::init) {
        runtime->debug_payload["init_logs"].push_back(safe_message);
        return;
    }

    if (runtime->debug_phase == custom_sampler_debug_phase::sample) {
        if (auto * step = get_current_step_json(runtime)) {
            (*step)["logs"].push_back(safe_message);
        }
    }
}

void begin_debug_step(duktape_custom_sampler::impl * runtime, llama_token_data_array * candidates) {
    if (!runtime->debug_enabled) {
        return;
    }

    json step = json::object();
    step["token_index"] = runtime->debug_payload["steps"].size();
    step["logs"] = json::array();
    step["pre"] = make_candidate_snapshot(candidates);
    step["post"] = json::object();
    runtime->debug_payload["steps"].push_back(std::move(step));
    runtime->debug_phase = custom_sampler_debug_phase::sample;
}

void finish_debug_step(duktape_custom_sampler::impl * runtime, llama_token_data_array * candidates) {
    if (!runtime->debug_enabled) {
        return;
    }

    if (auto * step = get_current_step_json(runtime)) {
        (*step)["post"] = make_candidate_snapshot(candidates);
    }
    runtime->debug_phase = custom_sampler_debug_phase::none;
}

std::string build_log_message(duk_context * ctx) {
    std::string message;
    const duk_idx_t top = duk_get_top(ctx);
    for (duk_idx_t i = 0; i < top; ++i) {
        if (i > 0) {
            message += ' ';
        }
        message += duk_safe_to_string(ctx, i);
    }
    return message;
}

duktape_custom_sampler::impl * get_runtime(duk_context * ctx) {
    duktape_custom_sampler::impl * runtime = nullptr;

    duk_push_heap_stash(ctx);
    duk_get_prop_string(ctx, -1, KCPP_DUK_API_OBJECT);
    if (duk_is_object(ctx, -1)) {
        duk_get_prop_string(ctx, -1, KCPP_DUK_RUNTIME);
        runtime = static_cast<duktape_custom_sampler::impl *>(duk_get_pointer_default(ctx, -1, nullptr));
        duk_pop(ctx);
    }
    duk_pop_2(ctx);

    if (runtime == nullptr) {
        duk_error(ctx, DUK_ERR_ERROR, "custom sampler runtime unavailable");
    }
    return runtime;
}

custom_sampler_call_state & get_call_state(duk_context * ctx) {
    auto * runtime = get_runtime(ctx);
    if (runtime->current.candidates == nullptr || runtime->current.rng == nullptr) {
        duk_error(ctx, DUK_ERR_ERROR, "custom sampler is not active");
    }
    return runtime->current;
}

llama_token_data & require_candidate(duk_context * ctx, custom_sampler_call_state & state, duk_idx_t arg_idx) {
    const duk_int_t index = duk_require_int(ctx, arg_idx);
    if (index < 0 || static_cast<size_t>(index) >= state.candidates->size) {
        duk_error(ctx, DUK_ERR_RANGE_ERROR, "candidate index out of range");
    }
    return state.candidates->data[index];
}

void invalidate_probabilities(custom_sampler_call_state & state) {
    state.probabilities_dirty = true;
}

void ensure_probabilities(custom_sampler_call_state & state) {
    if (state.candidates->size == 0) {
        return;
    }
    if (state.probabilities_dirty) {
        sample_softmax(state.candidates, true);
        state.probabilities_dirty = false;
    }
}

size_t min_keep_arg(duk_context * ctx, duk_idx_t arg_idx) {
    const duk_int_t raw = duk_get_int_default(ctx, arg_idx, 1);
    return static_cast<size_t>(raw <= 0 ? 1 : raw);
}

duk_ret_t return_self(duk_context * ctx) {
    duk_push_this(ctx);
    return 1;
}

llama_token choose_random_token(duk_context * ctx, custom_sampler_call_state & state) {
    if (state.candidates->size == 0) {
        duk_error(ctx, DUK_ERR_RANGE_ERROR, "candidate set is empty");
    }
    ensure_probabilities(state);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    const float target = dist(*state.rng);
    float cumulative = 0.0f;
    for (size_t i = 0; i < state.candidates->size; ++i) {
        cumulative += state.candidates->data[i].p;
        if (target <= cumulative || i + 1 == state.candidates->size) {
            return state.candidates->data[i].id;
        }
    }
    return state.candidates->data[state.candidates->size - 1].id;
}

llama_token choose_greedy_token(duk_context * ctx, custom_sampler_call_state & state) {
    if (state.candidates->size == 0) {
        duk_error(ctx, DUK_ERR_RANGE_ERROR, "candidate set is empty");
    }
    auto * begin = state.candidates->data;
    auto * end = begin + state.candidates->size;
    auto * best = std::max_element(begin, end, [](const llama_token_data & lhs, const llama_token_data & rhs) {
        return lhs.logit < rhs.logit;
    });
    return best->id;
}

bool candidate_contains_token(const custom_sampler_call_state & state, llama_token token_id) {
    for (size_t i = 0; i < state.candidates->size; ++i) {
        if (state.candidates->data[i].id == token_id) {
            return true;
        }
    }
    return false;
}

duk_ret_t duk_size(duk_context * ctx) {
    auto & state = get_call_state(ctx);
    duk_push_int(ctx, static_cast<duk_int_t>(state.candidates->size));
    return 1;
}

duk_ret_t duk_id(duk_context * ctx) {
    auto & state = get_call_state(ctx);
    duk_push_int(ctx, require_candidate(ctx, state, 0).id);
    return 1;
}

duk_ret_t duk_logit(duk_context * ctx) {
    auto & state = get_call_state(ctx);
    duk_push_number(ctx, require_candidate(ctx, state, 0).logit);
    return 1;
}

duk_ret_t duk_prob(duk_context * ctx) {
    auto & state = get_call_state(ctx);
    const duk_int_t index = duk_require_int(ctx, 0);
    if (index < 0 || static_cast<size_t>(index) >= state.candidates->size) {
        duk_error(ctx, DUK_ERR_RANGE_ERROR, "candidate index out of range");
    }
    ensure_probabilities(state);
    duk_push_number(ctx, state.candidates->data[index].p);
    return 1;
}

duk_ret_t duk_set_logit(duk_context * ctx) {
    auto & state = get_call_state(ctx);
    auto & candidate = require_candidate(ctx, state, 0);
    candidate.logit = duk_require_number(ctx, 1);
    state.candidates->sorted = false;
    invalidate_probabilities(state);
    return return_self(ctx);
}

duk_ret_t duk_add_logit(duk_context * ctx) {
    auto & state = get_call_state(ctx);
    auto & candidate = require_candidate(ctx, state, 0);
    candidate.logit += duk_require_number(ctx, 1);
    state.candidates->sorted = false;
    invalidate_probabilities(state);
    return return_self(ctx);
}

duk_ret_t duk_softmax(duk_context * ctx) {
    auto & state = get_call_state(ctx);
    sample_softmax(state.candidates, true);
    state.probabilities_dirty = false;
    return return_self(ctx);
}

duk_ret_t duk_top_k(duk_context * ctx) {
    auto & state = get_call_state(ctx);
    sample_top_k(state.candidates, duk_require_int(ctx, 0));
    invalidate_probabilities(state);
    return return_self(ctx);
}

duk_ret_t duk_top_a(duk_context * ctx) {
    auto & state = get_call_state(ctx);
    sample_top_a(state.candidates, duk_require_number(ctx, 0), min_keep_arg(ctx, 1));
    invalidate_probabilities(state);
    return return_self(ctx);
}

duk_ret_t duk_top_p(duk_context * ctx) {
    auto & state = get_call_state(ctx);
    sample_top_p(state.candidates, duk_require_number(ctx, 0), min_keep_arg(ctx, 1));
    invalidate_probabilities(state);
    return return_self(ctx);
}

duk_ret_t duk_min_p(duk_context * ctx) {
    auto & state = get_call_state(ctx);
    sample_min_p(state.candidates, duk_require_number(ctx, 0), min_keep_arg(ctx, 1));
    invalidate_probabilities(state);
    return return_self(ctx);
}

duk_ret_t duk_tail_free(duk_context * ctx) {
    auto & state = get_call_state(ctx);
    sample_tail_free(state.candidates, duk_require_number(ctx, 0), min_keep_arg(ctx, 1));
    invalidate_probabilities(state);
    return return_self(ctx);
}

duk_ret_t duk_typical(duk_context * ctx) {
    auto & state = get_call_state(ctx);
    sampler_typical(state.candidates, duk_require_number(ctx, 0), min_keep_arg(ctx, 1));
    invalidate_probabilities(state);
    return return_self(ctx);
}

duk_ret_t duk_top_n_sigma(duk_context * ctx) {
    auto & state = get_call_state(ctx);
    sample_top_n_sigma(state.candidates, duk_require_number(ctx, 0));
    invalidate_probabilities(state);
    return return_self(ctx);
}

duk_ret_t duk_temperature(duk_context * ctx) {
    auto & state = get_call_state(ctx);
    sample_temperature(state.candidates, duk_require_number(ctx, 0), duk_get_number_default(ctx, 1, 0.0), duk_get_number_default(ctx, 2, 1.0));
    invalidate_probabilities(state);
    return return_self(ctx);
}

duk_ret_t duk_entropy(duk_context * ctx) {
    auto & state = get_call_state(ctx);
    sample_entropy(state.candidates,
                   duk_require_number(ctx, 0),
                   duk_require_number(ctx, 1),
                   duk_require_number(ctx, 2),
                   duk_get_number_default(ctx, 3, 0.0),
                   duk_get_number_default(ctx, 4, 1.0));
    invalidate_probabilities(state);
    return return_self(ctx);
}

duk_ret_t duk_rep_penalty(duk_context * ctx) {
    auto & state = get_call_state(ctx);
    sample_rep_pen(state.n_ctx,
                   duk_require_int(ctx, 0),
                   duk_require_number(ctx, 1),
                   duk_require_number(ctx, 2),
                   duk_require_number(ctx, 3),
                   state.candidates);
    invalidate_probabilities(state);
    return return_self(ctx);
}

duk_ret_t duk_xtc(duk_context * ctx) {
    auto & state = get_call_state(ctx);
    sample_xtc(state.candidates, duk_require_number(ctx, 0), duk_require_number(ctx, 1), *state.rng);
    invalidate_probabilities(state);
    return return_self(ctx);
}

duk_ret_t duk_pick(duk_context * ctx) {
    auto & state = get_call_state(ctx);
    duk_push_int(ctx, choose_random_token(ctx, state));
    return 1;
}

duk_ret_t duk_pick_greedy(duk_context * ctx) {
    auto & state = get_call_state(ctx);
    duk_push_int(ctx, choose_greedy_token(ctx, state));
    return 1;
}

duk_ret_t duk_pick_index(duk_context * ctx) {
    auto & state = get_call_state(ctx);
    duk_push_int(ctx, require_candidate(ctx, state, 0).id);
    return 1;
}

duk_ret_t duk_pick_token(duk_context * ctx) {
    auto & state = get_call_state(ctx);
    const duk_int_t token_id = duk_require_int(ctx, 0);
    if (!candidate_contains_token(state, token_id)) {
        duk_error(ctx, DUK_ERR_RANGE_ERROR, "token id is not present in current candidate set");
    }
    duk_push_int(ctx, token_id);
    return 1;
}

duk_ret_t duk_recent_count(duk_context * ctx) {
    duk_push_int(ctx, static_cast<duk_int_t>(custom_sampler_recent_token_count()));
    return 1;
}

duk_ret_t duk_recent_token(duk_context * ctx) {
    const duk_int_t back = duk_require_int(ctx, 0);
    const auto count = custom_sampler_recent_token_count();
    if (back < 0 || static_cast<size_t>(back) >= count) {
        duk_error(ctx, DUK_ERR_RANGE_ERROR, "recent token index out of range");
    }
    duk_push_int(ctx, custom_sampler_recent_token_from_back(static_cast<size_t>(back)));
    return 1;
}

duk_ret_t duk_random(duk_context * ctx) {
    auto & state = get_call_state(ctx);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    duk_push_number(ctx, dist(*state.rng));
    return 1;
}

duk_ret_t duk_token_text(duk_context * ctx) {
    const duk_int_t token_id = duk_require_int(ctx, 0);
    const std::string text = custom_sampler_token_text(token_id);
    duk_push_lstring(ctx, text.c_str(), text.size());
    return 1;
}

duk_ret_t duk_vocab_size(duk_context * ctx) {
    auto & state = get_call_state(ctx);
    duk_push_int(ctx, state.n_vocab);
    return 1;
}

duk_ret_t duk_context_size(duk_context * ctx) {
    auto & state = get_call_state(ctx);
    duk_push_int(ctx, state.n_ctx);
    return 1;
}

duk_ret_t duk_log(duk_context * ctx) {
    auto * runtime = get_runtime(ctx);
    append_debug_log(runtime, build_log_message(ctx));
    return return_self(ctx);
}

void destroy_context(duktape_custom_sampler::impl * runtime) {
    if (runtime->ctx != nullptr) {
        duk_destroy_heap(runtime->ctx);
        runtime->ctx = nullptr;
    }
    runtime->heap_bytes = 0;
    runtime->memory_limit_triggered = false;
    runtime->exec_timeout_active = false;
    runtime->exec_timeout_triggered = false;
    runtime->enabled = false;
    runtime->current = {};
    runtime->debug_phase = custom_sampler_debug_phase::none;
}

void push_api_object_from_stash(duk_context * ctx) {
    duk_push_heap_stash(ctx);
    duk_get_prop_string(ctx, -1, KCPP_DUK_API_OBJECT);
    duk_remove(ctx, -2);
}

void register_method(duk_context * ctx, const char * name, sampler_method method, duk_idx_t nargs) {
    duk_push_c_function(ctx, method, nargs);
    duk_put_prop_string(ctx, -2, name);
}

void create_api_object(duk_context * ctx, duktape_custom_sampler::impl * runtime) {
    static const sampler_method_def methods[] = {
        { "size", duk_size, 0 },
        { "id", duk_id, 1 },
        { "logit", duk_logit, 1 },
        { "prob", duk_prob, 1 },
        { "setLogit", duk_set_logit, 2 },
        { "addLogit", duk_add_logit, 2 },
        { "softmax", duk_softmax, 0 },
        { "topK", duk_top_k, 1 },
        { "topA", duk_top_a, 2 },
        { "topP", duk_top_p, 2 },
        { "minP", duk_min_p, 2 },
        { "tailFree", duk_tail_free, 2 },
        { "typical", duk_typical, 2 },
        { "topNSigma", duk_top_n_sigma, 1 },
        { "temperature", duk_temperature, 3 },
        { "entropy", duk_entropy, 5 },
        { "repPenalty", duk_rep_penalty, 4 },
        { "xtc", duk_xtc, 2 },
        { "pick", duk_pick, 0 },
        { "pickGreedy", duk_pick_greedy, 0 },
        { "pickIndex", duk_pick_index, 1 },
        { "pickToken", duk_pick_token, 1 },
        { "recentCount", duk_recent_count, 0 },
        { "recentToken", duk_recent_token, 1 },
        { "log", duk_log, DUK_VARARGS },
        { "random", duk_random, 0 },
        { "tokenText", duk_token_text, 1 },
        { "vocabSize", duk_vocab_size, 0 },
        { "contextSize", duk_context_size, 0 },
    };

    duk_push_object(ctx);
    duk_push_pointer(ctx, runtime);
    duk_put_prop_string(ctx, -2, KCPP_DUK_RUNTIME);

    for (const auto & method : methods) {
        register_method(ctx, method.name, method.method, method.nargs);
    }

    duk_push_heap_stash(ctx);
    duk_dup(ctx, -2);
    duk_put_prop_string(ctx, -2, KCPP_DUK_API_OBJECT);
    duk_pop(ctx);
    duk_pop(ctx);
}

bool read_error_string(duk_context * ctx, duktape_custom_sampler::impl * runtime, std::string & error_message, const char * prefix = nullptr) {
    const std::string raw_error = duk_safe_to_string(ctx, -1);
    if (runtime != nullptr && runtime->exec_timeout_triggered) {
        error_message = "custom sampler execution timed out";
        if (!raw_error.empty()) {
            error_message += ": ";
            error_message += raw_error;
        }
    } else if (runtime != nullptr && runtime->memory_limit_triggered) {
        error_message = "custom sampler exceeded the Duktape heap limit";
        if (!raw_error.empty()) {
            error_message += ": ";
            error_message += raw_error;
        }
    } else if (prefix != nullptr && prefix[0] != '\0') {
        error_message = prefix;
        if (!raw_error.empty()) {
            error_message += ": ";
            error_message += raw_error;
        }
    } else {
        error_message = raw_error;
    }
    duk_pop(ctx);
    return false;
}

bool validate_sample_function(duk_context * ctx, std::string & error_message) {
    duk_get_global_string(ctx, "sample");
    const bool ok = duk_is_function(ctx, -1) != 0;
    duk_pop(ctx);
    if (!ok) {
        error_message = "custom sampler must define a sample(s) function";
    }
    return ok;
}

duk_ret_t decode_json_argument(duk_context * ctx, void * udata) {
    (void) udata;
    duk_json_decode(ctx, -1);
    return 1;
}

bool push_init_params_value(duk_context * ctx, duktape_custom_sampler::impl * runtime, const char * params_json, std::string & error_message) {
    if (params_json == nullptr || params_json[0] == '\0') {
        duk_push_undefined(ctx);
        return true;
    }

    duk_push_lstring(ctx, params_json, std::strlen(params_json));
    if (duk_safe_call(ctx, decode_json_argument, nullptr, 1, 1) != 0) {
        return read_error_string(ctx, runtime, error_message, "custom_sampler_params must be valid JSON");
    }
    return true;
}

bool call_init_function(duk_context * ctx, const char * params_json, std::string & error_message) {
    auto * runtime = get_runtime(ctx);
    duk_get_global_string(ctx, "init");
    if (!duk_is_function(ctx, -1)) {
        duk_pop(ctx);
        return true;
    }

    begin_exec_budget(runtime, KCPP_DUK_INIT_TIMEOUT);
    if (!push_init_params_value(ctx, runtime, params_json, error_message)) {
        end_exec_budget(runtime);
        duk_pop(ctx);
        return false;
    }
    push_api_object_from_stash(ctx);
    runtime->debug_phase = custom_sampler_debug_phase::init;
    if (duk_pcall(ctx, 2) != 0) {
        runtime->debug_phase = custom_sampler_debug_phase::none;
        const bool ok = read_error_string(ctx, runtime, error_message);
        end_exec_budget(runtime);
        return ok;
    }
    end_exec_budget(runtime);
    runtime->debug_phase = custom_sampler_debug_phase::none;
    duk_pop(ctx);
    return true;
}

} // namespace

extern "C" int kcpp_duktape_exec_timeout_check(void * udata) {
    auto * runtime = static_cast<duktape_custom_sampler::impl *>(udata);
    if (runtime == nullptr || !runtime->exec_timeout_active) {
        return 0;
    }
    if (runtime->exec_timeout_triggered) {
        return 1;
    }
    if (steady_clock::now() >= runtime->exec_deadline) {
        runtime->exec_timeout_triggered = true;
        return 1;
    }
    return 0;
}

duktape_custom_sampler::duktape_custom_sampler()
    : impl_(new impl()) {
}

duktape_custom_sampler::~duktape_custom_sampler() {
    destroy_context(impl_);
    delete impl_;
}

bool duktape_custom_sampler::initialize(const char * source, const char * params_json, std::string & error_message, bool debug_enabled) {
    reset_debug_state(impl_, debug_enabled);
    destroy_context(impl_);

    if (source == nullptr || source[0] == '\0') {
        return true;
    }

    impl_->ctx = duk_create_heap(limited_alloc, limited_realloc, limited_free, impl_, nullptr);
    if (impl_->ctx == nullptr) {
        error_message = "failed to create Duktape heap";
        return false;
    }

    create_api_object(impl_->ctx, impl_);

    begin_exec_budget(impl_, KCPP_DUK_INIT_TIMEOUT);
    if (duk_peval_lstring(impl_->ctx, source, std::strlen(source)) != 0) {
        read_error_string(impl_->ctx, impl_, error_message);
        end_exec_budget(impl_);
        destroy_context(impl_);
        return false;
    }
    end_exec_budget(impl_);
    duk_pop(impl_->ctx);

    if (!validate_sample_function(impl_->ctx, error_message)) {
        destroy_context(impl_);
        return false;
    }

    if (!call_init_function(impl_->ctx, params_json, error_message)) {
        destroy_context(impl_);
        return false;
    }

    impl_->enabled = true;
    return true;
}

bool duktape_custom_sampler::active() const {
    return impl_->enabled;
}

custom_sampler_call_result duktape_custom_sampler::apply(llama_token_data_array * candidates, int n_ctx, int n_vocab, std::mt19937 & rng) {
    custom_sampler_call_result result;

    if (!impl_->enabled || impl_->ctx == nullptr) {
        return result;
    }

    impl_->current.candidates = candidates;
    impl_->current.n_ctx = n_ctx;
    impl_->current.n_vocab = n_vocab;
    impl_->current.rng = &rng;
    impl_->current.probabilities_dirty = true;
    begin_debug_step(impl_, candidates);

    begin_exec_budget(impl_, KCPP_DUK_SAMPLE_TIMEOUT);
    duk_get_global_string(impl_->ctx, "sample");
    push_api_object_from_stash(impl_->ctx);
    if (duk_pcall(impl_->ctx, 1) != 0) {
        read_error_string(impl_->ctx, impl_, result.error_message);
        note_debug_error(impl_, result.error_message);
    } else if (!duk_is_undefined(impl_->ctx, -1) && !duk_is_null(impl_->ctx, -1)) {
        const llama_token token_id = duk_get_int(impl_->ctx, -1);
        if (!candidate_contains_token(impl_->current, token_id)) {
            result.error_message = "custom sampler returned a token id outside the current candidate set";
            note_debug_error(impl_, result.error_message);
        } else {
            result.has_selection = true;
            result.token_id = token_id;
            if (auto * step = get_current_step_json(impl_)) {
                const json token_value = token_json_value(token_id);
                (*step)["returned_token_id"] = token_id;
                (*step)["returned_token"] = token_value["token"];
                (*step)["returned_bytes"] = token_value["bytes"];
            }
        }
        duk_pop(impl_->ctx);
    } else {
        duk_pop(impl_->ctx);
    }

    end_exec_budget(impl_);

    if (result.error_message.empty() && !result.has_selection && impl_->current.candidates != nullptr && impl_->current.candidates->size == 0) {
        result.error_message = "custom sampler removed all candidates from the active set";
        note_debug_error(impl_, result.error_message);
    }

    finish_debug_step(impl_, candidates);
    impl_->current = {};
    return result;
}

void duktape_custom_sampler::record_final_choice(llama_token token_id) {
    if (!impl_->debug_enabled) {
        return;
    }

    if (auto * step = get_current_step_json(impl_)) {
        const json token_value = token_json_value(token_id);
        (*step)["final_token_id"] = token_id;
        (*step)["final_token"] = token_value["token"];
        (*step)["final_bytes"] = token_value["bytes"];
    }
}

std::string duktape_custom_sampler::debug_json() const {
    if (!impl_->debug_enabled) {
        return "";
    }
    return impl_->debug_payload.dump();
}
