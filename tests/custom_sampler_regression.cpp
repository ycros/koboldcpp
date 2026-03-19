#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "../custom_sampler_duktape.h"
#include "../vendor/nlohmann/json.hpp"

namespace {

std::vector<llama_token> g_recent_tokens;

void sort_candidates(llama_token_data_array * cur_p) {
    std::sort(cur_p->data, cur_p->data + cur_p->size, [](const llama_token_data & lhs, const llama_token_data & rhs) {
        return lhs.logit > rhs.logit;
    });
    cur_p->sorted = true;
}

bool expect(bool condition, const std::string & name) {
    if (condition) {
        std::cout << "[PASS] " << name << '\n';
        return true;
    }

    std::cerr << "[FAIL] " << name << '\n';
    return false;
}

} // namespace

void sample_softmax(llama_token_data_array * cur_p, bool do_sort) {
    if (do_sort && !cur_p->sorted) {
        sort_candidates(cur_p);
    }

    float max_logit = cur_p->data[0].logit;
    if (!cur_p->sorted) {
        for (size_t i = 1; i < cur_p->size; ++i) {
            max_logit = std::max(max_logit, cur_p->data[i].logit);
        }
    }

    float total = 0.0f;
    for (size_t i = 0; i < cur_p->size; ++i) {
        cur_p->data[i].p = std::exp(cur_p->data[i].logit - max_logit);
        total += cur_p->data[i].p;
    }
    for (size_t i = 0; i < cur_p->size; ++i) {
        cur_p->data[i].p /= total;
    }
}

void sample_top_k(llama_token_data_array * cur_p, int32_t k) {
    if (!cur_p->sorted) {
        sort_candidates(cur_p);
    }
    if (k <= 0 || static_cast<size_t>(k) > cur_p->size) {
        return;
    }
    cur_p->size = static_cast<size_t>(k);
}

void sample_top_a(llama_token_data_array * candidates, float a, size_t min_keep) {
    sample_softmax(candidates, true);
    const float threshold = a * candidates->data[0].p * candidates->data[0].p;
    size_t keep = candidates->size;
    for (size_t i = 0; i < candidates->size; ++i) {
        if (candidates->data[i].p < threshold && i >= min_keep) {
            keep = i;
            break;
        }
    }
    candidates->size = keep;
}

void sample_top_p(llama_token_data_array * cur_p, float p, size_t min_keep) {
    sample_softmax(cur_p, true);
    float cumulative = 0.0f;
    size_t keep = cur_p->size;
    for (size_t i = 0; i < cur_p->size; ++i) {
        cumulative += cur_p->data[i].p;
        if (cumulative >= p && i + 1 >= min_keep) {
            keep = i + 1;
            break;
        }
    }
    cur_p->size = keep;
}

void sample_min_p(llama_token_data_array * cur_p, float p, size_t min_keep) {
    if (!cur_p->sorted) {
        sort_candidates(cur_p);
    }
    const float min_logit = cur_p->data[0].logit + std::log(p <= 0.0f ? 1.0f : p);
    size_t keep = cur_p->size;
    for (size_t i = 0; i < cur_p->size; ++i) {
        if (cur_p->data[i].logit < min_logit && i >= min_keep) {
            keep = i;
            break;
        }
    }
    cur_p->size = keep;
}

void sample_tail_free(llama_token_data_array * cur_p, float, size_t min_keep) {
    if (cur_p->size > min_keep + 1) {
        cur_p->size = min_keep + 1;
    }
}

void sampler_typical(llama_token_data_array * cur_p, float p, size_t min_keep) {
    sample_top_p(cur_p, p, min_keep);
}

void sample_top_n_sigma(llama_token_data_array * cur_p, float) {
    sample_softmax(cur_p, true);
}

void sample_entropy(llama_token_data_array * cur_p, float min_temp, float, float, float, float) {
    for (size_t i = 0; i < cur_p->size; ++i) {
        cur_p->data[i].logit /= std::max(min_temp, 0.0001f);
    }
    cur_p->sorted = false;
}

void sample_temperature(llama_token_data_array * candidates_p, float temp, float, float) {
    const float denom = temp <= 0.0f ? 0.0001f : temp;
    for (size_t i = 0; i < candidates_p->size; ++i) {
        candidates_p->data[i].logit /= denom;
    }
    candidates_p->sorted = false;
}

void sample_rep_pen(int, int rep_pen_range, float rep_pen, float, float presence_penalty, llama_token_data_array * candidates_p) {
    const size_t count = std::min(static_cast<size_t>(std::max(rep_pen_range, 0)), g_recent_tokens.size());
    for (size_t i = 0; i < candidates_p->size; ++i) {
        for (size_t j = 0; j < count; ++j) {
            if (candidates_p->data[i].id == g_recent_tokens[g_recent_tokens.size() - 1 - j]) {
                candidates_p->data[i].logit = (candidates_p->data[i].logit > 0.0f)
                    ? candidates_p->data[i].logit / rep_pen
                    : candidates_p->data[i].logit * rep_pen;
                candidates_p->data[i].logit -= presence_penalty;
                break;
            }
        }
    }
    candidates_p->sorted = false;
}

void sample_xtc(llama_token_data_array *, float, float, std::mt19937 &) {
}

size_t custom_sampler_recent_token_count() {
    return g_recent_tokens.size();
}

llama_token custom_sampler_recent_token_from_back(size_t back) {
    return g_recent_tokens[g_recent_tokens.size() - 1 - back];
}

std::string custom_sampler_token_text(llama_token token) {
    if (token == 70) {
        return std::string("\xE2", 1);
    }
    return "tok:" + std::to_string(token);
}

int main() {
    int failures = 0;

    {
        duktape_custom_sampler sampler;
        std::string error;
        failures += !expect(!sampler.initialize("function init() {}", nullptr, error) && error.find("sample") != std::string::npos,
                            "missing sample() is rejected");
    }

    {
        duktape_custom_sampler sampler;
        std::string error;
        failures += !expect(!sampler.initialize("function init(params) {} function sample(s) { return s.pick(); }", "not-json", error) && error.find("custom_sampler_params") != std::string::npos,
                            "invalid init params JSON is rejected cleanly");
    }

    {
        duktape_custom_sampler sampler;
        std::string error;
        const char * source =
            "var boost = 0;\n"
            "function init(params) { boost = params.boost || 0; }\n"
            "function sample(s) { s.addLogit(1, boost); return s.pickGreedy(); }\n";
        failures += !expect(sampler.initialize(source, "{\"boost\": 3.5}", error), "init(params) runs");

        std::vector<llama_token_data> candidates = {
            { 10, 0.0f, 0.0f },
            { 11, 0.0f, 0.0f },
        };
        llama_token_data_array array = { candidates.data(), candidates.size(), -1, false };
        std::mt19937 rng(1234);
        const auto result = sampler.apply(&array, 32, 1000, rng);
        failures += !expect(result.has_selection && result.token_id == 11 && result.error_message.empty(),
                            "returned token uses params-initialized state");
    }

    {
        duktape_custom_sampler sampler;
        std::string error;
        const char * source =
            "function sample(s) { s.topK(1); }\n";
        failures += !expect(sampler.initialize(source, nullptr, error), "transform-only sampler initializes");

        std::vector<llama_token_data> candidates = {
            { 20, -1.0f, 0.0f },
            { 21, 3.0f, 0.0f },
            { 22, 1.0f, 0.0f },
        };
        llama_token_data_array array = { candidates.data(), candidates.size(), -1, false };
        std::mt19937 rng(99);
        const auto result = sampler.apply(&array, 32, 1000, rng);
        failures += !expect(!result.has_selection && result.error_message.empty() && array.size == 1 && array.data[0].id == 21,
                            "transform-only sampler mutates candidate set");
    }

    {
        duktape_custom_sampler sampler;
        std::string error;
        const char * source =
            "function sample(s) { s.topA(100, 0); }\n";
        failures += !expect(sampler.initialize(source, nullptr, error), "minKeep clamp sampler initializes");

        std::vector<llama_token_data> candidates = {
            { 30, -1.0f, 0.0f },
            { 31, 3.0f, 0.0f },
            { 32, 1.0f, 0.0f },
        };
        llama_token_data_array array = { candidates.data(), candidates.size(), -1, false };
        std::mt19937 rng(99);
        const auto result = sampler.apply(&array, 32, 1000, rng);
        failures += !expect(!result.has_selection && result.error_message.empty() && array.size == 1,
                            "candidate-pruning minKeep is clamped to preserve one token");
    }

    {
        g_recent_tokens = { 40, 41, 42 };
        duktape_custom_sampler sampler;
        std::string error;
        const char * source =
            "function sample(s) {\n"
            "  if (s.recentCount() > 0 && s.tokenText(s.recentToken(0)) === 'tok:42') {\n"
            "    return s.pickIndex(1);\n"
            "  }\n"
            "  return s.pickIndex(0);\n"
            "}\n";
        failures += !expect(sampler.initialize(source, nullptr, error), "recent token helpers initialize");

        std::vector<llama_token_data> candidates = {
            { 42, 3.0f, 0.0f },
            { 43, 2.0f, 0.0f },
        };
        llama_token_data_array array = { candidates.data(), candidates.size(), -1, true };
        std::mt19937 rng(5);
        const auto result = sampler.apply(&array, 32, 1000, rng);
        failures += !expect(result.has_selection && result.token_id == 43, "recentToken()/tokenText() are exposed");
    }

    {
        duktape_custom_sampler sampler;
        std::string error;
        const char * source = "function sample(s) { return 999999; }\n";
        failures += !expect(sampler.initialize(source, nullptr, error), "invalid-return sampler initializes");

        std::vector<llama_token_data> candidates = {
            { 50, 0.5f, 0.0f },
            { 51, 0.4f, 0.0f },
        };
        llama_token_data_array array = { candidates.data(), candidates.size(), -1, true };
        std::mt19937 rng(1);
        const auto result = sampler.apply(&array, 16, 256, rng);
        failures += !expect(!result.has_selection && !result.error_message.empty(), "invalid token return is rejected");
    }

    {
        duktape_custom_sampler sampler;
        std::string error;
        const char * source = "function sample(s) { for (;;) {} }\n";
        failures += !expect(sampler.initialize(source, nullptr, error), "timeout sampler initializes");

        std::vector<llama_token_data> candidates = {
            { 52, 0.5f, 0.0f },
            { 53, 0.4f, 0.0f },
        };
        llama_token_data_array array = { candidates.data(), candidates.size(), -1, true };
        std::mt19937 rng(2);
        const auto result = sampler.apply(&array, 16, 256, rng);
        failures += !expect(!result.has_selection && result.error_message.find("timed out") != std::string::npos,
                            "infinite-loop sampler times out cleanly");
    }

    {
        duktape_custom_sampler sampler;
        std::string error;
        const char * source =
            "function init(params, s) { s.log('init', params.mode); }\n"
            "function sample(s) { s.log('before', s.size()); s.topK(2); s.log('after', s.size()); }\n";
        failures += !expect(sampler.initialize(source, "{\"mode\":\"debug\"}", error, true), "debug-enabled sampler initializes");

        std::vector<llama_token_data> candidates = {
            { 60, 3.0f, 0.0f },
            { 61, 2.0f, 0.0f },
            { 62, 1.0f, 0.0f },
        };
        llama_token_data_array array = { candidates.data(), candidates.size(), -1, false };
        std::mt19937 rng(7);
        const auto result = sampler.apply(&array, 32, 512, rng);
        sampler.record_final_choice(60);

        auto debug = nlohmann::json::parse(sampler.debug_json());
        failures += !expect(!result.has_selection && result.error_message.empty(), "debug sampler leaves native pipeline running");
        failures += !expect(debug["init_logs"].size() == 1 && debug["init_logs"][0] == "init debug", "init log is captured");
        failures += !expect(debug["steps"].size() == 1, "per-token debug step is captured");
        failures += !expect(debug["steps"][0]["logs"].size() == 2, "sample logs are captured");
        failures += !expect(debug["steps"][0]["pre"]["candidate_count"] == 3 && debug["steps"][0]["post"]["candidate_count"] == 2, "pre and post candidate snapshots are captured");
        failures += !expect(debug["steps"][0]["final_token_id"] == 60, "final token choice is captured");
    }

    {
        duktape_custom_sampler sampler;
        std::string error;
        const char * source = "function sample(s) { return s.pickToken(70); }\n";
        failures += !expect(sampler.initialize(source, nullptr, error, true), "debug sampler with invalid utf8 token initializes");

        std::vector<llama_token_data> candidates = {
            { 70, 1.0f, 0.0f },
        };
        llama_token_data_array array = { candidates.data(), candidates.size(), -1, true };
        std::mt19937 rng(3);
        const auto result = sampler.apply(&array, 16, 128, rng);
        sampler.record_final_choice(70);

        auto debug = nlohmann::json::parse(sampler.debug_json());
        failures += !expect(result.has_selection && result.token_id == 70, "invalid utf8 token can still be selected");
        failures += !expect(debug["steps"][0]["returned_token"].is_string() && debug["steps"][0]["returned_bytes"].size() == 1 && debug["steps"][0]["returned_bytes"][0] == 0xE2,
                            "debug json sanitizes invalid utf8 tokens and preserves bytes");
    }

    if (failures != 0) {
        std::cerr << failures << " custom sampler regression test(s) failed\n";
        return 1;
    }

    std::cout << "all custom sampler regression tests passed\n";
    return 0;
}
