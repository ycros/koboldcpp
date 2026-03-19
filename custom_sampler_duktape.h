#pragma once

#include <random>
#include <string>

#include "include/llama.h"

struct custom_sampler_call_result {
    bool has_selection = false;
    llama_token token_id = 0;
    std::string error_message;
};

class duktape_custom_sampler {
public:
    struct impl;

    duktape_custom_sampler();
    ~duktape_custom_sampler();

    duktape_custom_sampler(const duktape_custom_sampler &) = delete;
    duktape_custom_sampler & operator=(const duktape_custom_sampler &) = delete;

    bool initialize(const char * source, const char * params_json, std::string & error_message, bool debug_enabled = false);
    bool active() const;
    custom_sampler_call_result apply(llama_token_data_array * candidates, int n_ctx, int n_vocab, std::mt19937 & rng);
    void record_final_choice(llama_token token_id);
    std::string debug_json() const;

    impl * impl_;
};
