#pragma once

// Defines sigaction on msys:
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "common.h"

#include "console.h"
#include "llama.h"
#include "build-info.h"
#include "grammar-parser.h"

#include <algorithm>
#include <atomic>
#include <limits>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined (_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <signal.h>
#endif

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
void sigint_handler(int signo);
#endif

void write_logfile(
    const llama_context * ctx, const gpt_params & params, const llama_model * model,
    const std::vector<llama_token> input_tokens, const std::string output, const std::vector<llama_token> output_tokens);

bool check_unsupported(const gpt_params * params);

bool initialize(llama_context **ctx_p, llama_model **model_p, gpt_params & params, std::vector<llama_token> & embd_inp, llama_grammar ** grammar_p);

bool feed_prompt(llama_context *ctx, const gpt_params * params, llama_token * tokens, int tokens_len, int n_past);
