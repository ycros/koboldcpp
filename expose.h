#pragma once

const int stop_token_max = 10;
// match kobold's sampler list and order
enum samplers
{
    KCPP_SAMPLER_TOP_K,
    KCPP_SAMPLER_TOP_A,
    KCPP_SAMPLER_TOP_P,
    KCPP_SAMPLER_TFS,
    KCPP_SAMPLER_TYP,
    KCPP_SAMPLER_TEMP,
    KCPP_SAMPLER_REP_PEN,
    KCPP_SAMPLER_MAX
};
struct load_model_inputs
{
    const int threads;
    const int blasthreads;
    const int max_context_length;
    const int batch_size;
    const bool f16_kv;
    const bool low_vram;
    const char * executable_path;
    const char * model_filename;
    const char * lora_filename;
    const char * lora_base;
    const bool use_mmap;
    const bool use_mlock;
    const bool use_smartcontext;
    const bool unban_tokens;
    const int clblast_info = 0;
    const int blasbatchsize = 512;
    const int debugmode = 0;
    const int forceversion = 0;
    const int gpulayers = 0;
};
struct generation_inputs
{
    const int seed;
    const char *prompt;
    const int max_context_length;
    const int max_length;
    const float temperature;
    const int top_k;
    const float top_a = 0.0f;
    const float top_p;
    const float typical_p;
    const float tfs;
    const float rep_pen;
    const int rep_pen_range;
    const int mirostat = 0;
    const float mirostat_eta;
    const float mirostat_tau;
    const samplers sampler_order[KCPP_SAMPLER_MAX];
    const int sampler_len;
    const char * stop_sequence[stop_token_max];
    const bool stream_sse;
};
struct generation_outputs
{
    int status = -1;
    char text[16384]; //16kb should be enough for any response
};

extern std::string executable_path;
extern std::string lora_filename;
extern std::string lora_base;
extern std::vector<std::string> generated_tokens;
extern bool generation_finished;
