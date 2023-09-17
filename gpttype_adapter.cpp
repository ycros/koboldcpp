//This is Concedo's shitty adapter for adding python bindings for llama

//Considerations:
//Don't want to use pybind11 due to dependencies on MSVCC
//ZERO or MINIMAL changes as possible to main.cpp - do not move their function declarations here!
//Leave main.cpp UNTOUCHED, We want to be able to update the repo and pull any changes automatically.
//No dynamic memory allocation! Setup structs with FIXED (known) shapes and sizes for ALL output fields
//Python will ALWAYS provide the memory, we just write to it.

#include <time.h>
#include <mutex>
#include "model_adapter.h"
#include "otherarch.h"

//for easier compilation
//concat source files into one file for compilation purposes
#include "llama_v2.cpp"
#include "llama_v3.cpp"
#include "llama.cpp"
#include "utils.cpp"
#include "gptj_v1.cpp"
#include "gptj_v2.cpp"
#include "gptj_v3.cpp"
#include "gpt2_v1.cpp"
#include "gpt2_v2.cpp"
#include "gpt2_v3.cpp"
#include "rwkv_v2.cpp"
#include "rwkv_v3.cpp"
#include "neox_v2.cpp"
#include "neox_v3.cpp"
#include "mpt_v3.cpp"

//shared
std::string executable_path = "";
std::string lora_filename = "";
std::string lora_base = "";
bool generation_finished;
float last_process_time = 0;
float last_eval_time = 0;
int last_token_count = 0;
stop_reason last_stop_reason = stop_reason::INVALID;
std::vector<std::string> generated_tokens;

//return val: 0=fail, 1=(original ggml, alpaca), 2=(ggmf), 3=(ggjt)
static FileFormat file_format = FileFormat::BADFORMAT;

static gpt_vocab vocab;

static gptj_v1_model gptj_ctx_v1;
static gptj_v2_model gptj_ctx_v2;
static gptj_model gptj_ctx_v3;

static gpt2_v1_model gpt2_ctx_v1;
static gpt2_v2_model gpt2_ctx_v2;
static gpt2_model gpt2_ctx_v3;

static gpt_neox_v2_model neox_ctx_v2;
static gpt_neox_model neox_ctx_v3;

static mpt_model mpt_ctx_v3;

static rwkv_v2_context * rwkv_ctx_v2;
static rwkv_context * rwkv_ctx_v3;
static llama_v2_context * llama_ctx_v2;
static llama_v3_context * llama_ctx_v3;
static llama_context * llama_ctx_v4;

static gpt_params params;
static int n_past = 0;
static int n_threads = 4;
static int n_blasthreads = 4;
static int n_batch = 8;
static bool useSmartContext = false;
static bool unbanTokens = false;
static int blasbatchsize = 512;
static int debugmode = 0; //-1 = hide all, 0 = normal, 1 = showall
static std::string modelname;
static std::vector<gpt_vocab::id> last_n_tokens;
static std::vector<gpt_vocab::id> current_context_tokens;
static size_t mem_per_token = 0;
static std::vector<float> logits;
static std::vector<int> smartcontext;
static std::vector<std::string> stop_sequence;
static std::vector<std::string> banned_tokens;
static std::vector<int> banned_token_ids;
static std::vector<llama_token_data> top_picks;
static int remaining_tokens = 0;
static int stopper_unused_tokens = 0;
static std::mutex concat_output_mtx;
static std::string concat_output = "";
static std::string concat_output_reader_copy = "";

inline bool IsNanCheck(float f)
{
    const unsigned int u = *(unsigned int*)&f;
    return (u&0x7F800000) == 0x7F800000 && (u&0x7FFFFF);    // Both NaN and qNan.
}

inline bool LogitsDuplicated(std::vector<float> & arr1, std::vector<float> & arr2)
{
    int compareQty = 5;
    if(arr1.size() < compareQty || arr2.size() < compareQty || arr1.size()!=arr2.size())
    {
        printf("\nError: Logit array sizes are bad!\n");
        return false;
    }
    for(int i=0;i<compareQty;++i)
    {
        if(arr1[i]!=arr2[i])
        {
            return false;
        }
    }
    return true;
}


llama_token sample_token(llama_token_data_array * candidates, std::mt19937 & rng)
{
    llama_sample_softmax(nullptr, candidates);
    std::vector<float> probs;
    probs.reserve(candidates->size);
    top_picks.clear();
    for (size_t i = 0; i < candidates->size; ++i) {
        probs.push_back(candidates->data[i].p);
    }

    std::discrete_distribution<> dist(probs.begin(), probs.end());
    int idx = dist(rng);

    if(debugmode==1)
    {
        top_picks.push_back(candidates->data[idx]);
        for (size_t i = 0; (i < candidates->size && i<4); ++i)
        {
            if(i!=idx)
            {
                top_picks.push_back(candidates->data[i]);
            }
        }
    }

    llama_token result = candidates->data[idx].id;
    return result;
}

llama_token sample_token_mirostat(int n_vocab, llama_token_data_array * candidates, std::mt19937 & rng, float tau, float eta, int m, float * mu)
{
    float N = float(n_vocab);
    llama_sample_softmax(nullptr, candidates);
    // Estimate s_hat using the most probable m tokens
    float s_hat = 0.0;
    float sum_ti_bi = 0.0;
    float sum_ti_sq = 0.0;
    for (size_t i = 0; i < size_t(m - 1) && i < candidates->size - 1; ++i) {
        float t_i = logf(float(i + 2) / float(i + 1));
        float b_i = logf(candidates->data[i].p / candidates->data[i + 1].p);
        sum_ti_bi += t_i * b_i;
        sum_ti_sq += t_i * t_i;
    }
    s_hat = sum_ti_bi / sum_ti_sq;
    // Compute k from the estimated s_hat and target surprise value
    float epsilon_hat = s_hat - 1;
    float k = powf((epsilon_hat * powf(2, *mu)) / (1 - powf(N, -epsilon_hat)), 1 / s_hat);
    // Sample the next word X using top-k sampling
    llama_sample_top_k(nullptr, candidates, int(k),1);
    llama_token X = sample_token(candidates, rng);    // Compute error as the difference between observed surprise and target surprise value
    size_t X_idx = std::distance(candidates->data, std::find_if(candidates->data, candidates->data + candidates->size, [&](const llama_token_data & candidate) {
        return candidate.id == X;
    }));
    float observed_surprise = -log2f(candidates->data[X_idx].p);
    float e = observed_surprise - tau;
    // Update mu using the learning rate and error
    *mu = *mu - eta * e;
    return X;
}

llama_token sample_token_mirostat_v2(llama_token_data_array * candidates, std::mt19937 & rng, float tau, float eta, float * mu)
{
    llama_sample_softmax(nullptr, candidates);
    // Truncate the words with surprise values greater than mu
    candidates->size = std::distance(candidates->data, std::find_if(candidates->data, candidates->data + candidates->size, [&](const llama_token_data & candidate) {
        return -log2f(candidate.p) > *mu;
    }));

    if (candidates->size == 0) {
        candidates->size = 1;
    }

    // Normalize the probabilities of the remaining words
    llama_sample_softmax(nullptr, candidates);
    // Sample the next word X from the remaining words
    llama_token X = sample_token(candidates,rng);

    // Compute error as the difference between observed surprise and target surprise value
    size_t X_idx = std::distance(candidates->data, std::find_if(candidates->data, candidates->data + candidates->size, [&](const llama_token_data & candidate) {
        return candidate.id == X;
    }));
    float observed_surprise = -log2f(candidates->data[X_idx].p);
    float e = observed_surprise - tau;
    // Update mu using the learning rate and error
    *mu = *mu - eta * e;
    return X;
}

// Top-a (remove all tokens that have softmax probability less than top_a*m^2 where m is the maximum softmax probability)
// top-a 0 is off (no effect)
void sample_top_a(llama_token_data_array * candidates, float a, size_t min_keep) {
    if (a <= 0.0f || candidates->size<=1) {
        return;
    }

    llama_sample_softmax(nullptr, candidates);

    // Compute the cumulative probabilities
    float maxprob = candidates->data[0].p;

    float threshold = a * maxprob * maxprob; //tokens with probs less than this are removed
    size_t last_idx = candidates->size;

    for (size_t i = 0; i < candidates->size; ++i) {
        // Go until we reach a value under the threshold
        float checkprob = candidates->data[i].p;
        if (checkprob < threshold && i >= min_keep) {
            last_idx = i;
            break;
        }
    }
    // printf("\n\nCandidates: %d, A:%f, MaxProb: %f, Threshold: %f, LastIdx: %d",candidates->size,a,maxprob,threshold,last_idx);
    // printf("\nCandidates: %f %f %f %f\n",candidates->data[0].p,candidates->data[1].p,candidates->data[2].p,candidates->data[3].p);

    // Resize the output vector to keep only the selected tokens
    candidates->size = last_idx;
}

void sample_rep_pen(int n_ctx, int rep_pen_range, float rep_pen, llama_token_data_array * candidates_p)
{
    auto last_n_repeat = std::min(std::min((int)last_n_tokens.size(), rep_pen_range), n_ctx);
    llama_sample_repetition_penalty(nullptr, candidates_p,
        last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
        last_n_repeat, rep_pen);
}

void sample_temperature(llama_token_data_array * candidates_p, float temp)
{
    if (temp <= 0)
    {
        // Imitate greedy sampling
        temp = 0.00390625f; //cannot be zero else div0, this is 1/256
        llama_sample_temperature(nullptr, candidates_p, temp);
        llama_sample_top_k(nullptr, candidates_p, 1, 1); //only want first candidate
    }
    else
    {
        llama_sample_temperature(nullptr, candidates_p, temp);
    }
}

int SampleLogits(const float * logits, int n_ctx, int n_vocab, int rep_pen_range, float rep_pen, float top_k, float top_a, float top_p, float typical_p, float tfs, float temp, std::mt19937 & rng,
int mirostat, float mirostat_tau, float mirostat_eta, const std::vector<samplers> & sampler_order)
{
    int id = 0;
    std::vector<llama_token_data> candidates;
    candidates.reserve(n_vocab);
    for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
        candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
    }

    llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

    for (auto & sr_params : params.seqrep_params) {
        if ((sr_params.flags & LLAMA_SEQREP_REWIND_MODE) != 0) continue;
        llama_sample_seqrep_penalty(llama_ctx_v4, &candidates_p, current_context_tokens.data(), current_context_tokens.size(), &sr_params);
    }

    if (mirostat == 1 || mirostat == 2)
    {
        static float mirostat_mu = 2.0f * mirostat_tau;
        const int mirostat_m = 100;
        sample_rep_pen(n_ctx, rep_pen_range, rep_pen, &candidates_p);
        sample_temperature(&candidates_p, temp);
        if (mirostat == 1)
        {
            id = sample_token_mirostat(n_vocab, &candidates_p, rng, mirostat_tau, mirostat_eta, mirostat_m, &mirostat_mu);
        }
        else
        {
            id = sample_token_mirostat_v2(&candidates_p, rng, mirostat_tau, mirostat_eta, &mirostat_mu);
        }
    }
    else
    {
        for (int i = 0; i < sampler_order.size(); i++)
        {
            switch (sampler_order[i])
            {
                case KCPP_SAMPLER_TOP_K:
                    llama_sample_top_k(nullptr, &candidates_p, top_k,1);
                    break;
                case KCPP_SAMPLER_TOP_A:
                    sample_top_a(&candidates_p,top_a,1);
                    break;
                case KCPP_SAMPLER_TOP_P:
                    llama_sample_top_p(nullptr, &candidates_p, top_p,1);
                    break;
                case KCPP_SAMPLER_TFS:
                    llama_sample_tail_free(nullptr, &candidates_p, tfs,1);
                    break;
                case KCPP_SAMPLER_TYP:
                    llama_sample_typical(nullptr, &candidates_p, typical_p,1);
                    break;
                case KCPP_SAMPLER_TEMP:
                    sample_temperature(&candidates_p, temp);
                    break;
                case KCPP_SAMPLER_REP_PEN:
                    sample_rep_pen(n_ctx, rep_pen_range, rep_pen, &candidates_p);
                    break;
                default:
                    printf("\nSampleLogits: Unknown Sampler : %d",sampler_order[i]);
                    break;
            }
        }
        id = sample_token(&candidates_p, rng);
    }

    return id;
}

static std::string FileFormatTokenizeID(int id, FileFormat file_format)
{
    if (file_format == FileFormat::GGML || file_format == FileFormat::GGHF || file_format == FileFormat::GGJT || file_format == FileFormat::GGJT_2)
    {
        return std::string(llama_v2_token_to_str(llama_ctx_v2, id));
    }
    else if (file_format == FileFormat::GGJT_3)
    {
        return std::string(llama_v3_token_to_str(llama_ctx_v3, id));
    }
    else if(file_format == FileFormat::GGUF_LLAMA || file_format==FileFormat::GGUF_FALCON)
    {
        return std::string(llama_token_to_str(llama_ctx_v4, id));
    }
    else
    {
        return vocab.id_to_token[id];
    }
}

static void TokenizeString(const std::string & str_to_tokenize, std::vector<int> & output_tokens, FileFormat file_format)
{
    if (file_format == FileFormat::GGML || file_format == FileFormat::GGHF || file_format == FileFormat::GGJT || file_format == FileFormat::GGJT_2  || file_format == FileFormat::GGJT_3 || file_format == FileFormat::GGUF_LLAMA || file_format==FileFormat::GGUF_FALCON)
    {
        if(file_format == FileFormat::GGHF || file_format == FileFormat::GGJT || file_format == FileFormat::GGJT_2 )
        {
            output_tokens = ::llama_v2_tokenize(llama_ctx_v2, str_to_tokenize, true);
        }
        else if (file_format == FileFormat::GGML)
        {
            output_tokens = ::legacy_llama_v2_tokenize(llama_ctx_v2, str_to_tokenize, true);
        }
        else if (file_format == FileFormat::GGJT_3)
        {
            output_tokens = ::llama_v3_tokenize(llama_ctx_v3, str_to_tokenize, true);
        }
        else
        {
            output_tokens = ::llama_tokenize(llama_ctx_v4, str_to_tokenize, true);
        }
    }
    else
    {
        // tokenize the prompt
        output_tokens = ::gpt_tokenize(vocab, str_to_tokenize);
    }
}

static float LowestLogit(const std::vector<float> & logits)
{
    int topid = std::min_element(logits.begin(), logits.end()) - logits.begin();
    float v = logits[topid];
    return (v < 0 ? (v-8) : 0);
}
static float LowestLogit(const float *logits, size_t size)
{
    if (size == 0) {
        // Handle the case of an empty array
        return 0.0;
    }
    int topid = std::min_element(logits, logits + size) - logits;
    float v = logits[topid];
    return (v < 0 ? (v-8) : 0);
}

static std::string RemoveBell(const std::string & input) //removes the bell character
{
    std::string word2;
    std::remove_copy(input.begin(), input.end(), std::back_inserter(word2), '\a');
    return word2;
}

ModelLoadResult gpttype_load_model(const load_model_inputs inputs, FileFormat in_file_format, FileFormatExtraMeta file_format_meta)
{
    ggml_time_init();

    file_format = in_file_format;
    n_threads = params.n_threads = inputs.threads;
    n_blasthreads = inputs.blasthreads;
    n_batch = params.n_batch = inputs.batch_size;
    modelname = params.model = inputs.model_filename;
    useSmartContext = inputs.use_smartcontext;
    debugmode = inputs.debugmode;
    unbanTokens = inputs.unban_tokens;
    blasbatchsize = inputs.blasbatchsize;
    if(blasbatchsize<=0)
    {
        blasbatchsize = 8;
    }
    params.memory_f16 = inputs.f16_kv;
    params.n_ctx = inputs.max_context_length;

    neox_ctx_v2.hparams.n_ctx  = neox_ctx_v3.hparams.n_ctx
    = gptj_ctx_v1.hparams.n_ctx = gptj_ctx_v2.hparams.n_ctx = gptj_ctx_v3.hparams.n_ctx
    = gpt2_ctx_v1.hparams.n_ctx = gpt2_ctx_v2.hparams.n_ctx = gpt2_ctx_v3.hparams.n_ctx
    = mpt_ctx_v3.hparams.n_ctx = params.n_ctx;

    //determine rope scaling params
    float rope_freq_scale = 1.0f;
    float rope_freq_base = 10000.0f;
    if(inputs.rope_freq_scale>0.0f)
    {
        rope_freq_scale = inputs.rope_freq_scale;
        rope_freq_base = inputs.rope_freq_base;
        printf("Using Custom RoPE scaling (scale:%.3f, base:%.1f).\n",rope_freq_scale,rope_freq_base);
    }
    else
    {
        rope_freq_scale = 1.0f;
        if (params.n_ctx <= 2048) //normie mode
        {
            rope_freq_base = 10000.0f;
        }
        else
        {
            //approximate NTK aware ctx
            auto effectivenctx = params.n_ctx;
            if((file_format == FileFormat::GGUF_LLAMA || file_format==FileFormat::GGUF_FALCON) && file_format_meta.n_ctx_train > 2048)
            {
                float factor = file_format_meta.n_ctx_train/2048;
                effectivenctx = effectivenctx/factor;
            }
            rope_freq_base = (effectivenctx <= 2048 ? 10000.0f : (effectivenctx <= 3072 ? 26000.0f : (effectivenctx <= 4096 ? 32000.0f : (effectivenctx <= 6144 ? 54000.0f : (effectivenctx <= 8192 ? 82684.0f : (effectivenctx <= 12288 ? 140000.0f : 200000.0f))))));

        }

        printf("Using automatic RoPE scaling (scale:%.3f, base:%.1f)\n",rope_freq_scale,rope_freq_base);
    }
    gptj_ctx_v3.hparams.rope_freq_scale = neox_ctx_v3.hparams.rope_freq_scale = rope_freq_scale;
    gptj_ctx_v3.hparams.rope_freq_base = neox_ctx_v3.hparams.rope_freq_base = rope_freq_base;

    //handle custom token bans
    banned_tokens.clear();
    for(int x=0;x<ban_token_max;++x)
    {
        std::string word = inputs.banned_tokens[x];
        if(word!="")
        {
            banned_tokens.push_back(word);
        }
    }

    // debug print seqrep params
    printf("\n\n### SeqRep Params: %s\n", inputs.seqrep_params);
    if (inputs.seqrep_params != "") {
        if (std::strcmp(inputs.seqrep_params, "help") == 0) {
            seqrep_sampler_help();
            return ModelLoadResult::FAIL;
        }
        llama_sampler_seqrep_params sr_params;
        seqrep_sampler_params_init(&sr_params);
        if (!seqrep_sampler_params_parse(inputs.seqrep_params, &sr_params)) {
            seqrep_sampler_help();
            fprintf(stderr, "%s: error: failed to parse seqrep params\n", __func__);
            return ModelLoadResult::FAIL;
        }
        params.seqrep_params.push_back(sr_params);
        seqrep_sampler_params_dump(&sr_params);
    }

    //this is used for the mem_per_token eval, openblas needs more RAM
    bool use_scratch = ggml_cpu_has_gpublas();

    int cu_parseinfo_maindevice = inputs.cublas_info<=0?0:inputs.cublas_info;

    printf("System Info: %s\n", llama_print_system_info());
    #if defined(GGML_USE_CUBLAS)
    if(ggml_cpu_has_gpublas() && cu_parseinfo_maindevice>0)
    {
        printf("CUBLAS: Set main device to %d\n",cu_parseinfo_maindevice);
        ggml_cuda_set_main_device(cu_parseinfo_maindevice);
    }
    #endif
    SetQuantsUnshuffled(false);
    if(file_format == FileFormat::GGML || file_format == FileFormat::GGHF || file_format == FileFormat::GGJT || file_format == FileFormat::GGJT_2)
    {
        //newer format has bit unshuffling
        SetQuantsUnshuffled(file_format == FileFormat::GGJT_2);
        llama_v2_context_params llama_ctx_params_v2 = llama_v2_context_default_params();
        llama_ctx_params_v2.n_ctx = inputs.max_context_length;
        //llama_ctx_params.n_parts = -1;
        llama_ctx_params_v2.seed = -1;
        llama_ctx_params_v2.f16_kv = inputs.f16_kv;
        llama_ctx_params_v2.logits_all = false;
        llama_ctx_params_v2.use_mmap = inputs.use_mmap;
        llama_ctx_params_v2.use_mlock = inputs.use_mlock;
        llama_ctx_params_v2.n_gpu_layers = inputs.gpulayers;

        llama_ctx_v2 = llama_v2_init_from_file(modelname.c_str(), llama_ctx_params_v2);

        if (llama_ctx_v2 == NULL)
        {
            fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, modelname.c_str());
            return ModelLoadResult::FAIL;
        }

        printf("\n---\nWarning: Your model may be an OUTDATED format (ver %d). Please reconvert it for better results!\n---\n", file_format);

        if (lora_filename != "")
        {
            printf("\nAttempting to apply LORA adapter: %s\n", lora_filename.c_str());

            const char * lora_base_arg = NULL;
            if (lora_base != "") {
                printf("Using LORA base model: %s\n", lora_base.c_str());
                lora_base_arg = lora_base.c_str();
            }

            int err = llama_v2_apply_lora_from_file(llama_ctx_v2,
                                                 lora_filename.c_str(),
                                                 lora_base_arg,
                                                 n_threads);
            if (err != 0)
            {
                fprintf(stderr, "%s: error: failed to apply lora adapter\n", __func__);
                return ModelLoadResult::FAIL;
            }
        }

        //determine mem per token
        const std::vector<int> tmp = {1, 2, 3, 4};
        llama_v2_eval(llama_ctx_v2, tmp.data(), tmp.size(), 0, params.n_threads);
        return ModelLoadResult::SUCCESS;
    }
    else if(file_format == FileFormat::GGJT_3)
    {
        llama_v3_context_params llama_ctx_params = llama_v3_context_default_params();
        llama_ctx_params.n_ctx = inputs.max_context_length;
        //llama_ctx_paran_parts = -1;
        llama_ctx_params.seed = -1;
        llama_ctx_params.f16_kv = inputs.f16_kv;
        llama_ctx_params.low_vram = inputs.low_vram;
        llama_ctx_params.mul_mat_q = inputs.use_mmq;
        llama_ctx_params.logits_all = false;
        llama_ctx_params.use_mmap = inputs.use_mmap;
        llama_ctx_params.use_mlock = inputs.use_mlock;
        llama_ctx_params.n_gpu_layers = inputs.gpulayers;
        llama_ctx_params.main_gpu = cu_parseinfo_maindevice;
        llama_ctx_params.rope_freq_base = rope_freq_base;
        llama_ctx_params.rope_freq_scale = rope_freq_scale;
        llama_ctx_params.n_batch = blasbatchsize;

        #if defined(GGML_USE_CUBLAS)
        bool ts_all_zero = true;
        for (int i = 0; i < tensor_split_max; ++i) {
            if (inputs.tensor_split[i] != 0.0f) {
                ts_all_zero = false;
                break;
            }
        }
        if(!ts_all_zero)
        {
            llama_ctx_params.tensor_split = inputs.tensor_split;
        }
        #endif

        llama_ctx_v3 = llama_v3_init_from_file(modelname.c_str(), llama_ctx_params);

        if (llama_ctx_v3 == NULL)
        {
            fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, modelname.c_str());
            return ModelLoadResult::FAIL;
        }
        if (lora_filename != "")
        {
            printf("\nAttempting to apply LORA adapter: %s\n", lora_filename.c_str());

            const char * lora_base_arg = NULL;
            if (lora_base != "") {
                printf("Using LORA base model: %s\n", lora_base.c_str());
                lora_base_arg = lora_base.c_str();
            }

            int err = llama_v3_apply_lora_from_file(llama_ctx_v3,
                                                 lora_filename.c_str(),
                                                 lora_base_arg,
                                                 n_threads);
            if (err != 0)
            {
                fprintf(stderr, "%s: error: failed to apply lora adapter\n", __func__);
                return ModelLoadResult::FAIL;
            }
        }

        //determine mem per token
        const std::vector<int> tmp = {1, 2, 3, 4};
        auto er = llama_v3_eval(llama_ctx_v3, tmp.data(), tmp.size(), 0, params.n_threads);
        if(er!=0)
        {
            printf("\nLLAMA EVAL returned nonzero!\n");
        }
        return ModelLoadResult::SUCCESS;
    }
    else if(file_format==FileFormat::GGUF_LLAMA || file_format==FileFormat::GGUF_FALCON)
    {
        llama_context_params llama_ctx_params = llama_context_default_params();
        llama_ctx_params.n_ctx = inputs.max_context_length;
        //llama_ctx_paran_parts = -1;
        llama_ctx_params.seed = -1;
        llama_ctx_params.f16_kv = inputs.f16_kv;
        llama_ctx_params.low_vram = inputs.low_vram;
        llama_ctx_params.mul_mat_q = inputs.use_mmq;
        llama_ctx_params.logits_all = false;
        llama_ctx_params.use_mmap = inputs.use_mmap;
        llama_ctx_params.use_mlock = inputs.use_mlock;
        llama_ctx_params.n_gpu_layers = inputs.gpulayers;
        #if defined(GGML_USE_CLBLAST)
        if(file_format==FileFormat::GGUF_FALCON && llama_ctx_params.n_gpu_layers>0)
        {
            printf("\nGPU layer offload for GGUF FALCON on OpenCL is known to have issues, it has been set to 0.\n");
            llama_ctx_params.n_gpu_layers = 0;
        }
        #endif
        llama_ctx_params.main_gpu = cu_parseinfo_maindevice;
        llama_ctx_params.rope_freq_base = rope_freq_base;
        llama_ctx_params.rope_freq_scale = rope_freq_scale;
        llama_ctx_params.n_batch = blasbatchsize;

        #if defined(GGML_USE_CUBLAS)
        bool ts_all_zero = true;
        for (int i = 0; i < tensor_split_max; ++i) {
            if (inputs.tensor_split[i] != 0.0f) {
                ts_all_zero = false;
                break;
            }
        }
        if(!ts_all_zero)
        {
            llama_ctx_params.tensor_split = inputs.tensor_split;
        }
        #endif

        llama_ctx_v4 = llama_init_from_file(modelname.c_str(), llama_ctx_params);

        if (llama_ctx_v4 == NULL)
        {
            fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, modelname.c_str());
            return ModelLoadResult::FAIL;
        }
        if (lora_filename != "")
        {
            printf("\nAttempting to apply LORA adapter: %s\n", lora_filename.c_str());

            const char * lora_base_arg = NULL;
            if (lora_base != "") {
                printf("Using LORA base model: %s\n", lora_base.c_str());
                lora_base_arg = lora_base.c_str();
            }

            int err = llama_apply_lora_from_file(llama_ctx_v4,
                                                 lora_filename.c_str(),
                                                 lora_base_arg,
                                                 n_threads);
            if (err != 0)
            {
                fprintf(stderr, "%s: error: failed to apply lora adapter\n", __func__);
                return ModelLoadResult::FAIL;
            }
        }

        //determine mem per token
        const std::vector<int> tmp = {1, 2, 3, 4};
        auto er = llama_eval(llama_ctx_v4, tmp.data(), tmp.size(), 0, params.n_threads);
        if(er!=0)
        {
            printf("\nLLAMA EVAL returned nonzero!\n");
        }
        return ModelLoadResult::SUCCESS;
    }
    else if (file_format == FileFormat::RWKV_1 || file_format==FileFormat::RWKV_2)
    {
        //start loading the models first
        bool useWorldTokenizer = false;
        if (file_format == FileFormat::RWKV_1)
        {
            rwkv_ctx_v2 = rwkv_v2_init_from_file(modelname.c_str(), n_threads);
        }
        else //rwkv_2
        {
            rwkv_ctx_v3 = rwkv_init_from_file(modelname.c_str(), n_threads);

            if(inputs.gpulayers>0)
            {
                rwkv_gpu_offload_layers(rwkv_ctx_v3,inputs.gpulayers);
            }

            const struct rwkv_file_header & header = rwkv_ctx_v3->instance->model.header;
            const size_t n_vocab = header.n_vocab;
            printf("\nDetected Vocab: %zu",n_vocab);
            if(n_vocab>60000)
            {
                printf("\nUsing WORLD TOKENIZER");
                useWorldTokenizer = true;
            }
        }

        std::string word;
        if(useWorldTokenizer)
        {
            read_rwkv_world_vocab();
        }
        else
        {
            read_rwkv_vocab();
        }

        int vocabsiz = rwkv_vocab.size();
        for (int i = 0; i < vocabsiz; i++)
        {
            uint32_t len;
            word = rwkv_vocab[i];
            vocab.token_to_id[word] = i;
            vocab.id_to_token[i] = word;
        }
        printf("\nRWKV Vocab: %u\n", vocabsiz);
        logits.resize(vocabsiz);

        if (file_format == FileFormat::RWKV_1)
        {
            n_batch = 1;

            //setup buffers for rwkv state
            auto padding = 512u;
            auto statebufsiz = rwkv_v2_get_state_buffer_element_count(rwkv_ctx_v2) * sizeof(float) + padding;
            auto logitbufsiz = rwkv_v2_get_logits_buffer_element_count(rwkv_ctx_v2) * sizeof(float) + padding;

            printf("\nRWKV old Init: State Buffer:%lu, Logit Buffer:%lu\n", statebufsiz, logitbufsiz);
            rwkv_ctx_v2->state_out = (float *)malloc(statebufsiz);
            rwkv_ctx_v2->logits_out = (float *)malloc(logitbufsiz);
            rwkv_ctx_v2->state_in = nullptr;

            bool testeval = rwkv_v2_eval(rwkv_ctx_v2, 0, rwkv_ctx_v2->state_in, rwkv_ctx_v2->state_out, rwkv_ctx_v2->logits_out);
            if (!testeval)
            {
                printf("\nError: RWKV old Init Eval Failed!\n");
            }

            memcpy(logits.data(), rwkv_ctx_v2->logits_out, sizeof(float) * vocabsiz);

            if (rwkv_ctx_v2 == NULL)
            {
                return ModelLoadResult::FAIL;
            }
            return ModelLoadResult::SUCCESS;
        }
        else
        {
            n_batch = 1; //do not use sequence mode to speedup until it is fixed

            //setup buffers for rwkv state
            auto padding = 512u;
            auto statebufsiz = rwkv_get_state_buffer_element_count(rwkv_ctx_v3) * sizeof(float) + padding;
            auto logitbufsiz = rwkv_get_logits_buffer_element_count(rwkv_ctx_v3) * sizeof(float) + padding;

            printf("\nRWKV Init: State Buffer:%lu, Logit Buffer:%lu\n", statebufsiz, logitbufsiz);
            rwkv_ctx_v3->state_out = (float *)malloc(statebufsiz);
            rwkv_ctx_v3->logits_out = (float *)malloc(logitbufsiz);
            rwkv_ctx_v3->state_in = nullptr;

            bool testeval = rwkv_eval(rwkv_ctx_v3, params.n_threads, 0, rwkv_ctx_v3->state_in, rwkv_ctx_v3->state_out, rwkv_ctx_v3->logits_out);
            if (!testeval)
            {
                printf("\nError: RWKV Init Eval Failed!\n");
            }

            memcpy(logits.data(), rwkv_ctx_v3->logits_out, sizeof(float) * vocabsiz);

            if (rwkv_ctx_v3 == NULL)
            {
                return ModelLoadResult::FAIL;
            }
            return ModelLoadResult::SUCCESS;
        }
    }
    else if (file_format == FileFormat::GPT2_1)
    {
        ModelLoadResult res = legacy_gpt2_model_load(params.model, gpt2_ctx_v1, vocab, file_format);
        if(res==ModelLoadResult::FAIL)
        {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
            return res;
        }
        else if(res==ModelLoadResult::RETRY_LOAD)
        {
            printf("\nTensor Transposition Detected! Retrying GPT-2 model loading...");
            return res;
        }
         // determine the required inference memory per token:
        legacy_gpt2_eval(gpt2_ctx_v1, params.n_threads, 0, { 0, 1, 2, 3 }, logits, mem_per_token, file_format);
        return ModelLoadResult::SUCCESS;
    }
    else if (file_format == FileFormat::GPT2_2 || file_format==FileFormat::GPT2_3 || file_format==FileFormat::GPT2_4)
    {
        if(file_format==FileFormat::GPT2_4)
        {
            ModelLoadResult res = gpt2_model_load(params.model, gpt2_ctx_v3, vocab, file_format, inputs.gpulayers);
            if(res==ModelLoadResult::FAIL)
            {
                fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
                return res;
            }
            else if(res==ModelLoadResult::RETRY_LOAD)
            {
                printf("\nTensor Transposition Detected! Retrying GPT-2 model loading...");
                return res;
            }
            // determine the required inference memory per token:
            gpt2_eval(gpt2_ctx_v3, params.n_threads, 0, { 0, 1, 2, 3 }, logits, mem_per_token, use_scratch);
            return ModelLoadResult::SUCCESS;
        }
        else
        {
            //newer format has bit unshuffling
            SetQuantsUnshuffled(file_format == FileFormat::GPT2_3);

            ModelLoadResult res = gpt2_v2_model_load(params.model, gpt2_ctx_v2, vocab, file_format, inputs.gpulayers);
            if(res==ModelLoadResult::FAIL)
            {
                fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
                return res;
            }
            else if(res==ModelLoadResult::RETRY_LOAD)
            {
                printf("\nTensor Transposition Detected! Retrying GPT-2 model loading...");
                return res;
            }
            // determine the required inference memory per token:
            gpt2_v2_eval(gpt2_ctx_v2, params.n_threads, 0, { 0, 1, 2, 3 }, logits, mem_per_token, file_format);
            return ModelLoadResult::SUCCESS;
        }
    }
    else if (file_format == FileFormat::GPTJ_1 || file_format == FileFormat::GPTJ_2)
    {
        ModelLoadResult res = legacy_gptj_model_load(params.model, gptj_ctx_v1, vocab, file_format);
        if(res==ModelLoadResult::FAIL)
        {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
            return res;
        }
        else if(res==ModelLoadResult::RETRY_LOAD)
        {
            printf("\nTensor Transposition Detected! Retrying GPT-J model loading...");
            return res;
        }
         // determine the required inference memory per token:
        legacy_gptj_eval(gptj_ctx_v1, params.n_threads, 0, { 0, 1, 2, 3 }, logits, mem_per_token, file_format);

        //if the logits are NAN or duplicated, it means the model is incompatible
        if(logits.size()>0 && IsNanCheck(logits[0]))
        {
            printf("\nBad Logits detected! Retrying GPT-J model loading...");
            ggml_v1_free(gptj_ctx_v1.ctx);
            return ModelLoadResult::RETRY_LOAD;
        }

        return ModelLoadResult::SUCCESS;
    }
    else if(file_format == FileFormat::GPTJ_3 || file_format == FileFormat::GPTJ_4 || file_format == FileFormat::GPTJ_5)
    {
        if(file_format == FileFormat::GPTJ_5)
        {
            ModelLoadResult loadresult = gptj_model_load(params.model, gptj_ctx_v3, vocab, inputs.gpulayers);
            if (loadresult == ModelLoadResult::FAIL)
            {
                fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
                return loadresult;
            }
            else if (loadresult == ModelLoadResult::RETRY_LOAD)
            {
                printf("\nTensor Transposition Detected! Retrying GPT-J model loading...");
                return loadresult;
            }

            // determine the required inference memory per token:
            gptj_eval(gptj_ctx_v3, params.n_threads, 0, { 0, 1, 2, 3 }, logits, mem_per_token, use_scratch);

            //if the logits are NAN or duplicated, it means the model is incompatible
            std::vector<float> oldlogits(logits);

            //this is another hack because they change the library - we run the eval through the model
            //twice and compare logits. if they give the same logits for different inputs, model is broken
            gptj_eval(gptj_ctx_v3, params.n_threads, 0, {4, 5, 6, 7}, logits, mem_per_token, use_scratch);

            if(logits.size()>0 && (IsNanCheck(logits[0]) || LogitsDuplicated(oldlogits,logits)))
            {
                printf("\nBad Logits detected! Retrying GPT-J model loading...");
                ggml_free(gptj_ctx_v3.ctx);
                return ModelLoadResult::RETRY_LOAD;
            }

            return ModelLoadResult::SUCCESS;
        }
        else
        {
            //newer format has bit unshuffling
            SetQuantsUnshuffled(file_format == FileFormat::GPTJ_4);

            ModelLoadResult loadresult = gptj_v2_model_load(params.model, gptj_ctx_v2, vocab, inputs.gpulayers);
            if (loadresult == ModelLoadResult::FAIL)
            {
                fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
                return loadresult;
            }
            else if (loadresult == ModelLoadResult::RETRY_LOAD)
            {
                printf("\nTensor Transposition Detected! Retrying GPT-J model loading...");
                return loadresult;
            }

            // determine the required inference memory per token:
            gptj_v2_eval(gptj_ctx_v2, params.n_threads, 0, { 0, 1, 2, 3 }, logits, mem_per_token);

            //if the logits are NAN or duplicated, it means the model is incompatible
            std::vector<float> oldlogits(logits);

            //this is another hack because they change the library - we run the eval through the model
            //twice and compare logits. if they give the same logits for different inputs, model is broken
            gptj_v2_eval(gptj_ctx_v2, params.n_threads, 0, {4, 5, 6, 7}, logits, mem_per_token);

            if(logits.size()>0 && (IsNanCheck(logits[0]) || LogitsDuplicated(oldlogits,logits)))
            {
                printf("\nBad Logits detected! Retrying GPT-J model loading...");
                ggml_v2_free(gptj_ctx_v2.ctx);
                return ModelLoadResult::RETRY_LOAD;
            }

            return ModelLoadResult::SUCCESS;
        }
    }
    else if(file_format==FileFormat::NEOX_1 || file_format==FileFormat::NEOX_2 || file_format==FileFormat::NEOX_3 || file_format==FileFormat::NEOX_4 || file_format==FileFormat::NEOX_5|| file_format==FileFormat::NEOX_6|| file_format==FileFormat::NEOX_7)
    {
        if(file_format==FileFormat::NEOX_6|| file_format==FileFormat::NEOX_7)
        {
            ModelLoadResult res = gpt_neox_model_load(params.model, neox_ctx_v3, vocab, file_format, inputs.gpulayers);
            if(res==ModelLoadResult::FAIL)
            {
                fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
                return res;
            }
            else if(res==ModelLoadResult::RETRY_LOAD)
            {
                printf("\nIncorrect Tensor Size Detected! Retrying GPT-NeoX model loading...");
                return res;
            }

            // determine the required inference memory per token:
            gpt_neox_eval(neox_ctx_v3, params.n_threads, 0, { 0, 1, 2, 3 }, logits, mem_per_token, use_scratch);

            return ModelLoadResult::SUCCESS;
        }
        else
        {
            //newer format has bit unshuffling
            SetQuantsUnshuffled(file_format==FileFormat::NEOX_4 || file_format==FileFormat::NEOX_5);

            ModelLoadResult res = gpt_neox_v2_model_load(params.model, neox_ctx_v2, vocab, file_format);
            if(res==ModelLoadResult::FAIL)
            {
                fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
                return res;
            }
            else if(res==ModelLoadResult::RETRY_LOAD)
            {
                printf("\nIncorrect Tensor Size Detected! Retrying GPT-NeoX model loading...");
                return res;
            }

            // determine the required inference memory per token:
            gpt_neox_v2_eval(neox_ctx_v2, params.n_threads, 0, { 0, 1, 2, 3 }, logits, mem_per_token);

            if(logits.size()>0 && file_format==FileFormat::NEOX_2 && !IsNanCheck(logits[0]))
            {
                //run the black magic eval to determine if it's redpajama. VERY UGLY HACK!
                std::vector<int> test_embd = ::gpt_tokenize(vocab, "1 2 3 4 5 6 7");
                auto orig_par_res = neox_ctx_v2.hparams.par_res;
                neox_ctx_v2.hparams.par_res = 0; //test with residual false
                gpt_neox_v2_eval(neox_ctx_v2, params.n_threads, 0, test_embd, logits, mem_per_token);
                neox_ctx_v2.hparams.par_res = orig_par_res;
                int topid = std::max_element(logits.begin(),logits.end())-logits.begin();
                std::string predicted = vocab.id_to_token[topid].c_str();
                auto findresult = predicted.find("8");
                if(findresult != std::string::npos && findresult<2)
                {
                    printf("\n---\nOld RedPajama NeoX Detected! Switching to new format! (use_parallel_residual=False)\n");
                    ggml_v2_free(neox_ctx_v2.ctx);
                    return ModelLoadResult::RETRY_LOAD;
                }
            }

            return ModelLoadResult::SUCCESS;
        }

    }
    else if(file_format==FileFormat::MPT_1)
    {
        bool res = mpt_model_load(params.model, mpt_ctx_v3, vocab, inputs.gpulayers);
        if(res==false)
        {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
            return ModelLoadResult::FAIL;
        }

        // determine the required inference memory per token:
        mpt_eval(mpt_ctx_v3, params.n_threads, 0, { 0, 1, 2, 3 }, logits, false, mem_per_token, use_scratch);
        return ModelLoadResult::SUCCESS;
    }
    else
    {
        printf("\nUnknown Model, cannot load.\n");
        return ModelLoadResult::FAIL;
    }

}

bool gpttype_generate_abort()
{
    stopper_unused_tokens = remaining_tokens;
    remaining_tokens = 0;
    return true;
}

int gpttype_token_count(const std::string & input)
{
    if(debugmode==1)
    {
        printf("\nFileFormat: %d, Tokenizing: %s",file_format ,input.c_str());
    }
    std::vector<int> toks;
    TokenizeString(input, toks, file_format);
    int tokcount = toks.size();
    if(debugmode==1)
    {
        printf("\nTokens Counted: %d\n",tokcount);
    }
    return tokcount;
}

const std::string & gpttype_get_pending_output()
{
    concat_output_mtx.lock();
    concat_output_reader_copy = concat_output;
    concat_output_mtx.unlock();
    return concat_output_reader_copy;
}

generation_outputs gpttype_generate(const generation_inputs inputs, generation_outputs &output)
{
    concat_output_mtx.lock();
    concat_output = "";
    concat_output_mtx.unlock();
    last_stop_reason = stop_reason::OUT_OF_TOKENS;
    stop_sequence.clear();
    for(int x=0;x<stop_token_max;++x)
    {
        std::string stopper = inputs.stop_sequence[x];
        if(stopper!="")
        {
            stop_sequence.push_back(stopper);
        }
    }
    params.prompt = inputs.prompt;
    params.seed = inputs.seed;
    params.n_predict = inputs.max_length;
    params.top_k = inputs.top_k;
    params.top_p = inputs.top_p;
    params.typical_p = inputs.typical_p;
    params.tfs_z = inputs.tfs;
    params.temp = inputs.temperature;
    params.repeat_last_n = inputs.rep_pen_range;
    params.repeat_penalty = inputs.rep_pen;
    params.mirostat = inputs.mirostat;
    params.mirostat_eta = inputs.mirostat_eta;
    params.mirostat_tau = inputs.mirostat_tau;
    params.n_ctx = inputs.max_context_length;
    params.n_batch = n_batch;
    params.n_threads = n_threads;
    bool stream_sse = inputs.stream_sse;

    generation_finished = false; // Set current generation status
    generated_tokens.clear(); // New Generation, new tokens

    if (params.repeat_last_n < 1)
    {
        params.repeat_last_n = 1;
    }
    if (params.top_k < 1)
    {
        params.top_k = 120; //to disable top_k we actually need to increase this value to a very high number
    }
    if (params.seed <= 0 || params.seed==0xFFFFFFFF)
    {
        params.seed = time(NULL);
    }

    // tokenize the prompt
    std::vector<int> embd_inp;
    TokenizeString(params.prompt, embd_inp, file_format);

    //truncate to front of the prompt if its too long
    int32_t nctx = params.n_ctx;

    if (embd_inp.size() + params.n_predict > nctx)
    {
        int offset = embd_inp.size() - nctx + params.n_predict;
        embd_inp = std::vector<int>(embd_inp.begin() + offset, embd_inp.end());
    }

    //determine how much npast we have to rewind from the current state
    std::vector<gpt_vocab::id> embd;

    int last_n_size = nctx;
    // int last_n_size = params.repeat_last_n;
    last_n_tokens.resize(last_n_size);

    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);
    n_past = 0;

    if (file_format == FileFormat::RWKV_1 || file_format==FileFormat::RWKV_2)
    {
        ContextFastForward(current_context_tokens, embd_inp, n_past, last_n_tokens, nctx, smartcontext, false, true);
    }
    else
    {
        ContextFastForward(current_context_tokens, embd_inp, n_past, last_n_tokens, nctx, smartcontext, useSmartContext, false);
    }

    //if using BLAS and prompt is big enough, switch to single thread and use a huge batch
    bool approved_format = !(file_format == FileFormat::BADFORMAT ||
                            file_format == FileFormat::GPT2_1 ||
                            file_format == FileFormat::GPTJ_1 ||
                            file_format == FileFormat::GPTJ_2 ||
                            file_format == FileFormat::RWKV_1 ||
                            file_format==FileFormat::RWKV_2);
    bool blasmode = (approved_format && embd_inp.size() >= 32 && ggml_cpu_has_blas() && blasbatchsize!=-1);
    // bool blasmode = false;
    int original_batch = params.n_batch;
    int original_threads = params.n_threads;
    if (blasmode)
    {
        //for non llama, limit to 256
        int bbs = blasbatchsize;
        if (file_format != FileFormat::GGML && file_format != FileFormat::GGHF && file_format != FileFormat::GGJT && file_format != FileFormat::GGJT_2 && file_format != FileFormat::GGJT_3 && file_format != FileFormat::GGUF_LLAMA && file_format!=FileFormat::GGUF_FALCON)
        {
            bbs = (blasbatchsize > 256 ? 256 : blasbatchsize);
        }

        params.n_batch = bbs; //received reports of 1024 and above crashing on some models
        if(!ggml_cpu_has_gpublas())
        {
            params.n_threads = 1; //do not limit here anymore.
        }
        else
        {
            params.n_threads = n_blasthreads;
        }
    }

    current_context_tokens.resize(n_past);

    remaining_tokens = params.n_predict;
    stopper_unused_tokens = 0;
    int input_consumed = 0;
    std::mt19937 rng(params.seed);

    //prepare sampler order
    std::vector<samplers> sampler_order;
    if(inputs.sampler_len<=0) //list by value
    {
        sampler_order = {
            KCPP_SAMPLER_REP_PEN,
            KCPP_SAMPLER_TOP_K,
            KCPP_SAMPLER_TOP_A,
            KCPP_SAMPLER_TFS,
            KCPP_SAMPLER_TYP,
            KCPP_SAMPLER_TOP_P,
            KCPP_SAMPLER_TEMP
        };
    }
    else
    {
        for(int i=0;i<inputs.sampler_len;++i)
        {
            sampler_order.push_back(inputs.sampler_order[i]);
        }
    }

    bool startedsampling = false;
    bool use_scratch = true; //for normal inference always use scratch

    timer_start();
    double time1 = 0, time2 = 0;
    int32_t n_vocab = 0;

    if (file_format == FileFormat::GGML || file_format == FileFormat::GGHF || file_format == FileFormat::GGJT || file_format == FileFormat::GGJT_2)
    {
        n_vocab = llama_v2_n_vocab(llama_ctx_v2);
    }
    else if(file_format == FileFormat::GGJT_3)
    {
        n_vocab = llama_v3_n_vocab(llama_ctx_v3);
    }
    else if(file_format == FileFormat::GGUF_LLAMA || file_format==FileFormat::GGUF_FALCON)
    {
        n_vocab = llama_n_vocab(llama_ctx_v4);
    }
    else if (file_format == FileFormat::GPTJ_1 || file_format == FileFormat::GPTJ_2)
    {
        n_vocab = gptj_ctx_v1.hparams.n_vocab;
    }
    else if(file_format == FileFormat::GPTJ_3 || file_format==FileFormat::GPTJ_4)
    {
        n_vocab = gptj_ctx_v2.hparams.n_vocab;
    }
    else if(file_format==FileFormat::GPTJ_5)
    {
        n_vocab = gptj_ctx_v3.hparams.n_vocab;
    }
    else if(file_format == FileFormat::GPT2_1)
    {
        n_vocab = gpt2_ctx_v1.hparams.n_vocab;
    }
    else if(file_format == FileFormat::GPT2_2 || file_format==FileFormat::GPT2_3)
    {
        n_vocab = gpt2_ctx_v2.hparams.n_vocab;
    }
    else if(file_format==FileFormat::GPT2_4)
    {
        n_vocab = gpt2_ctx_v3.hparams.n_vocab;
    }
    else if(file_format == FileFormat::NEOX_1 || file_format == FileFormat::NEOX_2 || file_format == FileFormat::NEOX_3 || file_format==FileFormat::NEOX_4 || file_format==FileFormat::NEOX_5)
    {
        n_vocab = neox_ctx_v2.hparams.n_vocab;
    }
    else if( file_format==FileFormat::NEOX_6|| file_format==FileFormat::NEOX_7)
    {
        n_vocab = neox_ctx_v3.hparams.n_vocab;
    }
    else if( file_format==FileFormat::MPT_1)
    {
        n_vocab = mpt_ctx_v3.hparams.n_vocab;
    }
    else if(file_format == FileFormat::RWKV_1 || file_format==FileFormat::RWKV_2)
    {
        n_vocab = vocab.id_to_token.size(); //handled seperately
        if(n_past==0)
        {
            if(file_format == FileFormat::RWKV_1)
            {
                rwkv_ctx_v2->state_in = nullptr;
            }
            else
            {
                rwkv_ctx_v3->state_in = nullptr;
            }
        }
        else
        {
            if (file_format == FileFormat::RWKV_1)
            {
                rwkv_ctx_v2->state_in = rwkv_ctx_v2->state_out;
            }
            else
            {
                rwkv_ctx_v3->state_in = rwkv_ctx_v3->state_out;
            }

            //if it's empty, push in the final previous token
            if(embd_inp.size()==0 && current_context_tokens.size()>0)
            {
                embd_inp.push_back(current_context_tokens[current_context_tokens.size()-1]);
                current_context_tokens.pop_back();
            }
        }
    }
    else
    {
        printf("Bad format!");
    }

    //prepare banned tokens
    if(banned_token_ids.size()==0 && banned_tokens.size()>0)
    {
        printf("\n[First Run] Banning %zu token sequences...",banned_tokens.size());
        for(int v=0;v<n_vocab;++v)
        {
            std::string word = FileFormatTokenizeID(v,file_format);
            for(int i=0;i<banned_tokens.size();++i)
            {
                if (word.find(banned_tokens[i]) != std::string::npos)
                {
                    banned_token_ids.push_back(v);
                    break;
                }
            }
        }
        printf("\nBanned a total of %zu tokens.\n",banned_token_ids.size());
    }

    if(debugmode!=-1)
    {
        printf("\n");
    }

    if (debugmode==1)
    {
        std::string outstr = "";
        printf("\n[Debug: Dump Input Tokens, format: %d]\n", file_format);

        std::string tmp = "";
        for (auto id : embd_inp)
        {
            tmp += "'" + FileFormatTokenizeID(id, file_format) + " (" + std::to_string(id) + ")', ";
        }
        ::utreplace(tmp, "\n", "\\n");
        outstr += tmp;

        outstr += "\n\n[Debug: Context Size = " + std::to_string(current_context_tokens.size()) + "]\n";
        tmp = "";
        for (auto id : current_context_tokens)
        {
            tmp += "'" + FileFormatTokenizeID(id, file_format) + " (" + std::to_string(id) + ")', ";
        }
        ::utreplace(tmp, "\n", "\\n");
        outstr += tmp;
        printf("%s\n\n", RemoveBell(outstr).c_str());
    }

    // llama_sampler_seqrep_params sr_params =
        // llama_seqrep_merge_params(params.seqrep_params, LLAMA_SEQREP_REWIND_MODE, 0);
    // seqrep_sampler_params_dump(&sr_params);

    size_t high_water_mark = 0;
    size_t prompt_size = embd_inp.size();
    std::vector<char> rewind_token_text_buf(128, 0);
    struct seqrep_rewind_state rewind_state(llama_n_vocab(llama_ctx_v4), nctx, 2000, 0);

    // rewind_state.set_logits_slot(llama_ctx_v4, 0);

    while (remaining_tokens > 0)
    {
        gpt_vocab::id id = 0;
        // predict
        unsigned int embdsize = embd.size();
        //print progress
        if (!startedsampling && debugmode!=-1)
        {
            printf("\rProcessing Prompt%s (%d / %zu tokens)", (blasmode ? " [BLAS]" : ""), input_consumed, embd_inp.size());
        }
        fflush(stdout);

        if (startedsampling) {
            // print current token
            printf("%s", FileFormatTokenizeID(current_context_tokens.back(), file_format).c_str());
            // // print embd
            // printf("\n[%zu, %s]\n", embd.size(), FileFormatTokenizeID(embd.back(), file_format).c_str());

            const size_t rewind_distance =
                llama_seqrep_handle_rewind(
                    llama_ctx_v4, rewind_state, current_context_tokens, prompt_size,
                    params.seqrep_params, &high_water_mark);

            if (rewind_distance > 0) {
                const size_t idx = current_context_tokens.size() - rewind_distance;
                const llama_token nl_id = llama_token_nl(llama_ctx_v4);

                // get rewound tokens as a string, then trim off the end of concat_output
                std::string rewound_tokens_str = "";
                for (size_t i = idx; i < current_context_tokens.size(); i++) {
                    if (current_context_tokens[i] == nl_id) {
                        rewound_tokens_str += "\\n";
                        continue;
                    }
                    if (rewound_tokens_str != "") {
                        rewound_tokens_str += "|";
                    }
                    const std::string token_str = llama_token_to_piece(llama_ctx_v4, current_context_tokens[i]);
                    rewound_tokens_str += token_str;
                }
                current_context_tokens.resize(idx);
                remaining_tokens += rewind_distance;
                n_past -= rewind_distance;

                // reset embd to last current_context_token
                embd.clear();
                embd.push_back(current_context_tokens.back());

                concat_output_mtx.lock();
                // regenerate concat_output from current_context_tokens
                concat_output = "";
                for (size_t i = prompt_size; i < current_context_tokens.size(); i++) {
                    const std::string token_str = llama_token_to_piece(llama_ctx_v4, current_context_tokens[i]);
                    concat_output += token_str;
                }
                concat_output_mtx.unlock();

                // print rewound tokens surrounded by brackets
                printf("\n[");
                printf("%s", rewound_tokens_str.c_str());
                printf("]\n");
            }
        }

        if (embdsize > 0)
        {
            bool evalres = false;

            if (file_format == FileFormat::GGML || file_format == FileFormat::GGHF || file_format == FileFormat::GGJT || file_format == FileFormat::GGJT_2)
            {
                evalres = (llama_v2_eval(llama_ctx_v2, embd.data(), embdsize, n_past, params.n_threads)==0);
            }
            else if(file_format == FileFormat::GGJT_3)
            {
                evalres = (llama_v3_eval(llama_ctx_v3, embd.data(), embdsize, n_past, params.n_threads)==0);
            }
            else if(file_format == FileFormat::GGUF_LLAMA || file_format==FileFormat::GGUF_FALCON)
            {
                evalres = (llama_eval(llama_ctx_v4, embd.data(), embdsize, n_past, params.n_threads)==0);
            }
            else if(file_format==FileFormat::RWKV_1 || file_format==FileFormat::RWKV_2)
            {
                if (file_format == FileFormat::RWKV_1)
                {
                    evalres = rwkv_v2_eval(rwkv_ctx_v2, embd[0], rwkv_ctx_v2->state_in, rwkv_ctx_v2->state_out, rwkv_ctx_v2->logits_out);
                    memcpy(logits.data(), rwkv_ctx_v2->logits_out, sizeof(float) * rwkv_vocab.size());
                    rwkv_ctx_v2->state_in = rwkv_ctx_v2->state_out;
                }
                else
                {
                    if(embd.size()>1)
                    {
                        evalres = rwkv_eval_sequence(rwkv_ctx_v3, params.n_threads, (uint32_t*)embd.data(), embd.size(), rwkv_ctx_v3->state_in, rwkv_ctx_v3->state_out, rwkv_ctx_v3->logits_out);
                    }
                    else
                    {
                    bool ignoreLogits = (!startedsampling && ((int)embd_inp.size() > input_consumed + 2));
                    evalres = rwkv_eval(rwkv_ctx_v3, params.n_threads, embd[0], rwkv_ctx_v3->state_in, rwkv_ctx_v3->state_out, ignoreLogits?nullptr:rwkv_ctx_v3->logits_out);
                    }

                    memcpy(logits.data(), rwkv_ctx_v3->logits_out, sizeof(float) * rwkv_vocab.size());
                    rwkv_ctx_v3->state_in = rwkv_ctx_v3->state_out;
                }
            }
            else if(file_format==FileFormat::GPT2_1)
            {
                evalres = legacy_gpt2_eval(gpt2_ctx_v1, params.n_threads, n_past, embd, logits, mem_per_token, file_format);
            }
            else if(file_format==FileFormat::GPT2_2 || file_format==FileFormat::GPT2_3)
            {
                evalres = gpt2_v2_eval(gpt2_ctx_v2, params.n_threads, n_past, embd, logits, mem_per_token, file_format);
            }
            else if(file_format==FileFormat::GPT2_4)
            {
                evalres = gpt2_eval(gpt2_ctx_v3, params.n_threads, n_past, embd, logits, mem_per_token, use_scratch);
            }
            else if(file_format==FileFormat::NEOX_1 || file_format == FileFormat::NEOX_2 || file_format == FileFormat::NEOX_3 || file_format==FileFormat::NEOX_4 || file_format==FileFormat::NEOX_5)
            {
                evalres = gpt_neox_v2_eval(neox_ctx_v2, params.n_threads, n_past, embd, logits, mem_per_token);
            }
            else if(file_format==FileFormat::NEOX_6|| file_format==FileFormat::NEOX_7)
            {
                evalres = gpt_neox_eval(neox_ctx_v3, params.n_threads, n_past, embd, logits, mem_per_token, use_scratch);
            }
            else if(file_format==FileFormat::GPTJ_1 || file_format==FileFormat::GPTJ_2)
            {
                evalres = legacy_gptj_eval(gptj_ctx_v1, params.n_threads, n_past, embd, logits, mem_per_token, file_format);
            }
            else if(file_format==FileFormat::GPTJ_3 || file_format==FileFormat::GPTJ_4)
            {
                evalres = gptj_v2_eval(gptj_ctx_v2, params.n_threads, n_past, embd, logits, mem_per_token);
            }
            else if(file_format==FileFormat::GPTJ_5)
            {
                evalres = gptj_eval(gptj_ctx_v3, params.n_threads, n_past, embd, logits, mem_per_token, use_scratch);
            }
            else if(file_format==FileFormat::MPT_1)
            {
                evalres = mpt_eval(mpt_ctx_v3, params.n_threads, n_past, embd, logits, false, mem_per_token, use_scratch);
            }
            else
            {
                printf("\nCannot find eval function\n");
            }

            if (!evalres)
            {
                fprintf(stderr, "Failed to predict\n");
                snprintf(output.text, sizeof(output.text), "%s", "");
                output.status = 0;
                generation_finished = true;
                return output;
            }
        }

        // printf("\ncurrent_context_tokens.size() = %zu\n", current_context_tokens.size());
        // printf("\nprompt_size = %zu\n", prompt_size);

        n_past += embd.size();
        embd.clear();
        if ((int)embd_inp.size() <= input_consumed)
        {
            // out of user input, sample next token
            const float top_k = params.top_k;
            const float top_p = params.top_p;
            const float temp = params.temp;
            const float top_a = inputs.top_a;
            const float repeat_penalty = params.repeat_penalty;
            const float typical_p = params.typical_p;
            const float tfs_z = params.tfs_z;

            if (!startedsampling)
            {
                startedsampling = true;
                params.n_batch = original_batch;
                params.n_threads = original_threads;
                prompt_size = current_context_tokens.size();
                high_water_mark = current_context_tokens.size();
                rewind_state.set_high_water_mark(high_water_mark - 1);
                rewind_state.set_logits_slot(llama_ctx_v4, high_water_mark - 1);
                time1 = timer_check();
                timer_start();
                if(debugmode!=-1)
                {
                    printf("\n");
                }
            }

            unsigned int eosID = 0;
            float * logitsPtr;
            int btsize = banned_token_ids.size();
            if(file_format == FileFormat::GGML || file_format == FileFormat::GGHF || file_format == FileFormat::GGJT || file_format == FileFormat::GGJT_2 || file_format == FileFormat::GGJT_3 || file_format == FileFormat::GGUF_LLAMA || file_format==FileFormat::GGUF_FALCON)
            {
                if(file_format == FileFormat::GGUF_LLAMA || file_format==FileFormat::GGUF_FALCON)
                {
                    logitsPtr = llama_get_logits(llama_ctx_v4);
                    eosID = llama_token_eos(llama_ctx_v4);
                }
                else if(file_format == FileFormat::GGJT_3)
                {
                    logitsPtr = llama_v3_get_logits(llama_ctx_v3);
                    eosID = llama_v3_token_eos();
                }
                else
                {
                    logitsPtr = llama_v2_get_logits(llama_ctx_v2);
                    eosID = llama_v3_token_eos();
                }

                float lowestLogit = LowestLogit(logitsPtr,n_vocab);
                if (!unbanTokens && !inputs.unban_tokens_rt)
                {
                    // set the logit of the eos token (2) to -INF to avoid sampling it
                    logitsPtr[eosID] = lowestLogit;
                }

                if(btsize>0)
                {
                    for(int t=0;t<btsize;++t)
                    {
                        logitsPtr[banned_token_ids[t]]=lowestLogit;
                    }
                }
            }
            else
            {
                logitsPtr = logits.data();
                float lowestLogit = LowestLogit(logits);
                if (!unbanTokens && !inputs.unban_tokens_rt)
                {
                    //gpt2 uses negative logits, so we cant zero it
                    // set the logit of the eos token to minimum to avoid sampling it
                    if (file_format == FileFormat::GPT2_1 ||
                         file_format == FileFormat::GPT2_2 ||
                         file_format == FileFormat::GPT2_3 ||
                         file_format == FileFormat::GPT2_4 ||
                         file_format == FileFormat::GPTJ_1 ||
                         file_format == FileFormat::GPTJ_2 ||
                         file_format == FileFormat::GPTJ_3 ||
                         file_format == FileFormat::GPTJ_4 ||
                         file_format == FileFormat::GPTJ_5)
                    {
                        eosID = 50256;

                        if(logits.size() > eosID)
                        {
                            logits[eosID] = lowestLogit;
                        }
                        else
                        {
                            //special case, starcoder models use ID 0 for EOS
                            if (file_format == FileFormat::GPT2_3 || file_format == FileFormat::GPT2_4)
                            {
                                eosID = 0;
                                logits[eosID] = lowestLogit;

                            }
                        }
                    }

                     // set the logit of the eos token (0) to minimum to avoid sampling it
                    if (file_format == FileFormat::RWKV_1 ||
                        file_format == FileFormat::RWKV_2 ||
                        file_format == FileFormat::NEOX_1 ||
                         file_format == FileFormat::NEOX_2 ||
                         file_format == FileFormat::NEOX_3 ||
                         file_format == FileFormat::NEOX_4 ||
                         file_format == FileFormat::NEOX_5 ||
                         file_format == FileFormat::NEOX_6 ||
                         file_format == FileFormat::NEOX_7 ||
                         file_format == FileFormat::MPT_1)
                    {
                        eosID = 0;
                        logits[eosID] = lowestLogit;
                    }
                }

                if(btsize>0)
                {
                    for (int t = 0; t < btsize; ++t)
                    {
                        logits[banned_token_ids[t]] = lowestLogit;
                    }
                }
            }

            id = SampleLogits(logitsPtr, nctx, n_vocab, last_n_size, repeat_penalty,
            top_k, top_a, top_p, typical_p, tfs_z, temp, rng,
            params.mirostat, params.mirostat_tau, params.mirostat_eta, sampler_order);

            last_n_tokens.erase(last_n_tokens.begin());
            last_n_tokens.push_back(id);
            current_context_tokens.push_back(id);

            // add it to the context
            embd.push_back(id);

            // decrement remaining sampling budget
            --remaining_tokens;

            if (startedsampling) {
                rewind_state.set_logits_slot(llama_ctx_v4, current_context_tokens.size() - 1);
            }

            for (auto id : embd)
            {
                std::string tokenizedstr = FileFormatTokenizeID(id, file_format);
                if(stream_sse)
                {
                    generated_tokens.push_back(tokenizedstr);
                }
                concat_output_mtx.lock();
                concat_output += tokenizedstr;
                concat_output_mtx.unlock();
            }

            if (startedsampling && debugmode!=-1)
            {
                // printf("\rGenerating (%d / %d tokens)", (params.n_predict - remaining_tokens), params.n_predict);
            }
            if(debugmode==1 && top_picks.size()>0)
            {
                printf(" [");
                bool firstloop = true;
                for (auto & pick : top_picks)
                {
                    if (!firstloop)
                    {
                        printf(" ");
                    }
                    firstloop = false;
                    std::string tokenizedstr = FileFormatTokenizeID(pick.id, file_format);
                    ::utreplace(tokenizedstr, "\n", "\\n");
                    printf("(%s %.2f%%)", RemoveBell(tokenizedstr).c_str(), pick.p*100);
                }
                printf("]\n");
            }

            if((unbanTokens||inputs.unban_tokens_rt) && id==eosID)
            {
                stopper_unused_tokens = remaining_tokens;
                printf("\n(EOS token triggered!)");
                remaining_tokens = 0;
                last_stop_reason = stop_reason::EOS_TOKEN;
            }

            for (const auto &matched : stop_sequence)
            {
                if (concat_output.find(matched) != std::string::npos)
                {
                    stopper_unused_tokens = remaining_tokens;
                    remaining_tokens = 0;
                    if(debugmode!=-1)
                    {
                        printf("\n(Stop sequence triggered: <%s>)", matched.c_str());
                    }
                    last_stop_reason = stop_reason::CUSTOM_STOPPER;
                    break;
                }
            }
            fflush(stdout);
        }
        else
        {
            // some user input remains from prompt or interaction, forward it to processing
            while ((int)embd_inp.size() > input_consumed)
            {
                embd.push_back(embd_inp[input_consumed]);
                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(embd_inp[input_consumed]);
                current_context_tokens.push_back(embd_inp[input_consumed]);
                ++input_consumed;
                if ((int)embd.size() >= params.n_batch)
                {
                    break;
                }
            }
        }
    }
    time2 = timer_check();
    float pt1 = (time1*1000.0/(embd_inp.size()==0?1:embd_inp.size()));
    int realnpredict = params.n_predict-stopper_unused_tokens;
    float pt2 = (time2*1000.0/(realnpredict==0?1:realnpredict));
    float tokens_per_second = (realnpredict == 0 ? 0 : realnpredict / (time1 + time2));
    printf("\nTime Taken - Processing:%.1fs (%.0fms/T), Generation:%.1fs (%.0fms/T), Total:%.1fs (%.1fT/s)", time1, pt1, time2, pt2, (time1 + time2), tokens_per_second);
    fflush(stdout);
    output.status = 1;
    generation_finished = true;
    last_eval_time = pt2;
    last_process_time = pt1;
    last_token_count = realnpredict;
    snprintf(output.text, sizeof(output.text), "%s", concat_output.c_str());

    return output;
}
