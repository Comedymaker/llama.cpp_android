#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <random>
#include <set>
#include <string>
#include <vector>

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

struct seq_draft {
    bool active   = false;
    bool drafting = false;
    bool skip     = false;

    int i_batch_dft = 0;
    std::vector<int> i_batch_tgt;

    std::vector<llama_token> tokens;
    std::vector<std::vector<llama_token_data>> dists;

    double cum_conf = 1.0;
    int    depth    = 0;

    struct common_sampler * smpl = nullptr;
};

struct provisional_result {
    int branch_id = -1;
    int branch_tokens = 0;
    llama_tokens tokens;
};

static float tree_cumulative_conf(const std::vector<seq_draft> & drafts) {
    double best_conf = 0.0;

    for (const auto & draft : drafts) {
        if (!draft.active || !draft.drafting) {
            continue;
        }

        best_conf = std::max(best_conf, draft.cum_conf);
    }

    return best_conf;
}

static double branch_deficit(const seq_draft & draft, int total_budget) {
    return total_budget * draft.cum_conf - draft.depth;
}

static int find_best_branch(const std::vector<seq_draft> & drafts) {
    int   best_idx   = -1;
    double best_score = -1.0;

    for (int i = 0; i < (int) drafts.size(); ++i) {
        const auto & draft = drafts[i];
        if (!draft.active) {
            continue;
        }

        const double score = draft.cum_conf;
        if (score > best_score) {
            best_score = score;
            best_idx = i;
        }
    }

    return best_idx;
}

static int find_pacer_branch(const std::vector<seq_draft> & drafts, int total_budget) {
    int   best_idx   = -1;
    double best_score = -1e30;

    for (int i = 0; i < (int) drafts.size(); ++i) {
        const auto & draft = drafts[i];
        if (!draft.active || !draft.drafting) {
            continue;
        }

        const double score = branch_deficit(draft, total_budget);
        if (score > best_score) {
            best_score = score;
            best_idx = i;
        }
    }

    return best_idx;
}

static provisional_result provisional_greedy_draft(
        llama_context * ctx,
        common_sampler * smpl,
        const llama_tokens & prompt,
        int n_predict,
        int branch_id) {
    provisional_result result;
    result.branch_id = branch_id;

    if (ctx == nullptr || smpl == nullptr || prompt.empty() || n_predict <= 0) {
        return result;
    }

    auto * mem = llama_get_memory(ctx);
    llama_memory_clear(mem, false);
    common_sampler_reset(smpl);

    if (prompt.size() > 1) {
        llama_decode(ctx, llama_batch_get_one(const_cast<llama_token *>(prompt.data()), prompt.size() - 1));
    }

    llama_token id_last = prompt.back();
    int n_past = prompt.size() - 1;

    llama_batch batch = llama_batch_init(llama_n_batch(ctx), 0, 1);

    for (int i = 0; i < n_predict; ++i) {
        common_batch_clear(batch);
        common_batch_add(batch, id_last, n_past++, { 0 }, true);

        if (llama_decode(ctx, batch) != 0) {
            break;
        }

        const llama_token id = common_sampler_sample(smpl, ctx, 0);
        common_sampler_accept(smpl, id, true);

        result.tokens.push_back(id);
        id_last = id;
    }

    result.branch_tokens = result.tokens.size();
    llama_batch_free(batch);
    return result;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;

    // needed to get candidate probs even for temp <= 0.0
    params.sampling.n_probs = 128;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    if (params.n_predict < -1) {
        LOG_ERR("%s: --n-predict must be >= -1\n", __func__);
        return 1;
    }

    common_init();

    if (params.speculative.mparams_dft.path.empty()) {
        LOG_ERR("%s: --model-draft is required\n", __func__);
        return 1;
    }

    const int n_seq_dft = params.n_parallel;
    const bool greedy_sampling = params.sampling.temp <= 0.0f;
    const bool enable_provisional = false && greedy_sampling && std::getenv("LLAMA_EDGELLM_PROV_EXPERIMENTAL") != nullptr;

    std::default_random_engine rng(params.sampling.seed == LLAMA_DEFAULT_SEED ? std::random_device()() : params.sampling.seed);
    std::uniform_real_distribution<> u_dist;

    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model_tgt = nullptr;
    llama_model * model_dft = nullptr;

    llama_context * ctx_tgt = nullptr;
    llama_context * ctx_dft = nullptr;
    llama_context * ctx_dft_prov = nullptr;

    auto llama_init_tgt = common_init_from_params(params);

    model_tgt = llama_init_tgt->model();
    ctx_tgt   = llama_init_tgt->context();

    params.devices = params.speculative.devices;
    params.model = params.speculative.mparams_dft;
    params.n_gpu_layers = params.speculative.n_gpu_layers;
    if (params.speculative.cpuparams.n_threads > 0) {
        params.cpuparams.n_threads = params.speculative.cpuparams.n_threads;
    }

    params.cpuparams_batch.n_threads = params.speculative.cpuparams_batch.n_threads;
    params.tensor_buft_overrides     = params.speculative.tensor_buft_overrides;

    auto llama_init_dft = common_init_from_params(params);

    model_dft = llama_init_dft->model();
    ctx_dft   = llama_init_dft->context();

    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);

    const bool vocab_type_tgt = llama_vocab_type(vocab_tgt);
    const bool vocab_type_dft = llama_vocab_type(vocab_dft);

    if (vocab_type_tgt != vocab_type_dft) {
        LOG_ERR("%s: draft model vocab type must match target model to use speculation but ", __func__);
        LOG_ERR("vocab_type_dft = %d while vocab_type_tgt = %d\n", vocab_type_dft, vocab_type_tgt);
        return 1;
    }

    if (
        llama_vocab_get_add_bos(vocab_tgt) != llama_vocab_get_add_bos(vocab_dft) ||
        llama_vocab_get_add_eos(vocab_tgt) != llama_vocab_get_add_eos(vocab_dft) ||
        llama_vocab_bos(vocab_tgt) != llama_vocab_bos(vocab_dft) ||
        llama_vocab_eos(vocab_tgt) != llama_vocab_eos(vocab_dft)
    ) {
        LOG_ERR("%s: draft model special tokens must match target model to use speculation\n", __func__);
        return 1;
    }

    {
        const int n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt);
        const int n_vocab_dft = llama_vocab_n_tokens(vocab_dft);
        const int vocab_diff  = n_vocab_tgt > n_vocab_dft ? n_vocab_tgt - n_vocab_dft : n_vocab_dft - n_vocab_tgt;

        if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
            LOG_ERR("%s: draft model vocab must closely match target model to use speculation but ", __func__);
            LOG_ERR("target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
                    n_vocab_tgt, llama_vocab_n_tokens(vocab_dft), vocab_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
            return 1;
        }

        for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
            const char * token_text_tgt = llama_vocab_get_text(vocab_tgt, i);
            const char * token_text_dft = llama_vocab_get_text(vocab_dft, i);
            if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
                LOG_ERR("%s: draft model vocab must match target model to use speculation but ", __func__);
                LOG_ERR("token %d content differs - target '%s', draft '%s'\n", i,
                        common_token_to_piece(ctx_tgt, i).c_str(),
                        common_token_to_piece(ctx_dft, i).c_str());
                return 1;
            }
        }
    }

    common_sampler * smpl_prov = nullptr;
    if (enable_provisional) {
        auto params_prov = params;
        params_prov.n_parallel = 1;
        params_prov.n_batch = llama_n_batch(ctx_dft);
        auto cparams_prov = common_context_params_to_llama(params_prov);
        ctx_dft_prov = llama_init_from_model(model_dft, cparams_prov);
        if (ctx_dft_prov != nullptr) {
            smpl_prov = common_sampler_init(model_dft, params.sampling);
        }
    }

    auto * mem_tgt = llama_get_memory(ctx_tgt);
    auto * mem_dft = llama_get_memory(ctx_dft);

    std::vector<llama_token> inp = common_tokenize(ctx_tgt, params.prompt, true, true);

    const int max_context_size     = llama_n_ctx(ctx_tgt);
    const int max_tokens_list_size = max_context_size - 4;

    if ((int) inp.size() > max_tokens_list_size) {
        LOG_ERR("%s: prompt too long (%d tokens, max %d)\n", __func__, (int) inp.size(), max_tokens_list_size);
        return 1;
    }

    LOG("\n\n");
    for (auto id : inp) {
        LOG("%s", common_token_to_piece(ctx_tgt, id).c_str());
    }

    const int n_input = inp.size();
    llama_tokens accepted_tokens(inp.begin(), inp.end());

    const auto t_enc_start = ggml_time_us();

    llama_decode(ctx_tgt, llama_batch_get_one(inp.data(), n_input - 1));
    llama_decode(ctx_tgt, llama_batch_get_one(&inp.back(), 1));
    llama_decode(ctx_dft, llama_batch_get_one(inp.data(), n_input));

    const auto t_enc_end = ggml_time_us();

    int n_draft = params.speculative.n_max;

    int n_predict = 0;
    int n_drafted = 0;
    int n_accept  = 0;

    int n_past_tgt = inp.size();
    int n_past_dft = inp.size();

    bool has_eos = false;
    double alpha = 0.01;

    struct common_sampler * smpl = common_sampler_init(model_tgt, params.sampling);

    std::vector<seq_draft> drafts(n_seq_dft);
    for (int s = 0; s < n_seq_dft; ++s) {
        drafts[s].smpl = common_sampler_init(model_dft, params.sampling);
    }

    llama_batch batch_dft = llama_batch_init(llama_n_batch(ctx_dft), 0, 1);
    llama_batch batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, n_seq_dft);

    const auto t_dec_start = ggml_time_us();

    drafts[0].i_batch_tgt.resize(1);
    drafts[0].i_batch_tgt[0] = 0;

    while (true) {
        std::set<int> active_seqs = {};

        for (int s = 0; s < n_seq_dft; ++s) {
            if (!drafts[s].active) {
                continue;
            }

            active_seqs.insert(s);
            LOG_DBG("draft %d: %s\n", s, string_from(ctx_dft, drafts[s].tokens).c_str());
        }

        int i_dft  = 0;
        int s_keep = 0;

        llama_token token_id = LLAMA_TOKEN_NULL;
        std::string token_str;

        while (true) {
            bool accept = false;

            if (params.sampling.temp > 0) {
                common_sampler_sample(smpl, ctx_tgt, drafts[s_keep].i_batch_tgt[i_dft], true);
                auto & dist_tgt = *common_sampler_get_candidates(smpl, true);

                float p_tgt = 0.0f;
                float p_dft = 0.0f;

                while (!active_seqs.empty()) {
                    std::uniform_int_distribution<unsigned int> u_int_dist(0, active_seqs.size() - 1);
                    int s = *std::next(active_seqs.begin(), u_int_dist(rng));
                    if (i_dft >= (int) drafts[s].tokens.size()) {
                        drafts[s].active = false;
                        active_seqs.erase(s);
                        continue;
                    }

                    if (accept) {
                        if (drafts[s].tokens[i_dft] != drafts[s_keep].tokens[i_dft]) {
                            drafts[s].active = false;
                            active_seqs.erase(s);
                        }
                        continue;
                    }

                    float r = u_dist(rng);
                    llama_token_data_array dist_dft = {
                        drafts[s].dists[i_dft].data(),
                        drafts[s].dists[i_dft].size(),
                        LLAMA_TOKEN_NULL,
                        true
                    };

                    for (size_t i = 0; i < dist_tgt.size; ++i) {
                        if (dist_tgt.data[i].id == drafts[s].tokens[i_dft]) {
                            p_tgt = dist_tgt.data[i].p;
                            break;
                        }
                    }

                    for (size_t i = 0; i < dist_dft.size; ++i) {
                        if (dist_dft.data[i].id == drafts[s].tokens[i_dft]) {
                            p_dft = dist_dft.data[i].p;
                            break;
                        }
                    }

                    if (p_dft > 0.0f && r <= p_tgt / p_dft) {
                        s_keep = s;
                        accept = true;
                        token_id = drafts[s].tokens[i_dft];
                        token_str = common_token_to_piece(ctx_tgt, token_id);
                        common_sampler_accept(smpl, token_id, true);
                        break;
                    }

                    drafts[s].active = false;

                    GGML_ASSERT(dist_tgt.sorted);
                    GGML_ASSERT(dist_dft.sorted);

                    std::sort(dist_tgt.data, dist_tgt.data + dist_tgt.size, [](const llama_token_data & a, const llama_token_data & b) {
                        return a.id < b.id;
                    });
                    std::sort(dist_dft.data, dist_dft.data + dist_dft.size, [](const llama_token_data & a, const llama_token_data & b) {
                        return a.id < b.id;
                    });

                    float sum_probs = 0.0f;
                    for (size_t i = 0; i < dist_tgt.size; ++i) {
                        if (i < dist_dft.size) {
                            dist_tgt.data[i].p = std::max(0.0f, dist_tgt.data[i].p - dist_dft.data[i].p);
                        } else {
                            dist_tgt.data[i].p = std::max(0.0f, dist_tgt.data[i].p);
                        }
                        sum_probs += dist_tgt.data[i].p;
                    }

                    if (sum_probs > 0.0f) {
                        for (size_t i = 0; i < dist_tgt.size; ++i) {
                            dist_tgt.data[i].p /= sum_probs;
                        }
                    }

                    std::sort(dist_tgt.data, dist_tgt.data + dist_tgt.size, [](const llama_token_data & a, const llama_token_data & b) {
                        return a.p > b.p;
                    });

                    active_seqs.erase(s);
                    for (int i = 0; i < n_seq_dft; ++i) {
                        if (i == s) {
                            continue;
                        }
                        if (drafts[i].active && drafts[i].tokens[i_dft] == drafts[s].tokens[i_dft]) {
                            drafts[i].active = false;
                        }
                    }
                }

                if (!accept) {
                    auto & dist_tgt = *common_sampler_get_candidates(smpl, true);
                    std::vector<float> probs(dist_tgt.size);
                    for (size_t i = 0; i < dist_tgt.size; ++i) {
                        probs[i] = dist_tgt.data[i].p;
                    }

                    std::discrete_distribution<> dist(probs.begin(), probs.end());
                    const int idx = dist(rng);
                    token_id = dist_tgt.data[idx].id;
                    common_sampler_accept(smpl, token_id, true);
                    token_str = common_token_to_piece(ctx_tgt, token_id);
                }
            } else {
                token_id = common_sampler_sample(smpl, ctx_tgt, drafts[s_keep].i_batch_tgt[i_dft]);
                common_sampler_accept(smpl, token_id, true);
                token_str = common_token_to_piece(ctx_tgt, token_id);

                for (int s = 0; s < n_seq_dft; ++s) {
                    if (!drafts[s].active) {
                        continue;
                    }

                    if (i_dft < (int) drafts[s].tokens.size() && token_id == drafts[s].tokens[i_dft]) {
                        s_keep = s;
                        accept = true;
                    } else {
                        drafts[s].active = false;
                    }
                }
            }

            if (llama_vocab_is_eog(vocab_tgt, token_id)) {
                has_eos = true;
            }

            ++n_predict;
            accepted_tokens.push_back(token_id);

            if (accept) {
                ++n_accept;
                ++n_past_tgt;
                ++n_past_dft;
                ++i_dft;
                if (params.use_color) {
                    LOG("\u001b[%dm%s\u001b[37m", (36 - s_keep % 6), token_str.c_str());
                } else {
                    LOG("%s", token_str.c_str());
                }
                continue;
            }

            LOG("%s", token_str.c_str());
            break;
        }

        int accepted_round = i_dft;

        llama_memory_seq_keep(mem_dft, s_keep);
        llama_memory_seq_cp  (mem_dft, s_keep, 0, -1, -1);
        llama_memory_seq_keep(mem_dft, 0);

        llama_memory_seq_rm  (mem_tgt, s_keep, n_past_tgt, -1);
        llama_memory_seq_keep(mem_tgt, s_keep);
        llama_memory_seq_cp  (mem_tgt, s_keep, 0, -1, -1);
        llama_memory_seq_keep(mem_tgt, 0);

        for (int s = 0; s < n_seq_dft; ++s) {
            drafts[s].active = false;
            drafts[s].tokens.clear();
            drafts[s].i_batch_tgt.clear();
            drafts[s].dists.clear();
            drafts[s].cum_conf = 1.0;
            drafts[s].depth = 0;
        }

        drafts[0].tokens.push_back(token_id);
        drafts[0].dists.push_back(std::vector<llama_token_data>());
        drafts[0].i_batch_tgt.push_back(0);

        common_batch_clear(batch_dft);
        common_batch_add  (batch_dft, token_id, n_past_dft, { 0 }, true);

        llama_memory_seq_rm(mem_dft, 0, n_past_dft, -1);
        llama_decode(ctx_dft, batch_dft);
        ++n_past_dft;

        if ((params.n_predict >= 0 && n_predict > params.n_predict) || has_eos) {
            break;
        }

        if (drafts[0].smpl) {
            common_sampler_free(drafts[0].smpl);
        }
        drafts[0].smpl = common_sampler_clone(smpl);

        int n_seq_cur  = 1;
        int n_past_cur = n_past_dft;

        for (int s = 0; s < n_seq_dft; ++s) {
            drafts[s].active   = false;
            drafts[s].drafting = false;
            drafts[s].skip     = false;
        }

        drafts[0].active      = true;
        drafts[0].drafting    = true;
        drafts[0].i_batch_dft = 0;
        drafts[0].cum_conf    = 1.0;
        drafts[0].depth       = 0;

        common_batch_clear(batch_tgt);
        common_batch_add  (batch_tgt, drafts[0].tokens[0], n_past_tgt, { 0 }, true);

        int drafted_tree_tokens = 0;

        double tc_before_verify = 1.0;

        while (drafted_tree_tokens < n_draft) {
            batch_dft.n_tokens = 0;

            const int s = find_pacer_branch(drafts, n_draft);
            if (s < 0) {
                break;
            }

            common_sampler_sample(drafts[s].smpl, ctx_dft, drafts[s].i_batch_dft, true);
            const auto * cur_p = common_sampler_get_candidates(drafts[s].smpl, true);

            if (cur_p->size == 0) {
                break;
            }

            // Snapshot the parent branch so siblings branch from the same prefix.
            const auto parent_tokens = drafts[s].tokens;
            const auto parent_dists = drafts[s].dists;
            const auto parent_i_batch_tgt = drafts[s].i_batch_tgt;
            const int parent_i_batch_dft = drafts[s].i_batch_dft;
            const double parent_conf = drafts[s].cum_conf;
            const int parent_depth = drafts[s].depth;
            common_sampler * parent_smpl = common_sampler_clone(drafts[s].smpl);

            for (size_t k = 0; k < cur_p->size && drafted_tree_tokens < n_draft; ++k) {
                const float prob = std::max(cur_p->data[k].p, 1e-6f);
                const double child_conf = parent_conf * prob;
                const int child_depth = parent_depth + 1;
                const double child_deficit = n_draft * child_conf - child_depth;

                // Only keep branches that still deserve more budget, or the best token
                // on the first step to avoid an empty tree.
                if (k > 0 && child_deficit <= 0.0) {
                    break;
                }

                int branch_idx = -1;
                if (k == 0) {
                    branch_idx = s;
                    drafts[branch_idx].tokens = parent_tokens;
                    drafts[branch_idx].dists = parent_dists;
                    drafts[branch_idx].i_batch_tgt = parent_i_batch_tgt;
                    drafts[branch_idx].i_batch_dft = parent_i_batch_dft;
                } else {
                    if (n_seq_cur >= n_seq_dft) {
                        break;
                    }

                    branch_idx = n_seq_cur++;
                    llama_memory_seq_rm(mem_dft, branch_idx, -1, -1);
                    llama_memory_seq_cp(mem_dft, s, branch_idx, -1, -1);

                    for (int t = 0; t < batch_tgt.n_tokens; ++t) {
                        for (int p = 0; p < batch_tgt.n_seq_id[t]; ++p) {
                            if (batch_tgt.seq_id[t][p] == s) {
                                batch_tgt.seq_id[t][batch_tgt.n_seq_id[t]] = branch_idx;
                                batch_tgt.n_seq_id[t]++;
                                break;
                            }
                        }
                    }

                    drafts[branch_idx].active = true;
                    drafts[branch_idx].tokens = parent_tokens;
                    drafts[branch_idx].dists = parent_dists;
                    drafts[branch_idx].i_batch_tgt = parent_i_batch_tgt;
                    drafts[branch_idx].i_batch_dft = parent_i_batch_dft;
                    drafts[branch_idx].cum_conf = parent_conf;
                    drafts[branch_idx].depth = parent_depth;

                    if (drafts[branch_idx].smpl) {
                        common_sampler_free(drafts[branch_idx].smpl);
                    }
                    drafts[branch_idx].smpl = common_sampler_clone(parent_smpl);
                }

                const llama_token id = cur_p->data[k].id;
                common_sampler_accept(drafts[branch_idx].smpl, id, true);

                drafts[branch_idx].tokens.push_back(id);
                drafts[branch_idx].dists.push_back({ cur_p->data, cur_p->data + cur_p->size });
                drafts[branch_idx].i_batch_tgt.push_back(batch_tgt.n_tokens);
                drafts[branch_idx].cum_conf = child_conf;
                drafts[branch_idx].depth = child_depth;
                drafts[branch_idx].drafting = child_deficit > 0.0;

                common_batch_add(batch_tgt, id, n_past_tgt + drafted_tree_tokens + 1, { branch_idx }, true);
                drafts[branch_idx].i_batch_dft = batch_dft.n_tokens;
                common_batch_add(batch_dft, id, n_past_cur, { branch_idx }, true);

                ++drafted_tree_tokens;
            }

            common_sampler_free(parent_smpl);

            if (batch_dft.n_tokens == 0) {
                break;
            }

            llama_decode(ctx_dft, batch_dft);
            ++n_past_cur;
            n_drafted += batch_dft.n_tokens;

            if (drafted_tree_tokens >= params.speculative.n_min) {
                tc_before_verify = tree_cumulative_conf(drafts);
                if (tc_before_verify < alpha) {
                    LOG_DBG("adaptive fallback: Tc=%.6f alpha=%.6f\n", tc_before_verify, alpha);
                    break;
                }
            }
        }

        int best_branch_seq = find_best_branch(drafts);
        int best_branch_len = best_branch_seq >= 0 ? (int) drafts[best_branch_seq].tokens.size() - 1 : 0;

        std::future<provisional_result> provisional_future;
        bool provisional_pending = false;

        if (enable_provisional && smpl_prov != nullptr && ctx_dft_prov != nullptr && best_branch_seq >= 0 && best_branch_len > 0) {
            llama_tokens prompt_prov = accepted_tokens;
            prompt_prov.insert(prompt_prov.end(), drafts[best_branch_seq].tokens.begin() + 1, drafts[best_branch_seq].tokens.end());

            provisional_future = std::async(std::launch::async, [ctx_dft_prov, smpl_prov, prompt_prov, best_branch_seq, n_draft]() {
                return provisional_greedy_draft(ctx_dft_prov, smpl_prov, prompt_prov, std::max(1, n_draft / 2), best_branch_seq);
            });
            provisional_pending = true;
        }

        llama_memory_seq_keep(mem_tgt, 0);
        for (int s = 1; s < n_seq_dft; ++s) {
            llama_memory_seq_cp(mem_tgt, 0, s, -1, -1);
        }

        llama_decode(ctx_tgt, batch_tgt);
        ++n_past_tgt;

        for (int s = 0; s < n_seq_dft; ++s) {
            if (!drafts[s].active) {
                continue;
            }

            drafts[s].tokens.erase(drafts[s].tokens.begin());
            drafts[s].dists.erase(drafts[s].dists.begin());
        }

        if (provisional_pending) {
            (void) provisional_future.get();
        }

        const int n_all = best_branch_len;
        const int n_correct = std::min(accepted_round, n_all);
        if (n_all > 0) {
            if (n_correct == n_all) {
                alpha *= 0.5;
            } else {
                const double tc_safe = std::max(tc_before_verify, 1e-8);
                const double frac_err = (double) (n_all - n_correct) / (double) n_all;
                alpha = alpha / std::pow(tc_safe, frac_err);
            }
            alpha = std::clamp(alpha, 1e-6, 0.5);
        }
    }

    auto t_dec_end = ggml_time_us();

    LOG("\n\n");

    LOG_INF("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input,   (t_enc_end - t_enc_start) / 1e6f, inp.size() / ((t_enc_end - t_enc_start) / 1e6f));
    LOG_INF("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, (t_dec_end - t_dec_start) / 1e6f, n_predict / ((t_dec_end - t_dec_start) / 1e6f));

    LOG_INF("\n");
    LOG_INF("n_draft   = %d\n", n_draft);
    LOG_INF("n_predict = %d\n", n_predict);
    LOG_INF("n_drafted = %d\n", n_drafted);
    LOG_INF("n_accept  = %d\n", n_accept);
    LOG_INF("accept    = %.3f%%\n", n_drafted > 0 ? 100.0f * n_accept / n_drafted : 0.0f);
    LOG_INF("alpha     = %.6f\n", alpha);
    LOG_INF("provisional = %s\n", enable_provisional && ctx_dft_prov != nullptr ? "enabled" : "disabled");

    LOG_INF("\n");
    LOG_INF("draft:\n\n");
    llama_perf_context_print(ctx_dft);

    LOG_INF("\n");
    LOG_INF("target:\n\n");
    common_perf_print(ctx_tgt, smpl);

    common_sampler_free(smpl);
    common_sampler_free(smpl_prov);
    for (int s = 0; s < n_seq_dft; ++s) {
        common_sampler_free(drafts[s].smpl);
    }

    if (ctx_dft_prov != nullptr) {
        llama_free(ctx_dft_prov);
    }

    llama_batch_free(batch_dft);
    llama_batch_free(batch_tgt);

    llama_backend_free();

    LOG("\n\n");

    return 0;
}
