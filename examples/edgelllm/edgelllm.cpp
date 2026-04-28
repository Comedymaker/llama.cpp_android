#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <deque>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

struct spec_opts {
    bool        debug_spec         = false;
    bool        debug_spec_compare_official = false;
    std::string dump_spec_csv;
    int         match_rate_tokens  = 0;
    int         fixed_k            = -1;
    bool        enable_adaptive    = false;
    bool        enable_tree        = false;
    bool        enable_provisional = false;
    double      adaptive_alpha_init = 0.01;
    double      adaptive_success_decay = 0.5;
    double      adaptive_growth_scale = 1.0;
    double      adaptive_error_bump = 1.5;
    double      adaptive_alpha_min = 1e-6;
    double      adaptive_alpha_max = 0.5;
    int         adaptive_max_proposal = -1;
    std::string adaptive_policy = "current";
    int         adaptive_safety_window = 0;
    double      adaptive_safety_min_accept = 0.60;
    double      adaptive_safety_early_reject = 0.35;
    double      adaptive_safety_alpha_boost = 1.35;
    int         tree_width = 2;
    int         tree_depth = 2;
    int         tree_max_tokens = 4;
    double      tree_min_branch_prob = 0.10;
    double      tree_max_branch_gap = 1.0;
    bool        tree_reuse_prefix = false;
    bool        tree_cache_branch_state = false;
    int         tree_verify_top_n = 0;
    double      tree_verify_conf_gap = 1.0;
    int         tree_verify_min_len = 0;
    bool        debug_tree = false;
};

struct token_topk {
    llama_token id = LLAMA_TOKEN_NULL;
    float prob = 0.0f;
};

struct target_probe {
    llama_token sampled = LLAMA_TOKEN_NULL;
    std::vector<llama_token_data> dist;
    std::vector<token_topk> top5;
    bool draft_hits_argmax = false;
};

struct draft_round {
    llama_tokens tokens;
    std::vector<std::vector<llama_token_data>> dists;
    double cum_conf = 1.0;
    bool fallback_triggered = false;
    double ms = 0.0;
};

struct verify_round {
    llama_tokens emitted;
    int accepted_len = 0;
    int reject_pos = -1;
    bool all_accepted = false;
    double ms = 0.0;
};

struct stats_total {
    int rounds = 0;
    int n_predict = 0;
    int n_drafted = 0;
    int n_accept = 0;
    int n_rollbacks = 0;
    double draft_ms = 0.0;
    double verify_ms = 0.0;
    double total_ms = 0.0;
    int n_verify = 0;
    int n_branch_total = 0;
    int n_tree_token_total = 0;
    int n_tree_verify_branch_total = 0;
    int n_tree_verify_token_total = 0;
    int n_tree_reuse_hits = 0;
    int n_tree_avoided_replays = 0;
    double tree_replay_ms = 0.0;
    double tree_expand_ms = 0.0;
    double tree_flatten_ms = 0.0;
    double tree_verify_ms = 0.0;
    double tree_select_ms = 0.0;
    double tree_sync_ms = 0.0;
};

struct adaptive_state {
    double alpha = 0.01;
    double alpha_min = 1e-6;
    double alpha_max = 0.5;
    std::deque<double> recent_accept_rates;
    std::deque<double> recent_reject_progress;
};

struct tree_node {
    int index = -1;
    int parent = -1;
    int depth = 0;
    int branch_id = 0;
    llama_token id = LLAMA_TOKEN_NULL;
    std::string text;
    double local_conf = 0.0;
    double cum_conf = 1.0;
};

struct tree_branch {
    int branch_id = 0;
    std::vector<int> node_indices;
    llama_tokens tokens;
    double cum_conf = 1.0;
};

struct tree_draft_round {
    std::vector<tree_node> nodes;
    std::vector<tree_branch> branches;
    int total_tokens = 0;
    double ms = 0.0;
    double replay_ms = 0.0;
    double expand_ms = 0.0;
    int reused_prefix_states = 0;
    int avoided_replays = 0;
};

struct tree_branch_verify {
    int branch_id = 0;
    int accepted_len = 0;
    int reject_pos = -1;
    double cum_conf = 1.0;
    llama_tokens emitted;
};

struct tree_verify_round {
    std::vector<tree_branch_verify> branches;
    double flatten_ms = 0.0;
    double verify_ms = 0.0;
    int pruned_count = 0;
    int verify_token_count = 0;
};

static std::string truncate_str(const std::string & s, size_t n) {
    if (s.size() <= n) {
        return s;
    }
    return s.substr(0, n) + "...";
}

static std::string token_str(llama_context * ctx, llama_token id) {
    return common_token_to_piece(ctx, id);
}

static std::string tail_text(llama_context * ctx, const llama_tokens & tokens, size_t tail_n) {
    std::string out;
    const size_t start = tokens.size() > tail_n ? tokens.size() - tail_n : 0;
    for (size_t i = start; i < tokens.size(); ++i) {
        out += common_token_to_piece(ctx, tokens[i]);
    }
    return out;
}

static std::string meta_val(const llama_model * model, const char * key) {
    char buf[4096];
    const int n = llama_model_meta_val_str(model, key, buf, sizeof(buf));
    if (n <= 0) {
        return "";
    }
    return std::string(buf, n);
}

static void seed_sampler(common_sampler * smpl, const llama_tokens & tokens) {
    for (llama_token tok : tokens) {
        common_sampler_accept(smpl, tok, true);
    }
}

static bool parse_spec_opts(int argc, char ** argv, spec_opts & opts, std::vector<char *> & filtered) {
    filtered.clear();
    filtered.push_back(argv[0]);

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        auto require_value = [&](const char * name) -> char * {
            if (i + 1 >= argc) {
                LOG_ERR("%s: missing value for %s\n", __func__, name);
                return nullptr;
            }
            return argv[++i];
        };

        if (arg == "--debug-spec") {
            opts.debug_spec = true;
            continue;
        }
        if (arg == "--debug-spec-compare-official") {
            opts.debug_spec_compare_official = true;
            continue;
        }
        if (arg == "--dump-spec-csv") {
            char * val = require_value("--dump-spec-csv");
            if (val == nullptr) {
                return false;
            }
            opts.dump_spec_csv = val;
            continue;
        }
        if (arg == "--spec-match-rate") {
            char * val = require_value("--spec-match-rate");
            if (val == nullptr) {
                return false;
            }
            opts.match_rate_tokens = std::atoi(val);
            continue;
        }
        if (arg == "--spec-fixed-k") {
            char * val = require_value("--spec-fixed-k");
            if (val == nullptr) {
                return false;
            }
            opts.fixed_k = std::atoi(val);
            continue;
        }
        if (arg == "--spec-adaptive-fallback") {
            opts.enable_adaptive = true;
            continue;
        }
        if (arg == "--enable-adaptive-fallback") {
            opts.enable_adaptive = true;
            continue;
        }
        if (arg == "--spec-tree") {
            opts.enable_tree = true;
            continue;
        }
        if (arg == "--enable-tree") {
            opts.enable_tree = true;
            continue;
        }
        if (arg == "--spec-provisional") {
            opts.enable_provisional = true;
            continue;
        }
        if (arg == "--tree-width") {
            char * val = require_value("--tree-width");
            if (val == nullptr) {
                return false;
            }
            opts.tree_width = std::atoi(val);
            continue;
        }
        if (arg == "--tree-depth") {
            char * val = require_value("--tree-depth");
            if (val == nullptr) {
                return false;
            }
            opts.tree_depth = std::atoi(val);
            continue;
        }
        if (arg == "--tree-max-tokens") {
            char * val = require_value("--tree-max-tokens");
            if (val == nullptr) {
                return false;
            }
            opts.tree_max_tokens = std::atoi(val);
            continue;
        }
        if (arg == "--tree-min-branch-prob") {
            char * val = require_value("--tree-min-branch-prob");
            if (val == nullptr) {
                return false;
            }
            opts.tree_min_branch_prob = std::atof(val);
            continue;
        }
        if (arg == "--tree-max-branch-gap") {
            char * val = require_value("--tree-max-branch-gap");
            if (val == nullptr) {
                return false;
            }
            opts.tree_max_branch_gap = std::atof(val);
            continue;
        }
        if (arg == "--tree-reuse-prefix") {
            opts.tree_reuse_prefix = true;
            continue;
        }
        if (arg == "--tree-cache-branch-state") {
            opts.tree_cache_branch_state = true;
            continue;
        }
        if (arg == "--tree-verify-top-n") {
            char * val = require_value("--tree-verify-top-n");
            if (val == nullptr) {
                return false;
            }
            opts.tree_verify_top_n = std::atoi(val);
            continue;
        }
        if (arg == "--tree-verify-conf-gap") {
            char * val = require_value("--tree-verify-conf-gap");
            if (val == nullptr) {
                return false;
            }
            opts.tree_verify_conf_gap = std::atof(val);
            continue;
        }
        if (arg == "--tree-verify-min-len") {
            char * val = require_value("--tree-verify-min-len");
            if (val == nullptr) {
                return false;
            }
            opts.tree_verify_min_len = std::atoi(val);
            continue;
        }
        if (arg == "--debug-tree") {
            opts.debug_tree = true;
            continue;
        }
        if (arg == "--adaptive-alpha-init") {
            char * val = require_value("--adaptive-alpha-init");
            if (val == nullptr) {
                return false;
            }
            opts.adaptive_alpha_init = std::atof(val);
            continue;
        }
        if (arg == "--adaptive-success-decay") {
            char * val = require_value("--adaptive-success-decay");
            if (val == nullptr) {
                return false;
            }
            opts.adaptive_success_decay = std::atof(val);
            continue;
        }
        if (arg == "--adaptive-growth-scale") {
            char * val = require_value("--adaptive-growth-scale");
            if (val == nullptr) {
                return false;
            }
            opts.adaptive_growth_scale = std::atof(val);
            continue;
        }
        if (arg == "--adaptive-error-bump") {
            char * val = require_value("--adaptive-error-bump");
            if (val == nullptr) {
                return false;
            }
            opts.adaptive_error_bump = std::atof(val);
            continue;
        }
        if (arg == "--adaptive-alpha-min") {
            char * val = require_value("--adaptive-alpha-min");
            if (val == nullptr) {
                return false;
            }
            opts.adaptive_alpha_min = std::atof(val);
            continue;
        }
        if (arg == "--adaptive-alpha-max") {
            char * val = require_value("--adaptive-alpha-max");
            if (val == nullptr) {
                return false;
            }
            opts.adaptive_alpha_max = std::atof(val);
            continue;
        }
        if (arg == "--adaptive-policy") {
            char * val = require_value("--adaptive-policy");
            if (val == nullptr) {
                return false;
            }
            opts.adaptive_policy = val;
            continue;
        }
        if (arg == "--adaptive-max-proposal") {
            char * val = require_value("--adaptive-max-proposal");
            if (val == nullptr) {
                return false;
            }
            opts.adaptive_max_proposal = std::atoi(val);
            continue;
        }
        if (arg == "--adaptive-safety-window") {
            char * val = require_value("--adaptive-safety-window");
            if (val == nullptr) {
                return false;
            }
            opts.adaptive_safety_window = std::atoi(val);
            continue;
        }
        if (arg == "--adaptive-safety-min-accept") {
            char * val = require_value("--adaptive-safety-min-accept");
            if (val == nullptr) {
                return false;
            }
            opts.adaptive_safety_min_accept = std::atof(val);
            continue;
        }
        if (arg == "--adaptive-safety-early-reject") {
            char * val = require_value("--adaptive-safety-early-reject");
            if (val == nullptr) {
                return false;
            }
            opts.adaptive_safety_early_reject = std::atof(val);
            continue;
        }
        if (arg == "--adaptive-safety-alpha-boost") {
            char * val = require_value("--adaptive-safety-alpha-boost");
            if (val == nullptr) {
                return false;
            }
            opts.adaptive_safety_alpha_boost = std::atof(val);
            continue;
        }

        filtered.push_back(argv[i]);
    }

    return true;
}

static bool check_vocab_compat(llama_context * ctx_tgt, llama_context * ctx_dft, llama_model * model_tgt, llama_model * model_dft) {
    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);

    const int vocab_type_tgt = llama_vocab_type(vocab_tgt);
    const int vocab_type_dft = llama_vocab_type(vocab_dft);

    LOG_INF("spec-check: vocab_type target=%d draft=%d\n", vocab_type_tgt, vocab_type_dft);
    LOG_INF("spec-check: add_bos target=%d draft=%d\n", llama_vocab_get_add_bos(vocab_tgt), llama_vocab_get_add_bos(vocab_dft));
    LOG_INF("spec-check: add_eos target=%d draft=%d\n", llama_vocab_get_add_eos(vocab_tgt), llama_vocab_get_add_eos(vocab_dft));
    LOG_INF("spec-check: bos target=%d draft=%d\n", llama_vocab_bos(vocab_tgt), llama_vocab_bos(vocab_dft));
    LOG_INF("spec-check: eos target=%d draft=%d\n", llama_vocab_eos(vocab_tgt), llama_vocab_eos(vocab_dft));
    LOG_INF("spec-check: eot target=%d draft=%d\n", llama_vocab_eot(vocab_tgt), llama_vocab_eot(vocab_dft));
    LOG_INF("spec-check: chat_template target='%s'\n", truncate_str(llama_model_chat_template(model_tgt, nullptr) ? llama_model_chat_template(model_tgt, nullptr) : "", 120).c_str());
    LOG_INF("spec-check: chat_template draft ='%s'\n", truncate_str(llama_model_chat_template(model_dft, nullptr) ? llama_model_chat_template(model_dft, nullptr) : "", 120).c_str());

    if (vocab_type_tgt != vocab_type_dft) {
        LOG_ERR("%s: draft model vocab type must match target model\n", __func__);
        return false;
    }

    if (
        llama_vocab_get_add_bos(vocab_tgt) != llama_vocab_get_add_bos(vocab_dft) ||
        llama_vocab_get_add_eos(vocab_tgt) != llama_vocab_get_add_eos(vocab_dft) ||
        llama_vocab_bos(vocab_tgt) != llama_vocab_bos(vocab_dft) ||
        llama_vocab_eos(vocab_tgt) != llama_vocab_eos(vocab_dft)
    ) {
        LOG_ERR("%s: draft model special tokens must match target model\n", __func__);
        return false;
    }

    const int n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt);
    const int n_vocab_dft = llama_vocab_n_tokens(vocab_dft);
    const int vocab_diff  = std::abs(n_vocab_tgt - n_vocab_dft);

    if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
        LOG_ERR("%s: target vocab size %d and draft vocab size %d differ too much\n", __func__, n_vocab_tgt, n_vocab_dft);
        return false;
    }

    for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
        const char * token_text_tgt = llama_vocab_get_text(vocab_tgt, i);
        const char * token_text_dft = llama_vocab_get_text(vocab_dft, i);
        if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
            LOG_ERR("%s: token %d differs: target '%s', draft '%s'\n", __func__, i,
                    token_str(ctx_tgt, i).c_str(),
                    ctx_dft ? token_str(ctx_dft, i).c_str() : token_text_dft);
            return false;
        }
    }

    return true;
}

static void print_sampling_config(const common_params_sampling & s) {
    LOG_INF("spec-check: sampler_seq=%s\n", common_sampler_types_from_chars("").empty() ? "" : s.print().c_str());
    LOG_INF("spec-check: temp=%.4f top_k=%d top_p=%.4f min_p=%.4f repeat_penalty=%.4f penalty_last_n=%d\n",
            s.temp, s.top_k, s.top_p, s.min_p, s.penalty_repeat, s.penalty_last_n);
    LOG_INF("spec-check: freq_penalty=%.4f present_penalty=%.4f mirostat=%d greedy=%s\n",
            s.penalty_freq, s.penalty_present, s.mirostat, s.temp <= 0.0f ? "true" : "false");
}

static std::vector<llama_token_data> copy_dist(const llama_token_data_array * arr) {
    return std::vector<llama_token_data>(arr->data, arr->data + arr->size);
}

static float find_prob(const std::vector<llama_token_data> & dist, llama_token tok) {
    for (const auto & item : dist) {
        if (item.id == tok) {
            return item.p;
        }
    }
    return 0.0f;
}

static std::vector<token_topk> topk_from_dist(std::vector<llama_token_data> dist, int k) {
    std::sort(dist.begin(), dist.end(), [](const llama_token_data & a, const llama_token_data & b) {
        return a.p > b.p;
    });

    std::vector<token_topk> out;
    for (int i = 0; i < std::min<int>(k, dist.size()); ++i) {
        out.push_back({ dist[i].id, dist[i].p });
    }
    return out;
}

static target_probe probe_target(common_sampler * smpl_state, llama_context * ctx, int idx, llama_token draft_tok) {
    target_probe out;

    common_sampler * probe = common_sampler_clone(smpl_state);
    out.sampled = common_sampler_sample(probe, ctx, idx, true);

    auto * cur = common_sampler_get_candidates(probe, true);
    out.dist = copy_dist(cur);
    out.top5 = topk_from_dist(out.dist, 5);
    out.draft_hits_argmax = !out.top5.empty() && out.top5[0].id == draft_tok;

    common_sampler_free(probe);
    return out;
}

static llama_token sample_residual(
        const std::vector<llama_token_data> & dist_tgt,
        const std::vector<llama_token_data> & dist_dft,
        std::default_random_engine & rng) {
    std::vector<llama_token_data> merged = dist_tgt;
    std::sort(merged.begin(), merged.end(), [](const llama_token_data & a, const llama_token_data & b) {
        return a.id < b.id;
    });

    std::vector<llama_token_data> dft = dist_dft;
    std::sort(dft.begin(), dft.end(), [](const llama_token_data & a, const llama_token_data & b) {
        return a.id < b.id;
    });

    size_t j = 0;
    float sum = 0.0f;

    for (auto & item : merged) {
        while (j < dft.size() && dft[j].id < item.id) {
            ++j;
        }
        const float p_dft = (j < dft.size() && dft[j].id == item.id) ? dft[j].p : 0.0f;
        item.p = std::max(0.0f, item.p - p_dft);
        sum += item.p;
    }

    if (sum <= 0.0f) {
        auto best = std::max_element(dist_tgt.begin(), dist_tgt.end(), [](const llama_token_data & a, const llama_token_data & b) {
            return a.p < b.p;
        });
        return best != dist_tgt.end() ? best->id : LLAMA_TOKEN_NULL;
    }

    std::vector<double> probs;
    probs.reserve(merged.size());
    for (const auto & item : merged) {
        probs.push_back(item.p / sum);
    }

    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return merged[dist(rng)].id;
}

static bool decode_one(llama_context * ctx, llama_token tok, int pos, llama_batch & batch) {
    common_batch_clear(batch);
    common_batch_add(batch, tok, pos, { 0 }, true);
    return llama_decode(ctx, batch) == 0;
}

static double token_confidence_from_logits(llama_context * ctx, llama_token tok) {
    const float * logits = llama_get_logits_ith(ctx, 0);
    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
    const int n_vocab = llama_vocab_n_tokens(vocab);

    float max_logit = -INFINITY;
    for (int i = 0; i < n_vocab; ++i) {
        max_logit = std::max(max_logit, logits[i]);
    }

    double denom = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        denom += std::exp(double(logits[i] - max_logit));
    }

    return std::exp(double(logits[tok] - max_logit)) / std::max(denom, 1e-12);
}

static std::vector<token_topk> topk_from_logits(llama_context * ctx, int k) {
    const float * logits = llama_get_logits_ith(ctx, 0);
    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
    const int n_vocab = llama_vocab_n_tokens(vocab);

    float max_logit = -INFINITY;
    for (int i = 0; i < n_vocab; ++i) {
        max_logit = std::max(max_logit, logits[i]);
    }

    double denom = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        denom += std::exp(double(logits[i] - max_logit));
    }

    std::vector<token_topk> all;
    all.reserve(n_vocab);
    for (int i = 0; i < n_vocab; ++i) {
        all.push_back({ i, float(std::exp(double(logits[i] - max_logit)) / std::max(denom, 1e-12)) });
    }

    std::partial_sort(all.begin(), all.begin() + std::min<int>(k, all.size()), all.end(), [](const token_topk & a, const token_topk & b) {
        return a.prob > b.prob;
    });
    all.resize(std::min<int>(k, all.size()));
    return all;
}

static std::vector<token_topk> topk_from_logits_ith(llama_context * ctx, int ith, int k) {
    const float * logits = llama_get_logits_ith(ctx, ith);
    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
    const int n_vocab = llama_vocab_n_tokens(vocab);

    float max_logit = -INFINITY;
    for (int i = 0; i < n_vocab; ++i) {
        max_logit = std::max(max_logit, logits[i]);
    }

    double denom = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        denom += std::exp(double(logits[i] - max_logit));
    }

    std::vector<token_topk> all;
    all.reserve(n_vocab);
    for (int i = 0; i < n_vocab; ++i) {
        all.push_back({ i, float(std::exp(double(logits[i] - max_logit)) / std::max(denom, 1e-12)) });
    }

    std::partial_sort(all.begin(), all.begin() + std::min<int>(k, all.size()), all.end(), [](const token_topk & a, const token_topk & b) {
        return a.prob > b.prob;
    });
    all.resize(std::min<int>(k, all.size()));
    return all;
}

static void sampler_accept_all(common_sampler * smpl, const llama_tokens & tokens) {
    if (smpl == nullptr) {
        return;
    }
    for (llama_token tok : tokens) {
        common_sampler_accept(smpl, tok, true);
    }
}

static void sync_draft_after_verify(
        llama_context * ctx_dft,
        common_sampler * smpl_dft,
        const llama_tokens & emitted,
        int prompt_len,
        int accepted_len,
        llama_batch & batch_dft) {
    if (ctx_dft == nullptr || emitted.empty()) {
        return;
    }

    const int rollback_pos = prompt_len + accepted_len;
    llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, rollback_pos, -1);

    const llama_token corrected = emitted.back();
    decode_one(ctx_dft, corrected, rollback_pos, batch_dft);

    sampler_accept_all(smpl_dft, emitted);
}

static bool prime_context(llama_context * ctx, const llama_tokens & prompt) {
    if (prompt.empty()) {
        return false;
    }
    if (prompt.size() > 1) {
        if (llama_decode(ctx, llama_batch_get_one(const_cast<llama_token *>(prompt.data()), prompt.size() - 1)) != 0) {
            return false;
        }
    }
    llama_token last = prompt.back();
    return llama_decode(ctx, llama_batch_get_one(&last, 1)) == 0;
}

static bool prime_context_exact(llama_context * ctx, const llama_tokens & prompt) {
    auto * mem = llama_get_memory(ctx);
    llama_memory_clear(mem, false);
    return prime_context(ctx, prompt);
}

static tree_draft_round draft_minimal_tree(
        llama_context * ctx_scratch,
        const spec_opts & opts,
        const llama_tokens & prompt_full) {
    tree_draft_round out;
    const int64_t t0 = ggml_time_us();

    out.branches.push_back(tree_branch{0, {}, {}, 1.0});
    int next_branch_id = 1;

    if (!(opts.tree_reuse_prefix || opts.tree_cache_branch_state)) {
        for (int depth = 0; depth < std::max(1, opts.tree_depth); ++depth) {
            std::vector<tree_branch> frontier = out.branches;
            std::vector<tree_branch> next_branches;
            bool expanded_any = false;

            for (const auto & branch : frontier) {
                if (out.total_tokens >= opts.tree_max_tokens) {
                    next_branches.push_back(branch);
                    continue;
                }

                llama_tokens seq = prompt_full;
                seq.insert(seq.end(), branch.tokens.begin(), branch.tokens.end());
                const int64_t t_replay = ggml_time_us();
                if (!prime_context_exact(ctx_scratch, seq)) {
                    out.replay_ms += 1e-3 * (ggml_time_us() - t_replay);
                    next_branches.push_back(branch);
                    continue;
                }
                out.replay_ms += 1e-3 * (ggml_time_us() - t_replay);

                const int64_t t_expand = ggml_time_us();
                auto top = topk_from_logits(ctx_scratch, std::max(2, opts.tree_width));
                if (top.empty()) {
                    out.expand_ms += 1e-3 * (ggml_time_us() - t_expand);
                    next_branches.push_back(branch);
                    continue;
                }

                const int parent = branch.node_indices.empty() ? -1 : branch.node_indices.back();

                tree_branch main_branch = branch;
                {
                    const auto & cand = top[0];
                    tree_node node;
                    node.index = out.nodes.size();
                    node.parent = parent;
                    node.depth = depth + 1;
                    node.branch_id = branch.branch_id;
                    node.id = cand.id;
                    node.text = token_str(ctx_scratch, cand.id);
                    node.local_conf = cand.prob;
                    node.cum_conf = branch.cum_conf * cand.prob;
                    out.nodes.push_back(node);
                    main_branch.node_indices.push_back(node.index);
                    main_branch.tokens.push_back(cand.id);
                    main_branch.cum_conf = node.cum_conf;
                    out.total_tokens++;
                    expanded_any = true;
                }
                next_branches.push_back(main_branch);

                if ((int) next_branches.size() < std::max(1, opts.tree_width) &&
                    out.total_tokens < opts.tree_max_tokens &&
                    (int) top.size() >= 2 &&
                    top[1].prob >= opts.tree_min_branch_prob &&
                    (top[0].prob - top[1].prob) <= opts.tree_max_branch_gap) {
                    const auto & cand = top[1];
                    tree_branch alt_branch = branch;
                    alt_branch.branch_id = next_branch_id++;

                    tree_node node;
                    node.index = out.nodes.size();
                    node.parent = parent;
                    node.depth = depth + 1;
                    node.branch_id = alt_branch.branch_id;
                    node.id = cand.id;
                    node.text = token_str(ctx_scratch, cand.id);
                    node.local_conf = cand.prob;
                    node.cum_conf = branch.cum_conf * cand.prob;
                    out.nodes.push_back(node);
                    alt_branch.node_indices.push_back(node.index);
                    alt_branch.tokens.push_back(cand.id);
                    alt_branch.cum_conf = node.cum_conf;
                    out.total_tokens++;
                    expanded_any = true;
                    next_branches.push_back(alt_branch);
                }
                out.expand_ms += 1e-3 * (ggml_time_us() - t_expand);
            }

            out.branches = next_branches;
            if (!expanded_any || out.total_tokens >= opts.tree_max_tokens) {
                break;
            }
        }
    } else {
        const int max_frontier = std::max(1, opts.tree_max_tokens + opts.tree_width + 2);
        llama_batch batch_scratch = llama_batch_init(1, 0, std::max(4, 2 * max_frontier));

        const int64_t t_replay = ggml_time_us();
        if (!prime_context_exact(ctx_scratch, prompt_full)) {
            llama_batch_free(batch_scratch);
            out.ms = 1e-3 * (ggml_time_us() - t0);
            return out;
        }
        out.replay_ms += 1e-3 * (ggml_time_us() - t_replay);

        std::vector<tree_branch> frontier = out.branches;
        std::vector<std::pair<llama_tokens, int>> cached_states;
        cached_states.push_back({ {}, 0 });
        int next_cache_seq = 2;

        auto find_cached_seq = [&](const llama_tokens & prefix) -> int {
            for (const auto & item : cached_states) {
                if (item.first == prefix) {
                    return item.second;
                }
            }
            return -1;
        };

        for (int depth = 0; depth < std::max(1, opts.tree_depth); ++depth) {
            std::vector<tree_branch> next_branches;
            bool expanded_any = false;

            for (const auto & branch : frontier) {
                if (out.total_tokens >= opts.tree_max_tokens) {
                    next_branches.push_back(branch);
                    continue;
                }

                llama_tokens replay_suffix = branch.tokens;
                int base_seq = 0;
                if (opts.tree_cache_branch_state && branch.tokens.size() > 1) {
                    llama_tokens prefix = branch.tokens;
                    prefix.pop_back();
                    const int cached_seq = find_cached_seq(prefix);
                    if (cached_seq >= 0) {
                        base_seq = cached_seq;
                        replay_suffix = { branch.tokens.back() };
                        out.reused_prefix_states++;
                    }
                }

                const int64_t t_branch_replay = ggml_time_us();
                auto * mem = llama_get_memory(ctx_scratch);
                const int work_seq = 1;
                llama_memory_seq_rm(mem, work_seq, -1, -1);
                llama_memory_seq_cp(mem, base_seq, work_seq, -1, -1);
                for (size_t i = 0; i < replay_suffix.size(); ++i) {
                    const int pos = prompt_full.size() + branch.tokens.size() - replay_suffix.size() + i;
                    if (!decode_one(ctx_scratch, replay_suffix[i], pos, batch_scratch)) {
                        replay_suffix.clear();
                        break;
                    }
                }
                out.replay_ms += 1e-3 * (ggml_time_us() - t_branch_replay);
                if (branch.tokens.empty()) {
                    out.avoided_replays++;
                } else if ((opts.tree_reuse_prefix || opts.tree_cache_branch_state) && replay_suffix.size() < branch.tokens.size()) {
                    out.avoided_replays++;
                } else if (opts.tree_reuse_prefix) {
                    out.avoided_replays++;
                }

                const int64_t t_expand = ggml_time_us();
                auto top = branch.tokens.empty() ? topk_from_logits(ctx_scratch, std::max(2, opts.tree_width))
                                                 : topk_from_logits(ctx_scratch, std::max(2, opts.tree_width));
                if (top.empty()) {
                    out.expand_ms += 1e-3 * (ggml_time_us() - t_expand);
                    next_branches.push_back(branch);
                    continue;
                }

                if (opts.tree_cache_branch_state && !branch.tokens.empty() && find_cached_seq(branch.tokens) < 0 && next_cache_seq < 2 * max_frontier) {
                    llama_memory_seq_rm(mem, next_cache_seq, -1, -1);
                    llama_memory_seq_cp(mem, work_seq, next_cache_seq, -1, -1);
                    cached_states.push_back({ branch.tokens, next_cache_seq++ });
                }

                const int parent = branch.node_indices.empty() ? -1 : branch.node_indices.back();
                tree_branch main_branch = branch;
                {
                    const auto & cand = top[0];
                    tree_node node;
                    node.index = out.nodes.size();
                    node.parent = parent;
                    node.depth = depth + 1;
                    node.branch_id = branch.branch_id;
                    node.id = cand.id;
                    node.text = token_str(ctx_scratch, cand.id);
                    node.local_conf = cand.prob;
                    node.cum_conf = branch.cum_conf * cand.prob;
                    out.nodes.push_back(node);
                    main_branch.node_indices.push_back(node.index);
                    main_branch.tokens.push_back(cand.id);
                    main_branch.cum_conf = node.cum_conf;
                    out.total_tokens++;
                    expanded_any = true;
                }
                next_branches.push_back(main_branch);

                if ((int) next_branches.size() < std::max(1, opts.tree_width) &&
                    out.total_tokens < opts.tree_max_tokens &&
                    (int) top.size() >= 2 &&
                    top[1].prob >= opts.tree_min_branch_prob &&
                    (top[0].prob - top[1].prob) <= opts.tree_max_branch_gap) {
                    const auto & cand = top[1];
                    tree_branch alt_branch = branch;
                    alt_branch.branch_id = next_branch_id++;

                    tree_node node;
                    node.index = out.nodes.size();
                    node.parent = parent;
                    node.depth = depth + 1;
                    node.branch_id = alt_branch.branch_id;
                    node.id = cand.id;
                    node.text = token_str(ctx_scratch, cand.id);
                    node.local_conf = cand.prob;
                    node.cum_conf = branch.cum_conf * cand.prob;
                    out.nodes.push_back(node);
                    alt_branch.node_indices.push_back(node.index);
                    alt_branch.tokens.push_back(cand.id);
                    alt_branch.cum_conf = node.cum_conf;
                    out.total_tokens++;
                    expanded_any = true;
                    next_branches.push_back(alt_branch);
                }
                out.expand_ms += 1e-3 * (ggml_time_us() - t_expand);
            }

            out.branches = next_branches;
            frontier = next_branches;
            if (!expanded_any || out.total_tokens >= opts.tree_max_tokens) {
                break;
            }
        }

        llama_batch_free(batch_scratch);
    }

    out.ms = 1e-3 * (ggml_time_us() - t0);
    return out;
}

static tree_verify_round verify_tree_branches(
        llama_context * ctx_tgt,
        common_sampler * smpl_tgt,
        const std::vector<tree_branch> & branches,
        llama_token id_last,
        int n_past,
        llama_batch & batch_tgt) {
    tree_verify_round out;
    if (branches.empty()) {
        return out;
    }

    auto * mem_tgt = llama_get_memory(ctx_tgt);
    for (size_t s = 1; s < branches.size(); ++s) {
        llama_memory_seq_cp(mem_tgt, 0, s, -1, -1);
    }

    const int64_t t_flatten = ggml_time_us();
    common_batch_clear(batch_tgt);
    std::vector<std::vector<int>> idxs(branches.size());

    for (size_t s = 0; s < branches.size(); ++s) {
        common_batch_add(batch_tgt, id_last, n_past, { int32_t(s) }, true);
        idxs[s].push_back(batch_tgt.n_tokens - 1);
        out.verify_token_count += 1;
        for (size_t i = 0; i < branches[s].tokens.size(); ++i) {
            common_batch_add(batch_tgt, branches[s].tokens[i], n_past + 1 + i, { int32_t(s) }, true);
            idxs[s].push_back(batch_tgt.n_tokens - 1);
            out.verify_token_count += 1;
        }
    }
    out.flatten_ms = 1e-3 * (ggml_time_us() - t_flatten);

    const int64_t t_verify = ggml_time_us();
    if (llama_decode(ctx_tgt, batch_tgt) != 0) {
        return out;
    }
    out.verify_ms = 1e-3 * (ggml_time_us() - t_verify);

    out.branches.reserve(branches.size());
    for (size_t s = 0; s < branches.size(); ++s) {
        common_sampler * branch_smpl = common_sampler_clone(smpl_tgt);
        auto emitted = common_sampler_sample_and_accept_n(branch_smpl, ctx_tgt, idxs[s], branches[s].tokens, false);
        common_sampler_free(branch_smpl);

        const int accepted_len = emitted.empty() ? 0 : int(emitted.size()) - 1;
        const int reject_pos = accepted_len < (int) branches[s].tokens.size() ? accepted_len : -1;
        out.branches.push_back({
            branches[s].branch_id,
            accepted_len,
            reject_pos,
            branches[s].cum_conf,
            emitted,
        });
    }

    return out;
}

static int pick_best_tree_branch(const std::vector<tree_branch_verify> & verified) {
    int best = -1;
    for (int i = 0; i < (int) verified.size(); ++i) {
        if (best < 0) {
            best = i;
            continue;
        }
        const auto & a = verified[i];
        const auto & b = verified[best];
        if (a.accepted_len != b.accepted_len) {
            if (a.accepted_len > b.accepted_len) {
                best = i;
            }
            continue;
        }
        if (std::abs(a.cum_conf - b.cum_conf) > 1e-12) {
            if (a.cum_conf > b.cum_conf) {
                best = i;
            }
            continue;
        }
        if (a.branch_id < b.branch_id) {
            best = i;
        }
    }
    return best;
}

static std::vector<tree_branch> prune_tree_branches(const spec_opts & opts, const std::vector<tree_branch> & branches, int * pruned_count) {
    if (pruned_count) {
        *pruned_count = 0;
    }
    if (branches.empty()) {
        return {};
    }

    std::vector<tree_branch> kept = branches;
    std::sort(kept.begin(), kept.end(), [](const tree_branch & a, const tree_branch & b) {
        if (std::abs(a.cum_conf - b.cum_conf) > 1e-12) {
            return a.cum_conf > b.cum_conf;
        }
        return a.branch_id < b.branch_id;
    });

    const double best_conf = kept.front().cum_conf;
    std::vector<tree_branch> out;
    out.reserve(kept.size());
    for (const auto & branch : kept) {
        if (opts.tree_verify_min_len > 0 && (int) branch.tokens.size() < opts.tree_verify_min_len) {
            continue;
        }
        if (opts.tree_verify_conf_gap < 1.0 && best_conf - branch.cum_conf > opts.tree_verify_conf_gap) {
            continue;
        }
        out.push_back(branch);
        if (opts.tree_verify_top_n > 0 && (int) out.size() >= opts.tree_verify_top_n) {
            break;
        }
    }

    if (out.empty()) {
        out.push_back(kept.front());
    }
    if (pruned_count) {
        *pruned_count = std::max(0, (int) branches.size() - (int) out.size());
    }
    return out;
}

static draft_round draft_linear_adaptive(
        llama_context * ctx_dft,
        common_sampler * smpl_dft,
        int prompt_len,
        int n_draft,
        int n_min,
        double alpha,
        llama_batch & batch_dft) {
    draft_round out;
    out.cum_conf = 1.0;

    const int64_t t0 = ggml_time_us();

    common_sampler * round_smpl = common_sampler_clone(smpl_dft);
    common_sampler_reset(round_smpl);

    for (int i = 0; i < n_draft; ++i) {
        const llama_token id = common_sampler_sample(round_smpl, ctx_dft, 0, true);
        auto * cur = common_sampler_get_candidates(round_smpl, true);
        const auto dist = copy_dist(cur);

        const double prob = std::max(token_confidence_from_logits(ctx_dft, id), 1e-8);

        out.tokens.push_back(id);
        out.dists.push_back(dist);
        out.cum_conf *= prob;

        common_sampler_accept(round_smpl, id, true);

        if (!decode_one(ctx_dft, id, prompt_len + i, batch_dft)) {
            break;
        }

        if ((int) out.tokens.size() >= std::max(1, n_min) && out.cum_conf < alpha) {
            out.fallback_triggered = true;
            break;
        }
    }

    common_sampler_free(round_smpl);
    out.ms = 1e-3 * (ggml_time_us() - t0);
    return out;
}

static int adaptive_proposal_limit(const spec_opts & opts, const adaptive_state & state, int fallback_n_max) {
    int limit = opts.adaptive_max_proposal > 0 ? opts.adaptive_max_proposal : fallback_n_max;

    if (state.alpha >= 0.20) {
        limit = std::min(limit, 2);
    } else if (state.alpha >= 0.08) {
        limit = std::min(limit, 3);
    } else if (state.alpha >= 0.03) {
        limit = std::min(limit, 4);
    }

    return std::max(limit, 1);
}

static double update_adaptive_alpha(adaptive_state & state, const spec_opts & opts, double tc, int n_correct, int n_all) {
    const double before = state.alpha;
    tc = std::max(tc, 1e-8);

    if (n_all <= 0) {
        return before;
    }

    if (n_correct >= n_all) {
        state.alpha *= opts.adaptive_success_decay;
    } else {
        const double miss_ratio = double(n_all - n_correct) / double(n_all);
        if (opts.adaptive_policy == "conservative-a") {
            state.alpha = state.alpha / std::pow(tc, miss_ratio * opts.adaptive_growth_scale);
        } else if (opts.adaptive_policy == "conservative-b") {
            const double bumped = state.alpha * std::max(opts.adaptive_error_bump, 1.0);
            const double scaled = state.alpha / std::pow(tc, miss_ratio * opts.adaptive_growth_scale);
            state.alpha = std::max(bumped, scaled);
        } else {
            state.alpha = state.alpha / std::pow(tc, miss_ratio * opts.adaptive_growth_scale);
        }
    }

    state.alpha = std::max(state.alpha_min, std::min(state.alpha_max, state.alpha));
    return before;
}

static void apply_adaptive_safety(
        adaptive_state & state,
        const spec_opts & opts,
        int proposed_len,
        int accepted_len,
        int reject_pos) {
    if (opts.adaptive_safety_window <= 0 || proposed_len <= 0) {
        return;
    }

    const double accept_rate = double(accepted_len) / double(proposed_len);
    const double reject_progress = reject_pos >= 0 ? double(reject_pos) / double(proposed_len) : 1.0;

    state.recent_accept_rates.push_back(accept_rate);
    state.recent_reject_progress.push_back(reject_progress);

    while ((int) state.recent_accept_rates.size() > opts.adaptive_safety_window) {
        state.recent_accept_rates.pop_front();
    }
    while ((int) state.recent_reject_progress.size() > opts.adaptive_safety_window) {
        state.recent_reject_progress.pop_front();
    }

    if ((int) state.recent_accept_rates.size() < opts.adaptive_safety_window) {
        return;
    }

    double accept_sum = 0.0;
    double reject_sum = 0.0;
    for (double v : state.recent_accept_rates) {
        accept_sum += v;
    }
    for (double v : state.recent_reject_progress) {
        reject_sum += v;
    }

    const double avg_accept = accept_sum / state.recent_accept_rates.size();
    const double avg_reject_progress = reject_sum / state.recent_reject_progress.size();

    if (avg_accept < opts.adaptive_safety_min_accept || avg_reject_progress < opts.adaptive_safety_early_reject) {
        state.alpha *= opts.adaptive_safety_alpha_boost;
        state.alpha = std::max(state.alpha_min, std::min(state.alpha_max, state.alpha));
    }
}

static int run_match_rate(
        llama_model * model_tgt,
        llama_model * model_dft,
        const common_params & params,
        int n_tokens) {
    auto params_tgt = params;
    auto params_dft = params;

    params_tgt.sampling.temp = 0.0f;
    params_tgt.sampling.top_k = 1;
    params_tgt.sampling.top_p = 1.0f;
    params_tgt.sampling.min_p = 0.0f;
    params_tgt.n_parallel = 1;

    params_dft = params_tgt;
    params_dft.model = params.speculative.mparams_dft;
    params_dft.devices = params.speculative.devices;
    params_dft.n_gpu_layers = params.speculative.n_gpu_layers;
    params_dft.tensor_buft_overrides = params.speculative.tensor_buft_overrides;
    params_dft.n_ctx = params.speculative.n_ctx > 0 ? params.speculative.n_ctx : params.n_ctx;

    llama_context_ptr ctx_tgt(llama_init_from_model(model_tgt, common_context_params_to_llama(params_tgt)));
    llama_context_ptr ctx_dft(llama_init_from_model(model_dft, common_context_params_to_llama(params_dft)));

    if (!ctx_tgt || !ctx_dft) {
        LOG_ERR("%s: failed to init contexts for match-rate mode\n", __func__);
        return 1;
    }

    llama_tokens prompt = common_tokenize(ctx_tgt.get(), params.prompt, true, true);

    if (!prime_context(ctx_tgt.get(), prompt) || !prime_context(ctx_dft.get(), prompt)) {
        LOG_ERR("%s: failed to prime contexts in match-rate mode\n", __func__);
        return 1;
    }

    common_sampler * smpl_tgt = common_sampler_init(model_tgt, params_tgt.sampling);
    common_sampler * smpl_dft = common_sampler_init(model_dft, params_dft.sampling);
    seed_sampler(smpl_tgt, prompt);
    seed_sampler(smpl_dft, prompt);

    llama_batch batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt.get()), 0, 1);
    llama_batch batch_dft = llama_batch_init(llama_n_batch(ctx_dft.get()), 0, 1);

    int n_match = 0;
    int n_total = 0;
    int n_past = prompt.size();

    for (int i = 0; i < n_tokens; ++i) {
        const llama_token tok_tgt = common_sampler_sample(smpl_tgt, ctx_tgt.get(), 0, true);
        const llama_token tok_dft = common_sampler_sample(smpl_dft, ctx_dft.get(), 0, true);

        if (tok_tgt == tok_dft) {
            ++n_match;
        }
        ++n_total;

        common_sampler_accept(smpl_tgt, tok_tgt, true);
        common_sampler_accept(smpl_dft, tok_dft, true);

        if (!decode_one(ctx_tgt.get(), tok_tgt, n_past, batch_tgt) ||
            !decode_one(ctx_dft.get(), tok_dft, n_past, batch_dft)) {
            break;
        }

        ++n_past;
    }

    const double match_rate = n_total > 0 ? 100.0 * n_match / n_total : 0.0;
    LOG_INF("match-rate: compared=%d matched=%d rate=%.3f%%\n", n_total, n_match, match_rate);

    llama_batch_free(batch_tgt);
    llama_batch_free(batch_dft);
    common_sampler_free(smpl_tgt);
    common_sampler_free(smpl_dft);

    return 0;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;

    spec_opts opts;
    std::vector<char *> filtered_argv;
    if (!parse_spec_opts(argc, argv, opts, filtered_argv)) {
        return 1;
    }

    if (!common_params_parse((int) filtered_argv.size(), filtered_argv.data(), params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    if (opts.debug_spec || opts.debug_spec_compare_official) {
        params.sampling.n_probs = std::max(params.sampling.n_probs, 128);
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

    if (opts.enable_provisional) {
        LOG_WRN("spec-warning: provisional remains disabled in this debug baseline build\n");
        LOG_WRN("spec-warning: enable_provisional=%d\n", opts.enable_provisional);
    }

    const int fixed_k = opts.fixed_k > 0 ? opts.fixed_k : params.speculative.n_max;
    if (opts.enable_tree) {
        params.n_parallel = std::max(params.n_parallel, opts.tree_width);
    }

    std::default_random_engine rng(params.sampling.seed == LLAMA_DEFAULT_SEED ? std::random_device()() : params.sampling.seed);

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init_tgt = common_init_from_params(params);
    llama_model * model_tgt = llama_init_tgt->model();
    llama_context * ctx_tgt = llama_init_tgt->context();

    llama_model_ptr model_dft;
    {
        auto params_dft = params;
        params_dft.n_parallel   = 1;
        params_dft.n_ctx        = params.speculative.n_ctx;
        params_dft.n_batch      = llama_n_ctx_seq(ctx_tgt);
        params_dft.devices      = params.speculative.devices;
        params_dft.model        = params.speculative.mparams_dft;
        params_dft.n_gpu_layers = params.speculative.n_gpu_layers;

        if (params.speculative.cpuparams.n_threads > 0) {
            params_dft.cpuparams.n_threads       = params.speculative.cpuparams.n_threads;
            params_dft.cpuparams_batch.n_threads = params.speculative.cpuparams_batch.n_threads;
        }

        params_dft.tensor_buft_overrides = params.speculative.tensor_buft_overrides;

        auto mparams_dft = common_model_params_to_llama(params_dft);
        model_dft.reset(llama_model_load_from_file(params_dft.model.path.c_str(), mparams_dft));
        if (model_dft == nullptr) {
            LOG_ERR("failed to load draft model, '%s'\n", params_dft.model.path.c_str());
            return 1;
        }

        params.speculative.model_dft = model_dft.get();
        params.speculative.cparams_dft = common_context_params_to_llama(params_dft);
    }

    llama_model * model_dft_raw = model_dft.get();
    const bool need_debug_draft_ctx = opts.debug_spec || opts.debug_spec_compare_official || opts.enable_adaptive || opts.enable_tree;
    llama_context * ctx_dft_dbg = nullptr;
    if (need_debug_draft_ctx) {
        ctx_dft_dbg = llama_init_from_model(model_dft_raw, params.speculative.cparams_dft);
        if (ctx_dft_dbg == nullptr) {
            LOG_ERR("%s: failed to create debug draft context\n", __func__);
            return 1;
        }
    }

    if (!check_vocab_compat(ctx_tgt, ctx_dft_dbg, model_tgt, model_dft_raw)) {
        return 1;
    }

    print_sampling_config(params.sampling);
    LOG_INF("spec-check: verification uses target logits and accepted_prefix + corrected_token merge\n");
    LOG_INF("spec-check: fixed linear baseline, adaptive_fallback=%s, tree=%s, provisional=%s\n",
            opts.enable_adaptive ? "on" : "off",
            opts.enable_tree ? "on" : "off",
            opts.enable_provisional ? "on" : "off");
    if (opts.enable_adaptive) {
        LOG_INF("spec-check: adaptive policy=%s alpha_init=%.6f success_decay=%.6f growth_scale=%.6f error_bump=%.6f alpha_min=%.6f alpha_max=%.6f max_proposal=%d safety_window=%d safety_min_accept=%.3f safety_early_reject=%.3f safety_alpha_boost=%.3f\n",
                opts.adaptive_policy.c_str(),
                opts.adaptive_alpha_init,
                opts.adaptive_success_decay,
                opts.adaptive_growth_scale,
                opts.adaptive_error_bump,
                opts.adaptive_alpha_min,
                opts.adaptive_alpha_max,
                opts.adaptive_max_proposal,
                opts.adaptive_safety_window,
                opts.adaptive_safety_min_accept,
                opts.adaptive_safety_early_reject,
                opts.adaptive_safety_alpha_boost);
    }
    if (opts.enable_tree) {
        LOG_INF("spec-check: tree width=%d depth=%d max_tokens=%d min_branch_prob=%.4f max_branch_gap=%.4f reuse_prefix=%s cache_branch_state=%s verify_top_n=%d verify_conf_gap=%.4f verify_min_len=%d debug_tree=%s\n",
                opts.tree_width, opts.tree_depth, opts.tree_max_tokens, opts.tree_min_branch_prob, opts.tree_max_branch_gap,
                opts.tree_reuse_prefix ? "true" : "false",
                opts.tree_cache_branch_state ? "true" : "false",
                opts.tree_verify_top_n,
                opts.tree_verify_conf_gap,
                opts.tree_verify_min_len,
                opts.debug_tree ? "true" : "false");
    }

    if (opts.match_rate_tokens > 0) {
        const int rc = run_match_rate(model_tgt, model_dft_raw, params, opts.match_rate_tokens);
        if (ctx_dft_dbg) {
            llama_free(ctx_dft_dbg);
        }
        llama_backend_free();
        return rc;
    }

    llama_tokens inp = common_tokenize(ctx_tgt, params.prompt, true, true);
    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);

    if ((int) inp.size() >= (int) llama_n_ctx(ctx_tgt) || (ctx_dft_dbg && (int) inp.size() >= (int) llama_n_ctx(ctx_dft_dbg))) {
        LOG_ERR("%s: prompt too long (%d tokens)\n", __func__, (int) inp.size());
        return 1;
    }

    LOG("\n\n");
    for (auto id : inp) {
        LOG("%s", token_str(ctx_tgt, id).c_str());
    }

    FILE * csv = nullptr;
    if (!opts.dump_spec_csv.empty()) {
        csv = std::fopen(opts.dump_spec_csv.c_str(), "w");
        if (csv == nullptr) {
            LOG_ERR("%s: failed to open CSV file '%s'\n", __func__, opts.dump_spec_csv.c_str());
            return 1;
        }
        std::fprintf(csv, "round_id,prompt_len,proposed_len,accepted_len,accept_rate,reject_pos,draft_ms,target_ms,total_ms,cum_conf,alpha_before,alpha_after,verify_called,fallback_triggered,branch_count,tree_token_count\n");
        std::fflush(csv);
    }

    const int64_t t_enc_start = ggml_time_us();

    common_sampler * smpl_tgt = common_sampler_init(model_tgt, params.sampling);
    if (llama_decode(ctx_tgt, llama_batch_get_one(inp.data(), inp.size() - 1)) != 0) {
        LOG_ERR("%s: failed to decode prompt into target context\n", __func__);
        return 1;
    }

    llama_token id_last = inp.back();
    llama_tokens prompt_tgt(inp.begin(), inp.end() - 1);
    prompt_tgt.reserve(llama_n_ctx(ctx_tgt));

    int n_past = inp.size() - 1;

    common_sampler * smpl_dft_dbg = nullptr;
    if (ctx_dft_dbg) {
        if (!prime_context(ctx_dft_dbg, inp)) {
            LOG_ERR("%s: failed to prime debug draft context\n", __func__);
            return 1;
        }

        smpl_dft_dbg = common_sampler_init(model_dft_raw, params.sampling);
        seed_sampler(smpl_dft_dbg, inp);
    }

    llama_context * ctx_tree_scratch = nullptr;
    if (opts.enable_tree) {
        auto cparams_tree = params.speculative.cparams_dft;
        cparams_tree.n_seq_max = std::max(cparams_tree.n_seq_max, (uint32_t) (2 * std::max(1, opts.tree_max_tokens + opts.tree_width + 2)));
        ctx_tree_scratch = llama_init_from_model(model_dft_raw, cparams_tree);
        if (ctx_tree_scratch == nullptr) {
            LOG_ERR("%s: failed to create tree scratch draft context\n", __func__);
            return 1;
        }
    }

    adaptive_state adaptive;
    adaptive.alpha = opts.adaptive_alpha_init;
    adaptive.alpha_min = opts.adaptive_alpha_min;
    adaptive.alpha_max = opts.adaptive_alpha_max;

    common_params_speculative params_spec = params.speculative;
    params_spec.n_max = fixed_k;
    struct common_speculative * spec = common_speculative_init(params_spec, ctx_tgt);
    if (spec == nullptr) {
        LOG_ERR("%s: failed to init speculative helper\n", __func__);
        return 1;
    }
    common_speculative_begin(spec, prompt_tgt);

    const int64_t t_enc_end = ggml_time_us();

    llama_batch batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, std::max(1, params.n_parallel));
    llama_batch batch_dft_dbg = llama_batch_init(1, 0, 1);

    stats_total stats;
    bool has_eos = false;
    int round_id = 0;
    int next_window = 100;
    int window_token_base = 0;
    int64_t window_t0 = ggml_time_us();

    const int64_t t_dec_start = ggml_time_us();

    while (true) {
        if ((params.n_predict >= 0 && stats.n_predict >= params.n_predict) || has_eos) {
            break;
        }

        const int prompt_len = prompt_tgt.size() + 1;
        const int64_t round_t0 = ggml_time_us();

        draft_round draft_info;
        llama_tokens draft;
        tree_draft_round tree_info;
        std::vector<tree_branch_verify> tree_verified;
        int branch_count = 1;
        int tree_token_count = 0;
        double alpha_before = adaptive.alpha;
        double alpha_after = adaptive.alpha;

        if (opts.enable_tree) {
            const int adaptive_n_max = adaptive_proposal_limit(opts, adaptive, params_spec.n_max);
            spec_opts tree_opts = opts;
            tree_opts.tree_max_tokens = std::min(opts.tree_max_tokens, adaptive_n_max);
            llama_tokens prompt_full = prompt_tgt;
            prompt_full.push_back(id_last);
            tree_info = draft_minimal_tree(ctx_tree_scratch, tree_opts, prompt_full);
            branch_count = std::max(1, (int) tree_info.branches.size());
            tree_token_count = tree_info.total_tokens;
            draft_info.ms = tree_info.ms;
            draft_info.cum_conf = tree_info.branches.empty() ? 1.0 : tree_info.branches[0].cum_conf;
            if (!tree_info.branches.empty()) {
                draft = tree_info.branches[0].tokens;
            }
        } else if (opts.enable_adaptive) {
            const int adaptive_n_max = adaptive_proposal_limit(opts, adaptive, params_spec.n_max);
            draft_info = draft_linear_adaptive(
                    ctx_dft_dbg,
                    smpl_dft_dbg,
                    prompt_len,
                    adaptive_n_max,
                    params_spec.n_min,
                    adaptive.alpha,
                    batch_dft_dbg);
            draft = draft_info.tokens;
        } else {
            const int64_t draft_t0 = ggml_time_us();
            draft = common_speculative_draft(spec, params_spec, prompt_tgt, id_last);
            draft_info.ms = 1e-3 * (ggml_time_us() - draft_t0);

            if (draft.size() < (size_t) params_spec.n_min) {
                draft.clear();
            }

            draft_info.tokens = draft;
            draft_info.cum_conf = 1.0;
        }
        const double draft_ms = draft_info.ms;

        if (!opts.enable_adaptive && ctx_dft_dbg && smpl_dft_dbg && !draft.empty()) {
            common_sampler * round_smpl = common_sampler_clone(smpl_dft_dbg);
            for (size_t i = 0; i < draft.size(); ++i) {
                const llama_token id = common_sampler_sample(round_smpl, ctx_dft_dbg, 0, true);
                common_sampler_accept(round_smpl, id, true);
                decode_one(ctx_dft_dbg, id, prompt_len + i, batch_dft_dbg);
            }
            common_sampler_free(round_smpl);
        }

        if (draft.empty() && params_spec.n_min > 0) {
            LOG_WRN("spec-warning: draft round produced 0 tokens\n");
        }

        const int64_t verify_t0 = ggml_time_us();
        std::vector<target_probe> probes;
        probes.reserve(draft.size() + 1);
        llama_tokens ids;
        int chosen_branch = 0;
        double tree_flatten_ms = 0.0;
        double tree_verify_only_ms = 0.0;
        double tree_select_ms = 0.0;
        double tree_sync_ms = 0.0;
        int tree_pruned_count = 0;
        int tree_verify_branch_count = branch_count;
        int tree_verify_token_count = 0;

        if (opts.enable_tree) {
            std::vector<tree_branch> verify_branches = prune_tree_branches(opts, tree_info.branches, &tree_pruned_count);
            tree_verify_branch_count = verify_branches.size();
            tree_verify_round tree_verify = verify_tree_branches(ctx_tgt, smpl_tgt, verify_branches, id_last, n_past, batch_tgt);
            tree_verified = std::move(tree_verify.branches);
            tree_flatten_ms = tree_verify.flatten_ms;
            tree_verify_only_ms = tree_verify.verify_ms;
            tree_verify_token_count = tree_verify.verify_token_count;
            const int64_t t_select = ggml_time_us();
            chosen_branch = pick_best_tree_branch(tree_verified);
            tree_select_ms = 1e-3 * (ggml_time_us() - t_select);
            if (chosen_branch < 0 || chosen_branch >= (int) tree_verified.size()) {
                LOG_ERR("%s: tree verification produced no valid branch\n", __func__);
                break;
            }
            ids = tree_verified[chosen_branch].emitted;
        } else {
            common_batch_clear(batch_tgt);
            common_batch_add(batch_tgt, id_last, n_past++, { 0 }, true);
            for (size_t i = 0; i < draft.size(); ++i) {
                common_batch_add(batch_tgt, draft[i], n_past + i, { 0 }, true);
            }

            if (llama_decode(ctx_tgt, batch_tgt) != 0) {
                LOG_ERR("%s: target verification decode failed\n", __func__);
                break;
            }

            if (opts.debug_spec || opts.debug_spec_compare_official) {
                common_sampler * probe_smpl = common_sampler_clone(smpl_tgt);
                for (size_t i = 0; i < draft.size(); ++i) {
                    target_probe probe = probe_target(probe_smpl, ctx_tgt, i, draft[i]);
                    common_sampler_accept(probe_smpl, probe.sampled, true);
                    probes.push_back(std::move(probe));
                    if (probes.back().sampled != draft[i]) {
                        break;
                    }
                }
                if (probes.size() == draft.size()) {
                    target_probe probe = probe_target(probe_smpl, ctx_tgt, draft.size(), LLAMA_TOKEN_NULL);
                    probes.push_back(std::move(probe));
                }
                common_sampler_free(probe_smpl);
            }

            ids = common_sampler_sample_and_accept_n(smpl_tgt, ctx_tgt, draft);
        }
        const double verify_ms = 1e-3 * (ggml_time_us() - verify_t0);
        stats.n_verify++;

        GGML_ASSERT(!ids.empty());

        const int proposed_len = opts.enable_tree ? tree_token_count : (int) draft.size();
        const int accepted_len = ids.size() - 1;
        const int emitted_len = ids.size();
        const int reject_pos = accepted_len < (int) draft.size() ? accepted_len : -1;
        if (reject_pos >= 0) {
            ++stats.n_rollbacks;
        }

        if (opts.enable_tree) {
            const int keep_pos = n_past + 1 + accepted_len;
            const int64_t t_sync = ggml_time_us();
            auto * mem_tgt = llama_get_memory(ctx_tgt);
            llama_memory_seq_rm(mem_tgt, chosen_branch, keep_pos, -1);
            llama_memory_seq_keep(mem_tgt, chosen_branch);
            llama_memory_seq_cp(mem_tgt, chosen_branch, 0, -1, -1);
            llama_memory_seq_keep(mem_tgt, 0);
            for (int s = 1; s < branch_count; ++s) {
                llama_memory_seq_rm(mem_tgt, s, -1, -1);
            }
            for (llama_token tok : ids) {
                common_sampler_accept(smpl_tgt, tok, true);
            }
            tree_sync_ms = 1e-3 * (ggml_time_us() - t_sync);
        }
        n_past += opts.enable_tree ? int(ids.size()) : int(ids.size()) - 1;

        if (opts.enable_adaptive) {
            alpha_before = adaptive.alpha;
            update_adaptive_alpha(adaptive, opts, draft_info.cum_conf, accepted_len, std::max(1, proposed_len));
            apply_adaptive_safety(adaptive, opts, proposed_len, accepted_len, reject_pos);
            alpha_after = adaptive.alpha;
            if (!opts.enable_tree) {
                sync_draft_after_verify(ctx_dft_dbg, smpl_dft_dbg, ids, prompt_len, accepted_len, batch_dft_dbg);
            }
        }

        if (!opts.enable_adaptive && smpl_dft_dbg != nullptr) {
            common_sampler_accept(smpl_dft_dbg, id_last, true);
        }
        for (size_t i = 0; i < ids.size(); ++i) {
            prompt_tgt.push_back(id_last);
            id_last = ids[i];
            if (!opts.enable_adaptive && smpl_dft_dbg != nullptr) {
                common_sampler_accept(smpl_dft_dbg, id_last, true);
            }
            if (llama_vocab_is_eog(vocab_tgt, id_last)) {
                has_eos = true;
                break;
            }
            const std::string piece = token_str(ctx_tgt, id_last);
            if (params.use_color && i + 1 < ids.size()) {
                LOG("\u001b[36m%s\u001b[37m", piece.c_str());
            } else {
                LOG("%s", piece.c_str());
            }
        }

        stats.rounds++;
        stats.n_drafted += proposed_len;
        stats.n_accept += accepted_len;
        stats.n_predict += emitted_len;
        stats.draft_ms += draft_ms;
        stats.verify_ms += verify_ms;
        stats.n_branch_total += branch_count;
        stats.n_tree_token_total += tree_token_count;
        stats.n_tree_verify_branch_total += tree_verify_branch_count;
        stats.n_tree_verify_token_total += tree_verify_token_count;
        stats.n_tree_reuse_hits += tree_info.reused_prefix_states;
        stats.n_tree_avoided_replays += tree_info.avoided_replays;
        stats.tree_replay_ms += tree_info.replay_ms;
        stats.tree_expand_ms += tree_info.expand_ms;
        stats.tree_flatten_ms += tree_flatten_ms;
        stats.tree_verify_ms += tree_verify_only_ms;
        stats.tree_select_ms += tree_select_ms;
        stats.tree_sync_ms += tree_sync_ms;

        const double total_ms = 1e-3 * (ggml_time_us() - round_t0);
        stats.total_ms += total_ms;
        const double accept_rate = proposed_len > 0 ? 100.0 * accepted_len / proposed_len : 0.0;

        if (opts.debug_spec || opts.debug_spec_compare_official) {
            LOG_INF("spec-debug: round_ctx='%s'\n", tail_text(ctx_tgt, llama_tokens(prompt_tgt.begin(), prompt_tgt.end()), 32).c_str());
            for (size_t i = 0; i < draft.size(); ++i) {
                LOG_INF("spec-debug: draft[%zu]=%d '%s'\n", i, draft[i], token_str(ctx_tgt, draft[i]).c_str());
                if (i < probes.size()) {
                    for (size_t k = 0; k < probes[i].top5.size(); ++k) {
                        LOG_INF("spec-debug: tgt_top5[%zu][%zu]=%d '%s' p=%.6f\n", i, k,
                                probes[i].top5[k].id, token_str(ctx_tgt, probes[i].top5[k].id).c_str(), probes[i].top5[k].prob);
                    }
                    LOG_INF("spec-debug: draft_hits_target_argmax=%s corrected='%s'\n",
                            probes[i].draft_hits_argmax ? "true" : "false",
                            token_str(ctx_tgt, probes[i].sampled).c_str());
                }
                if ((int) i == reject_pos) {
                    break;
                }
            }
        }

        if (opts.debug_spec_compare_official) {
            LOG_INF("spec-compare[%d]: prompt_len=%d proposed='%s' accepted_len=%d reject_pos=%d draft_reset=1 draft_accept=%d target_reset=0 target_accept=%d draft_ms=%.3f verify_ms=%.3f total_ms=%.3f\n",
                    round_id,
                    prompt_len,
                    truncate_str(string_from(ctx_tgt, draft), 160).c_str(),
                    accepted_len,
                    reject_pos,
                    proposed_len,
                    emitted_len,
                    draft_ms,
                    verify_ms,
                    total_ms);
        }

        if (opts.debug_tree && opts.enable_tree) {
            LOG_INF("tree-debug[%d]: ctx_len=%d branches=%d verify_branches=%d tree_tokens=%d verify_tokens=%d chosen_branch=%d pruned=%d reused_prefix_states=%d avoided_replays=%d replay_ms=%.3f expand_ms=%.3f flatten_ms=%.3f verify_ms=%.3f select_ms=%.3f sync_ms=%.3f\n",
                    round_id, prompt_len, branch_count, tree_verify_branch_count, tree_token_count, tree_verify_token_count, chosen_branch, tree_pruned_count,
                    tree_info.reused_prefix_states, tree_info.avoided_replays,
                    tree_info.replay_ms, tree_info.expand_ms, tree_flatten_ms, tree_verify_only_ms, tree_select_ms, tree_sync_ms);
            for (const auto & node : tree_info.nodes) {
                LOG_INF("tree-debug[%d]: node=%d parent=%d branch=%d depth=%d tok=%d '%s' local=%.6f cum=%.6f\n",
                        round_id, node.index, node.parent, node.branch_id, node.depth, node.id, node.text.c_str(), node.local_conf, node.cum_conf);
            }
            for (const auto & branch : tree_info.branches) {
                LOG_INF("tree-debug[%d]: branch=%d seq='%s' cum=%.6f\n",
                        round_id, branch.branch_id, truncate_str(string_from(ctx_tgt, branch.tokens), 160).c_str(), branch.cum_conf);
            }
            for (const auto & branch : tree_verified) {
                LOG_INF("tree-debug[%d]: verify branch=%d accepted=%d reject_pos=%d emitted='%s'\n",
                        round_id, branch.branch_id, branch.accepted_len, branch.reject_pos, truncate_str(string_from(ctx_tgt, branch.emitted), 160).c_str());
            }
        }

        if (opts.enable_adaptive) {
            LOG_INF("spec-adaptive[%d]: alpha=%.8f cum_conf=%.8f verify=yes fallback=%s proposed=%d accepted=%d\n",
                    round_id,
                    alpha_before,
                    draft_info.cum_conf,
                    draft_info.fallback_triggered ? "yes" : "no",
                    proposed_len,
                    accepted_len);
            LOG_INF("spec-adaptive[%d]: alpha_update before=%.8f after=%.8f\n",
                    round_id, alpha_before, alpha_after);
        }

        if (opts.debug_spec || opts.debug_spec_compare_official || csv != nullptr) {
            LOG_INF("spec-round[%d]: proposed=%d accepted=%d accept_rate=%.2f%% reject_pos=%d draft_ms=%.3f verify_ms=%.3f total_ms=%.3f rollback=%s\n",
                    round_id, proposed_len, accepted_len, accept_rate, reject_pos, draft_ms, verify_ms, total_ms,
                    reject_pos >= 0 ? "yes" : "no");
            LOG_INF("spec-round[%d]: rollback_ctx_lens target=%d draft=%d branch_count=%d tree_tokens=%d\n", round_id, (int) (prompt_tgt.size() + 1), (int) (prompt_tgt.size() + 1), branch_count, tree_token_count);
        }

        if (!opts.enable_tree) {
            llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0, n_past, -1);
        }

        if (csv != nullptr) {
            std::fprintf(csv, "%d,%d,%d,%d,%.6f,%d,%.6f,%.6f,%.6f,%.8f,%.8f,%.8f,%d,%d,%d,%d\n",
                    round_id, prompt_len, proposed_len, accepted_len, accept_rate, reject_pos, draft_ms, verify_ms, total_ms,
                    draft_info.cum_conf, alpha_before, alpha_after, 1, draft_info.fallback_triggered ? 1 : 0, branch_count, tree_token_count);
            std::fflush(csv);
        }

        while (stats.n_predict >= next_window) {
            const int64_t now = ggml_time_us();
            const int window_tokens = stats.n_predict - window_token_base;
            const double window_tps = window_tokens > 0 ? 1e6 * window_tokens / (now - window_t0) : 0.0;
            LOG_INF("spec-window: generated=%d avg_throughput=%.3f t/s over last %d tokens\n", stats.n_predict, window_tps, window_tokens);
            window_t0 = now;
            window_token_base = stats.n_predict;
            next_window += 100;
        }

        ++round_id;
    }

    const int64_t t_dec_end = ggml_time_us();
    const int n_input = inp.size();

    LOG("\n\n");
    LOG_INF("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input, (t_enc_end - t_enc_start) / 1e6f, n_input / ((t_enc_end - t_enc_start) / 1e6f));
    LOG_INF("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", stats.n_predict, (t_dec_end - t_dec_start) / 1e6f, stats.n_predict / ((t_dec_end - t_dec_start) / 1e6f));
    LOG_INF("\n");
    LOG_INF("n_draft   = %d\n", fixed_k);
    LOG_INF("n_rounds  = %d\n", stats.rounds);
    LOG_INF("n_predict = %d\n", stats.n_predict);
    LOG_INF("n_drafted = %d\n", stats.n_drafted);
    LOG_INF("n_accept  = %d\n", stats.n_accept);
    LOG_INF("accept    = %.3f%%\n", stats.n_drafted > 0 ? 100.0 * stats.n_accept / stats.n_drafted : 0.0);
    LOG_INF("rollback  = %d\n", stats.n_rollbacks);
    LOG_INF("draft_ms_total  = %.3f\n", stats.draft_ms);
    LOG_INF("verify_ms_total = %.3f\n", stats.verify_ms);
    LOG_INF("total_ms_rounds = %.3f\n", stats.total_ms);
    LOG_INF("n_verify  = %d\n", stats.n_verify);
    LOG_INF("verify_per_100tok = %.3f\n", stats.n_predict > 0 ? 100.0 * stats.n_verify / stats.n_predict : 0.0);
    LOG_INF("avg_branch_per_round = %.3f\n", stats.rounds > 0 ? double(stats.n_branch_total) / stats.rounds : 0.0);
    LOG_INF("avg_tree_tokens_per_round = %.3f\n", stats.rounds > 0 ? double(stats.n_tree_token_total) / stats.rounds : 0.0);
    LOG_INF("avg_tree_verify_branches_per_round = %.3f\n", stats.rounds > 0 ? double(stats.n_tree_verify_branch_total) / stats.rounds : 0.0);
    LOG_INF("avg_tree_verify_tokens_per_round = %.3f\n", stats.rounds > 0 ? double(stats.n_tree_verify_token_total) / stats.rounds : 0.0);
    LOG_INF("tree_reuse_hits_total = %d\n", stats.n_tree_reuse_hits);
    LOG_INF("tree_avoided_replays_total = %d\n", stats.n_tree_avoided_replays);
    LOG_INF("tree_replay_ms_total = %.3f\n", stats.tree_replay_ms);
    LOG_INF("tree_expand_ms_total = %.3f\n", stats.tree_expand_ms);
    LOG_INF("tree_flatten_ms_total = %.3f\n", stats.tree_flatten_ms);
    LOG_INF("tree_verify_only_ms_total = %.3f\n", stats.tree_verify_ms);
    LOG_INF("tree_select_ms_total = %.3f\n", stats.tree_select_ms);
    LOG_INF("tree_sync_ms_total = %.3f\n", stats.tree_sync_ms);

    LOG_INF("\n");
    LOG_INF("draft:\n\n");
    if (ctx_dft_dbg) {
        llama_perf_context_print(ctx_dft_dbg);
    } else {
        LOG_INF("draft perf unavailable: aligned baseline uses common_speculative internal draft context\n");
    }

    LOG_INF("\n");
    LOG_INF("target:\n\n");
    common_perf_print(ctx_tgt, smpl_tgt);

    if (csv != nullptr) {
        std::fclose(csv);
    }

    common_sampler_free(smpl_tgt);
    common_sampler_free(smpl_dft_dbg);
    common_speculative_free(spec);
    if (ctx_dft_dbg) {
        llama_free(ctx_dft_dbg);
    }
    if (ctx_tree_scratch) {
        llama_free(ctx_tree_scratch);
    }
    llama_batch_free(batch_tgt);
    llama_batch_free(batch_dft_dbg);
    llama_backend_free();

    LOG("\n\n");
    return 0;
}
