// Test for the optional rejection-reason tracking in the sampling chain.
// Uses local controllable fixtures: small vocabulary, fixed random seed,
// hand-crafted logits to exercise normal and abnormal paths:
//   - top-k filtering: tokens beyond k rejected by top-k constraint
//   - disabled tokens: logit-bias sets tokens to -INFINITY
//   - empty candidates: all tokens rejected -> no candidate remains
// Also verifies tracking-off gives identical result and normalization is unaffected.

#include "ggml.h"
#include "llama.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct rejection_info {
    llama_token token;
    std::string reason;
};

struct rejection_tracker {
    bool enabled = false;
    std::vector<rejection_info> rejections;
    llama_token selected = LLAMA_TOKEN_NULL;
    bool empty_candidates = false;

    void reset() {
        rejections.clear();
        selected = LLAMA_TOKEN_NULL;
        empty_candidates = false;
    }

    void record(llama_token token, const std::string & reason) {
        for (const auto & r : rejections) {
            if (r.token == token) return;
        }
        rejections.push_back({token, reason});
    }
};

static std::vector<llama_token> visible_ids(const llama_token_data_array & cur_p) {
    std::vector<llama_token> ids;
    ids.reserve(cur_p.size);
    for (size_t i = 0; i < cur_p.size; ++i) {
        if (cur_p.data[i].logit > -INFINITY) {
            ids.push_back(cur_p.data[i].id);
        }
    }
    return ids;
}

static void apply_tracked(llama_sampler * smpl, llama_token_data_array * cur_p, rejection_tracker & tracker) {
    if (!smpl) return;
    const auto before = visible_ids(*cur_p);
    llama_sampler_apply(smpl, cur_p);
    const auto after = visible_ids(*cur_p);
    const char * name = llama_sampler_name(smpl);
    for (llama_token id : before) {
        bool found = false;
        for (llama_token a : after) {
            if (a == id) { found = true; break; }
        }
        if (!found) tracker.record(id, name);
    }
}

static void apply_chain_tracked(llama_sampler * chain, llama_token_data_array * cur_p, rejection_tracker & tracker) {
    const int n = llama_sampler_chain_n(chain);
    for (int i = 0; i < n; ++i) {
        auto * smpl = llama_sampler_chain_get(chain, i);
        if (smpl->iface->apply == nullptr) continue;
        apply_tracked(smpl, cur_p, tracker);
    }
}

static std::vector<llama_token_data> make_candidates(const std::vector<float> & logits) {
    std::vector<llama_token_data> cur;
    cur.reserve(logits.size());
    for (llama_token i = 0; i < (llama_token)logits.size(); ++i) {
        cur.push_back({i, logits[i], 0.0f});
    }
    return cur;
}

static std::string find_reason(const std::vector<rejection_info> & rejections, llama_token token) {
    for (const auto & r : rejections) {
        if (r.token == token) return r.reason;
    }
    return "";
}

static void test_top_k() {
    printf("=== test_top_k ===\n");
    const int n_vocab = 8;
    const int k = 3;
    std::vector<float> logits(n_vocab);
    for (int i = 0; i < n_vocab; ++i) logits[i] = (float)i;

    {
        auto cur = make_candidates(logits);
        llama_token_data_array cur_p = {cur.data(), cur.size(), -1, false};
        rejection_tracker tracker;
        tracker.enabled = true;
        tracker.reset();
        llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(chain, llama_sampler_init_top_k(k));
        llama_sampler_chain_add(chain, llama_sampler_init_dist(42));
        apply_chain_tracked(chain, &cur_p, tracker);
        GGML_ASSERT(cur_p.selected != -1);
        tracker.selected = cur_p.data[cur_p.selected].id;
        printf("  selected token: %d\n", tracker.selected);
        printf("  candidates after top-k: %zu\n", cur_p.size);
        for (const auto & r : tracker.rejections)
            printf("    token %d rejected by %s\n", r.token, r.reason.c_str());
        GGML_ASSERT(cur_p.size == (size_t)k);
        for (int i = 0; i < n_vocab - k; ++i)
            GGML_ASSERT(find_reason(tracker.rejections, i) == "top-k");
        for (int i = n_vocab - k; i < n_vocab; ++i)
            GGML_ASSERT(find_reason(tracker.rejections, i) == "");
        llama_sampler_free(chain);
    }

    {
        auto cur_tracked = make_candidates(logits);
        llama_token_data_array cur_p_tracked = {cur_tracked.data(), cur_tracked.size(), -1, false};
        auto cur_orig = make_candidates(logits);
        llama_token_data_array cur_p_orig = {cur_orig.data(), cur_orig.size(), -1, false};
        llama_sampler * chain_tracked = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(chain_tracked, llama_sampler_init_top_k(k));
        llama_sampler_chain_add(chain_tracked, llama_sampler_init_dist(42));
        llama_sampler * chain_orig = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(chain_orig, llama_sampler_init_top_k(k));
        llama_sampler_chain_add(chain_orig, llama_sampler_init_dist(42));
        rejection_tracker tracker;
        tracker.enabled = true;
        tracker.reset();
        apply_chain_tracked(chain_tracked, &cur_p_tracked, tracker);
        llama_sampler_apply(chain_orig, &cur_p_orig);
        GGML_ASSERT(cur_p_tracked.selected == cur_p_orig.selected);
        GGML_ASSERT(cur_p_tracked.data[cur_p_tracked.selected].id == cur_p_orig.data[cur_p_orig.selected].id);
        GGML_ASSERT(cur_p_tracked.size == cur_p_orig.size);
        for (size_t i = 0; i < cur_p_tracked.size; ++i)
            GGML_ASSERT(fabsf(cur_p_tracked.data[i].p - cur_p_orig.data[i].p) < 1e-6f);
        printf("  tracking off/on consistency: OK\n");
        llama_sampler_free(chain_tracked);
        llama_sampler_free(chain_orig);
    }
    printf("  OK\n\n");
}

static void test_disabled_tokens() {
    printf("=== test_disabled_tokens ===\n");
    const int n_vocab = 8;
    std::vector<float> logits(n_vocab);
    for (int i = 0; i < n_vocab; ++i) logits[i] = (float)i;
    std::vector<llama_logit_bias> biases;
    for (int i = 0; i < n_vocab; ++i)
        if (i % 2 == 1) biases.push_back({(llama_token)i, -INFINITY});
    auto cur = make_candidates(logits);
    llama_token_data_array cur_p = {cur.data(), cur.size(), -1, false};
    rejection_tracker tracker;
    tracker.enabled = true;
    tracker.reset();
    llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(chain, llama_sampler_init_logit_bias(n_vocab, biases.size(), biases.data()));
    llama_sampler_chain_add(chain, llama_sampler_init_dist(42));
    apply_chain_tracked(chain, &cur_p, tracker);
    GGML_ASSERT(cur_p.selected != -1);
    tracker.selected = cur_p.data[cur_p.selected].id;
    printf("  selected token: %d\n", tracker.selected);
    for (const auto & r : tracker.rejections)
        printf("    token %d rejected by %s\n", r.token, r.reason.c_str());
    for (int i = 0; i < n_vocab; ++i) {
        if (i % 2 == 1) {
            GGML_ASSERT(find_reason(tracker.rejections, i) == "logit-bias");
        } else {
            GGML_ASSERT(find_reason(tracker.rejections, i) == "");
        }
    }
    GGML_ASSERT(tracker.selected % 2 == 0);
    printf("  OK\n\n");
    llama_sampler_free(chain);
}

static void test_empty_candidates() {
    printf("=== test_empty_candidates ===\n");
    const int n_vocab = 8;
    std::vector<float> logits(n_vocab);
    for (int i = 0; i < n_vocab; ++i) logits[i] = (float)i;
    std::vector<llama_logit_bias> biases;
    for (int i = 0; i < n_vocab; ++i) biases.push_back({(llama_token)i, -INFINITY});
    auto cur = make_candidates(logits);
    llama_token_data_array cur_p = {cur.data(), cur.size(), -1, false};
    rejection_tracker tracker;
    tracker.enabled = true;
    tracker.reset();
    llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(chain, llama_sampler_init_logit_bias(n_vocab, biases.size(), biases.data()));
    llama_sampler_chain_add(chain, llama_sampler_init_dist(42));
    apply_chain_tracked(chain, &cur_p, tracker);
    for (int i = 0; i < n_vocab; ++i)
        GGML_ASSERT(find_reason(tracker.rejections, i) == "logit-bias");
    GGML_ASSERT(cur_p.selected == -1);
    tracker.empty_candidates = true;
    printf("  selected: %lld (expected -1)\n", (long long)cur_p.selected);
    printf("  empty_candidates: %d\n", tracker.empty_candidates);
    printf("  OK\n\n");
    llama_sampler_free(chain);
}

static void test_combined() {
    printf("=== test_combined ===\n");
    const int n_vocab = 8;
    const int k = 4;
    std::vector<float> logits(n_vocab);
    for (int i = 0; i < n_vocab; ++i) logits[i] = (float)i;
    std::vector<llama_logit_bias> biases;
    biases.push_back({7, -INFINITY});
    auto cur = make_candidates(logits);
    llama_token_data_array cur_p = {cur.data(), cur.size(), -1, false};
    rejection_tracker tracker;
    tracker.enabled = true;
    tracker.reset();
    llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(chain, llama_sampler_init_logit_bias(n_vocab, biases.size(), biases.data()));
    llama_sampler_chain_add(chain, llama_sampler_init_top_k(k));
    llama_sampler_chain_add(chain, llama_sampler_init_dist(42));
    apply_chain_tracked(chain, &cur_p, tracker);
    GGML_ASSERT(cur_p.selected != -1);
    tracker.selected = cur_p.data[cur_p.selected].id;
    printf("  selected token: %d\n", tracker.selected);
    for (const auto & r : tracker.rejections)
        printf("    token %d rejected by %s\n", r.token, r.reason.c_str());
    GGML_ASSERT(find_reason(tracker.rejections, 7) == "logit-bias");
    // after logit-bias removes token 7, visible candidates are 0..6
    // top-k with k=4 keeps the top 4: 3,4,5,6, so 0,1,2 are rejected by top-k
    for (int i = 0; i < 3; ++i)
        GGML_ASSERT(find_reason(tracker.rejections, i) == "top-k");
    for (int i = 3; i < 7; ++i)
        GGML_ASSERT(find_reason(tracker.rejections, i) == "");
    printf("  OK\n\n");
    llama_sampler_free(chain);
}

int main(void) {
    ggml_time_init();
    test_top_k();
    test_disabled_tokens();
    test_empty_candidates();
    test_combined();
    printf("All tests passed!\n");
    return 0;
}
