// Test fixture for the sampling rejection-reason tracking (common/sampling.cpp).
//
// Uses a fixed RNG seed and a small vocabulary (8 tokens) to exercise:
//   - top-k filtering
//   - disabled tokens (logit-bias)
//   - empty candidate sets (failure path)
//   - compatibility: tracking on/off must produce identical sampling results
//     and identical probability normalization

#include "common.h"
#include "sampling.h"

#include "ggml.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
        exit(1); \
    } \
} while (0)

static const int32_t n_vocab = 8;

// deterministic logits for the small vocab, descending quality by token id
static std::vector<float> make_logits() {
    std::vector<float> logits(n_vocab);
    for (int32_t i = 0; i < n_vocab; i++) {
        logits[i] = 4.0f - 0.7f*i; // token 0 has the highest logit
    }
    return logits;
}

struct candidates {
    std::vector<llama_token_data> data;
    llama_token_data_array        arr;

    explicit candidates(const std::vector<float> & logits) {
        for (size_t i = 0; i < logits.size(); i++) {
            data.push_back({(llama_token) i, logits[i], 0.0f});
        }
        arr = { data.data(), data.size(), -1, false };
    }

    // empty candidate set
    candidates() {
        arr = { nullptr, 0, -1, false };
    }
};

static llama_sampler * make_chain_top_k(int32_t k, uint32_t seed) {
    auto * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(chain, llama_sampler_init_top_k(k));
    llama_sampler_chain_add(chain, llama_sampler_init_dist(seed));
    return chain;
}

static llama_sampler * make_chain_bias_top_k(const std::vector<llama_logit_bias> & biases, int32_t k, uint32_t seed) {
    auto * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(chain, llama_sampler_init_logit_bias(n_vocab, (int32_t) biases.size(), biases.data()));
    llama_sampler_chain_add(chain, llama_sampler_init_top_k(k));
    llama_sampler_chain_add(chain, llama_sampler_init_dist(seed));
    return chain;
}

// count how many times (token, reason) appears in the rejection report
static int count_rejections(const std::vector<common_sampler_rejection> & rej, llama_token token, const std::string & reason) {
    int n = 0;
    for (const auto & r : rej) {
        if (r.token == token && r.reason == reason) {
            n++;
        }
    }
    return n;
}

static int count_rejections(const std::vector<common_sampler_rejection> & rej, const std::string & reason) {
    int n = 0;
    for (const auto & r : rej) {
        if (r.reason == reason) {
            n++;
        }
    }
    return n;
}

// the probabilities of the surviving candidates must match exactly between
// the tracked and the non-tracked apply (same ops, same order)
static void check_same_result(const llama_token_data_array & a, const llama_token_data_array & b) {
    CHECK(a.selected == b.selected);
    CHECK(a.size == b.size);
    for (size_t i = 0; i < a.size; i++) {
        CHECK(a.data[i].id == b.data[i].id);
        CHECK(a.data[i].logit == b.data[i].logit);
        CHECK(a.data[i].p == b.data[i].p); // bitwise: normalization must not change
    }
}

// top-k: tokens outside the top-k must be reported as rejected by "top-k",
// and the tracked result must be identical to the plain chain apply
static void test_top_k() {
    const uint32_t seed = 1234;
    const int32_t  k    = 3;

    // control: plain chain apply (tracking off)
    llama_token_data_array control;
    llama_token control_id;
    {
        candidates c(make_logits());
        auto * chain = make_chain_top_k(k, seed);
        llama_sampler_apply(chain, &c.arr);
        control = c.arr;
        control_id = c.arr.data[c.arr.selected].id;
        llama_sampler_free(chain);
    }

    // tracked apply
    candidates c(make_logits());
    auto * chain = make_chain_top_k(k, seed);

    std::vector<common_sampler_rejection> rej;
    common_sampler_apply_tracked(chain, &c.arr, rej);

    check_same_result(control, c.arr);

    const llama_token tracked_id = c.arr.data[c.arr.selected].id;
    CHECK(tracked_id == control_id);

    // top_k(3) keeps tokens {0, 1, 2} (highest logits) and rejects {3..7}
    CHECK(c.arr.size == (size_t) k);
    CHECK(rej.size() == (size_t) (n_vocab - k));
    for (llama_token t = k; t < n_vocab; t++) {
        CHECK(count_rejections(rej, t, "top-k") == 1);
    }
    CHECK(count_rejections(rej, "dist") == 0); // dist selects but does not reject

    // probabilities are normalized over the surviving candidates only
    float sum = 0.0f;
    for (size_t i = 0; i < c.arr.size; i++) {
        sum += c.arr.data[i].p;
    }
    CHECK(fabsf(sum - 1.0f) < 1e-5);

    llama_sampler_free(chain);

    printf("test_top_k: OK (selected token %d, %zu rejections)\n", tracked_id, rej.size());
}

// disabled tokens: logit-bias with -INFINITY must be reported per token,
// and must not be double-reported by the following top-k
static void test_disabled_tokens() {
    const uint32_t seed = 42;
    const int32_t  k    = 4;

    const std::vector<llama_logit_bias> biases = {
        { 2, -INFINITY },
        { 5, -INFINITY },
    };

    // control: plain chain apply (tracking off)
    llama_token_data_array control;
    {
        candidates c(make_logits());
        auto * chain = make_chain_bias_top_k(biases, k, seed);
        llama_sampler_apply(chain, &c.arr);
        control = c.arr;
        llama_sampler_free(chain);
    }

    candidates c(make_logits());
    auto * chain = make_chain_bias_top_k(biases, k, seed);

    std::vector<common_sampler_rejection> rej;
    common_sampler_apply_tracked(chain, &c.arr, rej);

    check_same_result(control, c.arr);

    // tokens 2 and 5 are disabled by the logit-bias sampler
    CHECK(count_rejections(rej, 2, "logit-bias") == 1);
    CHECK(count_rejections(rej, 5, "logit-bias") == 1);

    // top_k(4) drops 4 of the 8 tokens; 2 of them (2 and 5) were already dead,
    // so exactly 2 live tokens are reported as rejected by top-k.
    // with descending logits, the live candidates sorted are {0,1,3,4,6,7}
    // -> top-k keeps {0,1,3,4} and rejects {6,7}
    CHECK(count_rejections(rej, "top-k") == 2);
    CHECK(count_rejections(rej, 6, "top-k") == 1);
    CHECK(count_rejections(rej, 7, "top-k") == 1);

    // no token is reported twice
    std::map<llama_token, int> per_token;
    for (const auto & r : rej) {
        per_token[r.token]++;
    }
    for (const auto & kv : per_token) {
        CHECK(kv.second == 1);
    }

    CHECK(rej.size() == 4);

    llama_sampler_free(chain);

    printf("test_disabled_tokens: OK (%zu rejections: 2x logit-bias, 2x top-k)\n", rej.size());
}

// empty candidates: applying the tracked chain to an empty candidate set must
// not crash, must record no rejections, and must leave selected == -1
static void test_empty_candidates() {
    // empty input candidate set
    {
        candidates c; // size 0
        auto * chain = make_chain_top_k(3, 1234);

        std::vector<common_sampler_rejection> rej;
        common_sampler_apply_tracked(chain, &c.arr, rej);

        CHECK(c.arr.size == 0);
        CHECK(c.arr.selected == -1); // dist reports "no selection" for empty input
        CHECK(rej.empty());

        llama_sampler_free(chain);
    }

    // all tokens disabled: every token is reported as rejected by logit-bias
    // and no live candidates remain
    {
        std::vector<llama_logit_bias> biases;
        for (llama_token t = 0; t < n_vocab; t++) {
            biases.push_back({ t, -INFINITY });
        }

        candidates c(make_logits());
        auto * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(chain, llama_sampler_init_logit_bias(n_vocab, (int32_t) biases.size(), biases.data()));

        std::vector<common_sampler_rejection> rej;
        common_sampler_apply_tracked(chain, &c.arr, rej);

        CHECK(rej.size() == (size_t) n_vocab);
        CHECK(count_rejections(rej, "logit-bias") == n_vocab);
        for (llama_token t = 0; t < n_vocab; t++) {
            CHECK(count_rejections(rej, t, "logit-bias") == 1);
        }

        // no live candidates remain
        for (size_t i = 0; i < c.arr.size; i++) {
            CHECK(c.arr.data[i].logit == -INFINITY);
        }

        llama_sampler_free(chain);
    }

    printf("test_empty_candidates: OK\n");
}

// compatibility: repeated sampling with a fixed seed must give identical
// results with tracking on and off, including probability normalization
static void test_tracking_does_not_change_results() {
    const uint32_t seed = 0xC0FFEE;

    std::vector<llama_token> ids_control;
    std::vector<llama_token> ids_tracked;

    // simulate a few sampling steps
    for (int step = 0; step < 16; step++) {
        candidates c_control(make_logits());
        candidates c_tracked(make_logits());

        auto * chain_control = make_chain_bias_top_k({{ 1, -INFINITY }}, 3, seed);
        auto * chain_tracked = make_chain_bias_top_k({{ 1, -INFINITY }}, 3, seed);

        llama_sampler_apply(chain_control, &c_control.arr);

        std::vector<common_sampler_rejection> rej;
        common_sampler_apply_tracked(chain_tracked, &c_tracked.arr, rej);

        check_same_result(c_control.arr, c_tracked.arr);

        const llama_token id_control = c_control.arr.data[c_control.arr.selected].id;
        const llama_token id_tracked = c_tracked.arr.data[c_tracked.arr.selected].id;
        CHECK(id_control == id_tracked);

        ids_control.push_back(id_control);
        ids_tracked.push_back(id_tracked);

        // every step must report token 1 as rejected by logit-bias
        CHECK(count_rejections(rej, 1, "logit-bias") == 1);

        llama_sampler_free(chain_control);
        llama_sampler_free(chain_tracked);
    }

    CHECK(ids_control == ids_tracked);

    printf("test_tracking_does_not_change_results: OK (16 steps identical)\n");
}

int main() {
    test_top_k();
    test_disabled_tokens();
    test_empty_candidates();
    test_tracking_does_not_change_results();

    printf("all rejection-tracking tests passed\n");
    return 0;
}
