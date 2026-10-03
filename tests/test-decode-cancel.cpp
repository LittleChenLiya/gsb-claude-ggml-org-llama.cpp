// Test: resource statistics after cancelling an inference round.
//
// Scenario (CPU fixture, tiny locally-generated model):
//   1. decode a prompt that spans multiple micro-batches
//   2. cancel the round mid-way via the abort callback -> llama_decode() returns 2
//   3. verify llama_get_cancel_stats() reports the remaining resources of the cancelled round
//      (tokens done/left, ubatches done, temporary batch bytes released)
//   4. reuse the same context for a new round and verify:
//      - the output matches a fresh context bit-for-bit (no stale state is read)
//      - the cancel statistics are zeroed/invalidated after the successful round

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <clocale>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

struct abort_ctrl {
    int n_calls     =  0; // number of times the callback was invoked
    int abort_after = -1; // abort once n_calls exceeds this value (< 0: never abort)
};

bool abort_cb(void * data) {
    auto * ctrl = static_cast<abort_ctrl *>(data);

    ctrl->n_calls++;

    return ctrl->abort_after >= 0 && ctrl->n_calls > ctrl->abort_after;
}

int decode_prompt(llama_context * ctx, const std::vector<llama_token> & tokens) {
    llama_batch batch = llama_batch_init((int32_t) tokens.size(), 0, 1);

    for (size_t i = 0; i < tokens.size(); ++i) {
        common_batch_add(batch, tokens[i], (llama_pos) i, { 0 }, i + 1 == tokens.size());
    }

    const int rc = llama_decode(ctx, batch);

    llama_batch_free(batch);

    return rc;
}

bool stats_are_zero(const llama_cancel_stats & stats) {
    return stats.n_tokens_total == 0 &&
           stats.n_tokens_done  == 0 &&
           stats.n_tokens_left  == 0 &&
           stats.n_ubatch_done  == 0 &&
           stats.n_batch_bytes  == 0;
}

#define CHECK(cond, ...) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "%s:%d: CHECK failed: ", __func__, __LINE__); \
            fprintf(stderr, __VA_ARGS__); \
            fprintf(stderr, "\n"); \
            return 1; \
        } \
    } while (0)

} // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    llama_backend_init();

    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), llama_model_default_params());
    if (model == nullptr) {
        fprintf(stderr, "%s: failed to load model '%s'\n", __func__, params.model.path.c_str());
        return 1;
    }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    // prompt spans exactly n_ubatch_tot micro-batches
    const uint32_t n_ubatch     = 16;
    const uint32_t n_ubatch_tot = 4;
    const uint32_t n_tokens     = n_ubatch*n_ubatch_tot;

    std::vector<llama_token> prompt(n_tokens);
    for (uint32_t i = 0; i < n_tokens; ++i) {
        prompt[i] = (llama_token) ((i*7 + 1) % (uint32_t) n_vocab);
    }

    auto make_ctx = [&]() {
        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx     = 2*n_tokens;
        cparams.n_batch   = n_tokens;
        cparams.n_ubatch  = n_ubatch;
        cparams.n_seq_max = 1;
        cparams.no_perf   = true;
        return llama_init_from_model(model, cparams);
    };

    llama_context * ctx = make_ctx();
    if (ctx == nullptr) {
        fprintf(stderr, "%s: failed to create context\n", __func__);
        return 1;
    }

    // no round happened yet -> statistics are invalid and zeroed
    {
        llama_cancel_stats stats = {};
        stats.n_tokens_total = 12345; // sentinel, must be overwritten

        CHECK(!llama_get_cancel_stats(ctx, &stats), "cancel stats should be invalid before any decode");
        CHECK(stats_are_zero(stats), "cancel stats should be zeroed before any decode");
    }

    abort_ctrl ctrl;

    llama_set_abort_callback(ctx, abort_cb, &ctrl);

    // calibration round: count the abort callback invocations of a full decode
    ctrl = { 0, -1 };

    int rc = decode_prompt(ctx, prompt);
    CHECK(rc == 0, "calibration decode failed with rc = %d", rc);
    CHECK(ctrl.n_calls > 0, "abort callback was not invoked during decode");

    const int n_calls_full = ctrl.n_calls;

    fprintf(stderr, "%s: calibration decode: %d abort callback invocations for %u tokens (%u ubatches)\n",
            __func__, n_calls_full, n_tokens, n_ubatch_tot);

    llama_memory_clear(llama_get_memory(ctx), true);

    // cancelled round: abort mid-way through the decode
    ctrl = { 0, n_calls_full/2 };

    rc = decode_prompt(ctx, prompt);
    CHECK(rc == 2, "cancelled decode should return 2, got %d", rc);

    // the cancelled round must report read-only remaining resource statistics
    llama_cancel_stats stats = {};
    CHECK(llama_get_cancel_stats(ctx, &stats), "cancel stats should be valid after a cancelled round");

    fprintf(stderr, "%s: cancelled round: tokens %u/%u done, %u left, ubatches done %u, released batch storage %zu bytes\n",
            __func__, stats.n_tokens_done, stats.n_tokens_total, stats.n_tokens_left,
            stats.n_ubatch_done, stats.n_batch_bytes);

    CHECK(stats.n_tokens_total == n_tokens, "n_tokens_total = %u, expected %u", stats.n_tokens_total, n_tokens);
    CHECK(stats.n_tokens_done  <  n_tokens, "n_tokens_done = %u, expected < %u", stats.n_tokens_done, n_tokens);
    CHECK(stats.n_tokens_done % n_ubatch == 0, "n_tokens_done = %u is not a multiple of n_ubatch = %u", stats.n_tokens_done, n_ubatch);
    CHECK(stats.n_tokens_left == n_tokens - stats.n_tokens_done, "n_tokens_left = %u, expected %u",
            stats.n_tokens_left, n_tokens - stats.n_tokens_done);
    CHECK(stats.n_ubatch_done == stats.n_tokens_done/n_ubatch, "n_ubatch_done = %u, expected %u",
            stats.n_ubatch_done, stats.n_tokens_done/n_ubatch);
    CHECK(stats.n_batch_bytes > 0, "temporary batch storage of the cancelled round was not released");

    // the statistics are a read-only snapshot - repeated queries return the same values
    {
        llama_cancel_stats stats2 = {};
        CHECK(llama_get_cancel_stats(ctx, &stats2), "cancel stats should still be valid");
        CHECK(memcmp(&stats, &stats2, sizeof(stats)) == 0, "cancel stats changed between queries");
    }

    // the ubatches processed before the cancellation remain in the memory of the context
    // note: for hybrid/recurrent memory the aborted ubatch may still be reported by seq_pos_max,
    //       so only verify that the processed tokens are still there
    if (stats.n_tokens_done > 0) {
        const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
        CHECK(pos_max >= (llama_pos) stats.n_tokens_done - 1,
                "pos_max = %d, expected >= %d", (int) pos_max, (int) stats.n_tokens_done - 1);
    }

    // reuse the context: clear the memory, disable the abort callback and decode the same prompt
    llama_memory_clear(llama_get_memory(ctx), true);

    ctrl = { 0, -1 };

    rc = decode_prompt(ctx, prompt);
    CHECK(rc == 0, "decode after cancellation failed with rc = %d", rc);

    const float * logits_reused = llama_get_logits_ith(ctx, -1);
    CHECK(logits_reused != nullptr, "no logits after reusing the context");

    std::vector<float> logits_ref(logits_reused, logits_reused + n_vocab);

    // a successful round invalidates (zeroes) the cancel statistics
    {
        llama_cancel_stats stats3 = {};
        CHECK(!llama_get_cancel_stats(ctx, &stats3), "cancel stats should be invalid after a successful round");
        CHECK(stats_are_zero(stats3), "cancel stats should be zeroed after a successful round");
    }

    // the reused context must not read stale state: its output must match a fresh context bit-for-bit
    {
        llama_context * ctx_fresh = make_ctx();
        CHECK(ctx_fresh != nullptr, "failed to create fresh context");

        rc = decode_prompt(ctx_fresh, prompt);
        CHECK(rc == 0, "fresh decode failed with rc = %d", rc);

        const float * logits_fresh = llama_get_logits_ith(ctx_fresh, -1);
        CHECK(logits_fresh != nullptr, "no logits from fresh context");

        CHECK(memcmp(logits_ref.data(), logits_fresh, n_vocab*sizeof(float)) == 0,
                "logits of reused context differ from fresh context (stale state)");

        llama_free(ctx_fresh);
    }

    llama_free(ctx);
    llama_model_free(model);

    llama_backend_free();

    fprintf(stderr, "%s: OK\n", __func__);

    return 0;
}
