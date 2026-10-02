// Test for context cancellation resource statistics.
//
// A tiny llama model is built in memory on the CPU. The context is cancelled
// mid-decode via the abort callback, then reused. The test verifies that:
//   - the cancellation is recorded with the aborted round size
//   - the temporary batch and queued resources are released (stats zero)
//   - the next decode does not read stale state and produces correct output

#include "llama.h"
#include "ggml.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// tiny llama model dimensions
static const uint32_t N_EMBD    = 64;
static const uint32_t N_HEAD    = 2;
static const uint32_t N_HEAD_KV = 2;
static const uint32_t N_FF      = 128;
static const uint32_t N_LAYER   = 2;
static const uint32_t N_VOCAB   = 128;

static const uint32_t N_EMBD_HEAD = N_EMBD / N_HEAD;

static void add_tensor(gguf_context * ctx, const std::string & name, std::initializer_list<int64_t> ne) {
    ggml_tensor t;
    memset(&t, 0, sizeof(t));
    t.type = GGML_TYPE_F32;
    int i = 0;
    for (int64_t n : ne) {
        t.ne[i++] = n;
    }
    ggml_set_name(&t, name.c_str());
    gguf_add_tensor(ctx, &t);
}

static gguf_context * build_gguf() {
    gguf_context * ctx = gguf_init_empty();

    gguf_set_val_str(ctx, "general.architecture", "llama");
    gguf_set_val_str(ctx, "general.name", "tiny");
    gguf_set_val_u32(ctx, "llama.context_length", 256);
    gguf_set_val_u32(ctx, "llama.embedding_length", N_EMBD);
    gguf_set_val_u32(ctx, "llama.block_count", N_LAYER);
    gguf_set_val_u32(ctx, "llama.feed_forward_length", N_FF);
    gguf_set_val_u32(ctx, "llama.attention.head_count", N_HEAD);
    gguf_set_val_u32(ctx, "llama.attention.head_count_kv", N_HEAD_KV);
    gguf_set_val_u32(ctx, "llama.rope.dimension_count", N_EMBD_HEAD);
    gguf_set_val_f32(ctx, "llama.attention.layer_norm_rms_epsilon", 1e-5f);

    std::vector<std::string> tokens(N_VOCAB);
    std::vector<float>       scores(N_VOCAB, 0.0f);
    for (uint32_t i = 0; i < N_VOCAB; i++) {
        tokens[i] = "tok_" + std::to_string(i);
    }
    gguf_set_val_str(ctx, "tokenizer.ggml.model", "test");
    gguf_set_val_str(ctx, "tokenizer.ggml.pre", "default");
    {
        std::vector<const char *> ptrs(N_VOCAB);
        for (uint32_t i = 0; i < N_VOCAB; i++) {
            ptrs[i] = tokens[i].c_str();
        }
        gguf_set_arr_str(ctx, "tokenizer.ggml.tokens", ptrs.data(), N_VOCAB);
    }
    gguf_set_arr_data(ctx, "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32, scores.data(), N_VOCAB);

    add_tensor(ctx, "token_embd.weight", {N_EMBD, N_VOCAB});
    add_tensor(ctx, "output_norm.weight", {N_EMBD});

    for (uint32_t il = 0; il < N_LAYER; il++) {
        const std::string p = "blk." + std::to_string(il) + ".";
        add_tensor(ctx, p + "attn_norm.weight",   {N_EMBD});
        add_tensor(ctx, p + "attn_q.weight",      {N_EMBD, N_HEAD * N_EMBD_HEAD});
        add_tensor(ctx, p + "attn_k.weight",      {N_EMBD, N_HEAD_KV * N_EMBD_HEAD});
        add_tensor(ctx, p + "attn_v.weight",      {N_EMBD, N_HEAD_KV * N_EMBD_HEAD});
        add_tensor(ctx, p + "attn_output.weight", {N_HEAD * N_EMBD_HEAD, N_EMBD});
        add_tensor(ctx, p + "ffn_norm.weight",    {N_EMBD});
        add_tensor(ctx, p + "ffn_gate.weight",    {N_EMBD, N_FF});
        add_tensor(ctx, p + "ffn_up.weight",      {N_EMBD, N_FF});
        add_tensor(ctx, p + "ffn_down.weight",    {N_FF, N_EMBD});
    }

    return ctx;
}

static void set_tensor_data(ggml_tensor * tensor, void * userdata) {
    const uint64_t seed = *(uint64_t *) userdata;
    const size_t ne = ggml_nelements(tensor);
    std::vector<float> data(ne);
    std::mt19937 gen(seed);
    std::normal_distribution<float> dis(0.0f, 0.1f);
    for (size_t i = 0; i < ne; i++) {
        data[i] = dis(gen);
    }
    ggml_backend_tensor_set(tensor, data.data(), 0, ne * sizeof(float));
}

struct abort_ctx {
    int calls = 0;
    int threshold;
};

static bool abort_callback(void * data) {
    abort_ctx * ctx = (abort_ctx *) data;
    ctx->calls++;
    return ctx->calls >= ctx->threshold;
}

static bool nearly_equal(const std::vector<float> & a, const std::vector<float> & b, float tol) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); i++) {
        if (std::fabs(a[i] - b[i]) > tol) {
            return false;
        }
    }
    return true;
}

int main() {
    llama_backend_init();

    gguf_context * gguf = build_gguf();

    llama_model_params mparams = llama_model_default_params();
    mparams.progress_callback = [](float, void *) { return true; };

    uint64_t seed = 1234;
    llama_model * model = llama_model_init_from_user(gguf, set_tensor_data, &seed, mparams);
    if (model == nullptr) {
        fprintf(stderr, "failed to load model\n");
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = 256;
    cparams.n_batch         = 8;
    cparams.n_ubatch        = 4;
    cparams.n_threads       = 1;
    cparams.n_threads_batch = 1;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "failed to create context\n");
        return 1;
    }

    // reference context: never aborted, used to verify the reused context output
    llama_context * ctx_ref = llama_init_from_model(model, cparams);
    if (ctx_ref == nullptr) {
        fprintf(stderr, "failed to create reference context\n");
        return 1;
    }

    const uint32_t n_tokens = 8;

    // build a batch of n_tokens tokens
    auto make_batch = [&](llama_batch & batch) {
        batch = llama_batch_init(n_tokens, 0, 1);
        for (uint32_t i = 0; i < n_tokens; i++) {
            batch.token[i]     = i % N_VOCAB;
            batch.pos[i]       = i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = (i == n_tokens - 1) ? 1 : 0;
        }
        batch.n_tokens = n_tokens;
    };

    llama_batch batch;
    make_batch(batch);

    // reference decode: no abort
    int ret_ref = llama_decode(ctx_ref, batch);
    if (ret_ref != 0) {
        fprintf(stderr, "reference decode failed: %d\n", ret_ref);
        return 1;
    }
    llama_synchronize(ctx_ref);

    const float * logits_ref = llama_get_logits_ith(ctx_ref, n_tokens - 1);
    std::vector<float> ref(logits_ref, logits_ref + N_VOCAB);

    // abort decode: abort on the first graph compute call
    abort_ctx actx = {0, 1};
    llama_set_abort_callback(ctx, abort_callback, &actx);

    int ret = llama_decode(ctx, batch);
    if (ret != 2) {
        fprintf(stderr, "expected abort (ret=2), got %d\n", ret);
        return 1;
    }
    fprintf(stderr, "decode aborted as expected (ret=2)\n");

    // read the cancellation stats: the round was cancelled, resources released
    llama_cancel_stats stats = llama_get_cancel_stats(ctx);
    fprintf(stderr, "after abort: cancelled=%d n_aborted_tokens=%u n_queued_tokens=%u n_batch_tokens=%u n_outputs=%u\n",
            stats.cancelled, stats.n_aborted_tokens, stats.n_queued_tokens, stats.n_batch_tokens, stats.n_outputs);

    if (!stats.cancelled) {
        fprintf(stderr, "FAIL: cancelled flag not set\n");
        return 1;
    }
    if (stats.n_aborted_tokens != n_tokens) {
        fprintf(stderr, "FAIL: n_aborted_tokens=%u, expected %u\n", stats.n_aborted_tokens, n_tokens);
        return 1;
    }
    // the temporary batch and queued resources must be released
    if (stats.n_queued_tokens != 0) {
        fprintf(stderr, "FAIL: n_queued_tokens=%u, expected 0\n", stats.n_queued_tokens);
        return 1;
    }
    if (stats.n_batch_tokens != 0) {
        fprintf(stderr, "FAIL: n_batch_tokens=%u, expected 0\n", stats.n_batch_tokens);
        return 1;
    }
    if (stats.n_outputs != 0) {
        fprintf(stderr, "FAIL: n_outputs=%u, expected 0\n", stats.n_outputs);
        return 1;
    }

    // clear the KV memory so the next decode starts from an empty state
    llama_memory_clear(llama_get_memory(ctx), true);

    // reuse the context: decode the same batch again
    llama_set_abort_callback(ctx, nullptr, nullptr);
    ret = llama_decode(ctx, batch);
    if (ret != 0) {
        fprintf(stderr, "reuse decode failed: %d\n", ret);
        return 1;
    }
    llama_synchronize(ctx);

    // the cancellation record must be cleared after a successful decode
    stats = llama_get_cancel_stats(ctx);
    fprintf(stderr, "after reuse: cancelled=%d n_aborted_tokens=%u n_queued_tokens=%u n_batch_tokens=%u n_outputs=%u\n",
            stats.cancelled, stats.n_aborted_tokens, stats.n_queued_tokens, stats.n_batch_tokens, stats.n_outputs);

    if (stats.cancelled) {
        fprintf(stderr, "FAIL: cancelled flag still set after successful decode\n");
        return 1;
    }
    if (stats.n_aborted_tokens != 0) {
        fprintf(stderr, "FAIL: n_aborted_tokens=%u, expected 0\n", stats.n_aborted_tokens);
        return 1;
    }

    // the output must match the reference decode (no stale state leaked)
    const float * logits = llama_get_logits_ith(ctx, n_tokens - 1);
    std::vector<float> out(logits, logits + N_VOCAB);

    if (!nearly_equal(ref, out, 1e-4f)) {
        fprintf(stderr, "FAIL: reused context output does not match reference\n");
        for (uint32_t i = 0; i < N_VOCAB; i++) {
            if (std::fabs(ref[i] - out[i]) > 1e-4f) {
                fprintf(stderr, "  idx %u: ref=%f out=%f diff=%f\n", i, ref[i], out[i], ref[i] - out[i]);
            }
        }
        return 1;
    }
    fprintf(stderr, "output matches reference\n");

    llama_batch_free(batch);
    llama_free(ctx);
    llama_free(ctx_ref);
    llama_model_free(model);
    gguf_free(gguf);
    llama_backend_free();

    fprintf(stderr, "ALL TESTS PASSED\n");
    return 0;
}
