/**
 * spec_offline.cu — greedy speculative decoding, offline, batch 1 (S5.2).
 *
 * The harness (plumbing) is written. spec_round() at the bottom is YOURS.
 *
 * What it does, per prompt:
 *   1. ORACLE  plain greedy 3B decoding — the ground truth and the baseline time
 *   2. SPEC    prefill both models, then call spec_round() until n_gen tokens
 *   3. CHECK   spec output vs oracle:
 *                identical                          -> PASS
 *                diverge at j, oracle top-2 gap < eps -> PASS (fp16 near-tie)
 *                diverge at j, clear winner           -> FAIL (a bug in spec_round)
 *   4. REPORT  tokens/round, acceptance histogram, ms/token vs oracle,
 *              and the prefill (TTFT) cost of running two models
 *
 * Memory layout: TWO KV pools (3B and 1B, different bytes per block), ONE
 * BlockAllocator, ONE block table uploaded to one d_table that both models'
 * forward passes read. A block id means the same logical positions in both.
 *
 * Build:
 *   nvcc -std=c++17 -O3 -I src -I kernels tools/spec_offline.cu \
 *        -o build/spec_offline -lcublas --extended-lambda -arch=sm_120
 * Run:
 *   W=/mnt/c/Users/siyua/source/repos/cuda_ml_infer_practice/llm_server
 *   ./build/spec_offline --target $W/weights_3b --target-config $W/onnx_3b/config.json \
 *       --draft $W/weights --draft-config $W/onnx/config.json \
 *       --tokenizer $W/onnx_3b/tokenizer.json --prompts prompts.txt \
 *       --n-prompts 16 --n-gen 128 --k 2
 */

#include "model_config.h"
#include "weights.h"
#include "varlen_layer.h"
#include "tokenizer.h"
#include "block_allocator.h"
#include "batched_pick.cuh"

#include <cublas_v2.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){                   \
    printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));      \
    exit(1);} } while(0)

using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// ── One model: weights, its own KV pool, its own scratch ────────────────────
struct Model {
    ModelConfig cfg;
    Weights w;
    DecodeScratch s;
    BatchedPicker pick;
    __half *k_pool = nullptr, *v_pool = nullptr, *x = nullptr, *logits = nullptr;

    void load(const std::string& wdir, const std::string& cfg_path, int max_len,
              int num_blocks, int block_size, int max_rows, int max_out) {
        cfg.load(cfg_path, max_len);
        w.load(wdir, cfg);
        // Same num_blocks as the other model (one table), different bytes/block.
        const size_t ppl = (size_t)num_blocks * cfg.num_kv_heads * block_size * cfg.head_dim;
        CK(cudaMalloc(&k_pool, ppl * cfg.num_layers * sizeof(__half)));
        CK(cudaMalloc(&v_pool, ppl * cfg.num_layers * sizeof(__half)));
        CK(cudaMalloc(&x,      (size_t)max_rows * cfg.hidden_dim * sizeof(__half)));
        CK(cudaMalloc(&logits, (size_t)max_out  * cfg.vocab_size * sizeof(__half)));
        s.alloc(cfg, max_rows, max_out);
        pick.alloc(cfg.vocab_size, max_out);
        printf("  pool: %d blocks x %.2f MB = %.0f MB\n", num_blocks,
               2.0 * cfg.num_layers * cfg.num_kv_heads * block_size * cfg.head_dim
                   * sizeof(__half) / 1e6,
               2.0 * ppl * cfg.num_layers * sizeof(__half) / 1e6);
    }
    void free_all() {
        cudaFree(k_pool); cudaFree(v_pool); cudaFree(x); cudaFree(logits);
        s.free(); w.free(); pick.free();
    }
};

// ── The engine: two models, one allocator, one table ────────────────────────
struct Engine {
    Model target, draft;
    BlockAllocator alloc;
    cublasHandle_t blas{};
    int block_size = 16, num_blocks = 0, max_blocks = 0, max_rows = 0, max_out = 0;
    int *d_ids = nullptr, *d_seq = nullptr, *d_pos = nullptr, *d_out = nullptr,
        *d_table = nullptr;

    /// Feed `ids` to model `m` for sequence `seq`, at positions
    /// start_pos .. start_pos + ids.size() - 1. Writes their KV into m's pool.
    ///
    /// Returns the argmax of the LAST row (all_rows = false: a draft/decode step)
    /// or of EVERY row (all_rows = true: a verify pass). ret[i] is the model's
    /// prediction for the token AFTER ids[i].
    ///
    /// PRECONDITION: the allocator already covers start_pos + ids.size()
    /// positions for `seq` (call alloc.append_tokens first). Checked.
    std::vector<int> run(Model& m, uint64_t seq, const std::vector<int>& ids,
                         int start_pos, bool all_rows) {
        const int n = (int)ids.size();
        if (n <= 0 || n > max_rows) throw std::runtime_error("run: bad q_len");
        if ((int)alloc.table(seq).size() < alloc.blocks_for(start_pos + n))
            throw std::runtime_error("run: allocator does not cover positions "
                                     "[start_pos, start_pos+n) — append_tokens first");
        std::vector<int> pos(n), out;
        for (int i = 0; i < n; ++i) pos[i] = start_pos + i;
        if (all_rows) for (int i = 0; i < n; ++i) out.push_back(i);
        else          out.push_back(n - 1);
        if ((int)out.size() > max_out) throw std::runtime_error("run: too many out rows");

        const std::vector<int> table = alloc.flatten({seq}, max_blocks);
        CK(cudaMemcpy(d_table, table.data(), table.size() * sizeof(int), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_ids, ids.data(), n * sizeof(int), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_pos, pos.data(), n * sizeof(int), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_out, out.data(), out.size() * sizeof(int), cudaMemcpyHostToDevice));
        forward_varlen(m.cfg, m.w, blas, m.s, d_ids, m.x, m.logits, m.k_pool, m.v_pool,
                       d_table, d_seq, d_pos, d_out, n, (int)out.size(), block_size,
                       num_blocks, max_blocks, start_pos + n, 0);
        std::vector<int> r;
        m.pick.argmax_batched(m.logits, (int)out.size(), r, 0);
        return r;
    }

    /// top1 - top2 logit of row 0 of m's most recent logits (diagnostics only).
    float top2_gap(Model& m) {
        std::vector<__half> h(m.cfg.vocab_size);
        CK(cudaMemcpy(h.data(), m.logits, h.size() * sizeof(__half), cudaMemcpyDeviceToHost));
        float a = -1e30f, b = -1e30f;
        for (const __half& v : h) {
            const float f = __half2float(v);
            if (f > a) { b = a; a = f; } else if (f > b) b = f;
        }
        return a - b;
    }
};

// ── The sequence state spec_round() works on ────────────────────────────────
struct SpecSeq {
    uint64_t id = 0;
    std::vector<int> tokens;  // prompt + every emitted token. tokens[p] = token at
                              // position p. tokens.back() is PENDING: emitted but
                              // not yet in either model's KV cache.
    int target_len = 0;       // positions [0, target_len) hold valid 3B KV.
                              // Invariant: target_len == tokens.size() - 1
    int draft_len  = 0;       // positions [0, draft_len) hold valid 1B KV.
                              // Invariant: target_len - 1 <= draft_len <= target_len
};

struct SpecStats {
    long rounds = 0, emitted = 0;
    std::vector<long> accepted_hist;   // accepted_hist[a] = rounds that accepted a
};

int spec_round(Engine& E, SpecSeq& s, int K, SpecStats& st);   // yours, below

// ── Oracle: plain greedy 3B ─────────────────────────────────────────────────
struct GenResult { std::vector<int> gen; double prefill_ms = 0, decode_ms = 0; };

static bool is_stop(int t, const Tokenizer& tok) { return t == tok.eos_id() || t == tok.eot_id(); }

static GenResult oracle(Engine& E, const std::vector<int>& prompt, int n_gen,
                        const Tokenizer& tok) {
    const uint64_t seq = 1;
    const int P = (int)prompt.size();
    GenResult r;
    E.alloc.allocate(seq, 0);
    E.alloc.append_tokens(seq, 0, P);
    auto t0 = Clock::now();
    r.gen.push_back(E.run(E.target, seq, prompt, 0, false)[0]);
    r.prefill_ms = ms_since(t0);
    t0 = Clock::now();
    while ((int)r.gen.size() < n_gen && !is_stop(r.gen.back(), tok)) {
        const int pos = P + (int)r.gen.size() - 1;
        E.alloc.append_tokens(seq, pos, 1);
        r.gen.push_back(E.run(E.target, seq, {r.gen.back()}, pos, false)[0]);
    }
    r.decode_ms = ms_since(t0);
    E.alloc.release(seq);
    return r;
}

// ── Speculative: prefill both, then rounds ──────────────────────────────────
static GenResult speculative(Engine& E, const std::vector<int>& prompt, int n_gen,
                             int K, const Tokenizer& tok, SpecStats& st, bool& ok) {
    const int P = (int)prompt.size();
    GenResult r;
    SpecSeq s;
    s.id = 2;
    E.alloc.allocate(s.id, 0);
    E.alloc.append_tokens(s.id, 0, P);

    auto t0 = Clock::now();
    const int first = E.run(E.target, s.id, prompt, 0, false)[0];
    E.run(E.draft, s.id, prompt, 0, false);            // the draft needs prompt KV too
    r.prefill_ms = ms_since(t0);

    s.tokens = prompt;
    s.tokens.push_back(first);
    s.target_len = P;
    s.draft_len  = P;

    ok = true;
    t0 = Clock::now();
    while ((int)s.tokens.size() - P < n_gen && !is_stop(s.tokens.back(), tok)) {
        const size_t before = s.tokens.size();
        const int n = spec_round(E, s, K, st);
        if (n <= 0) { ok = false; break; }
        if ((int)(s.tokens.size() - before) != n)
            throw std::runtime_error("spec_round: return value != tokens appended");
        if (s.target_len != (int)s.tokens.size() - 1)
            throw std::runtime_error("spec_round: broke invariant target_len == tokens.size()-1");
        if (s.draft_len > s.target_len || s.draft_len < s.target_len - 1)
            throw std::runtime_error("spec_round: broke invariant on draft_len");
        if ((int)E.alloc.table(s.id).size() != E.alloc.blocks_for(s.target_len))
            throw std::runtime_error("spec_round: table not truncated to target_len");
    }
    r.decode_ms = ms_since(t0);

    r.gen.assign(s.tokens.begin() + P, s.tokens.end());
    for (size_t i = 0; i < r.gen.size(); ++i)          // a round may overshoot EOS
        if (is_stop(r.gen[i], tok)) { r.gen.resize(i + 1); break; }
    if ((int)r.gen.size() > n_gen) r.gen.resize(n_gen); // ...or n_gen
    E.alloc.release(s.id);
    return r;
}

/// Top-2 gap of the 3B at the context prompt + gen[0..j-1]: was position j a
/// near-tie that fp16 accumulation order could flip?
static float gap_at(Engine& E, const std::vector<int>& prompt,
                    const std::vector<int>& gen, int j) {
    const uint64_t seq = 3;
    std::vector<int> ids = prompt;
    ids.insert(ids.end(), gen.begin(), gen.begin() + j);
    E.alloc.allocate(seq, 0);
    E.alloc.append_tokens(seq, 0, (int)ids.size());
    E.run(E.target, seq, ids, 0, false);
    const float g = E.top2_gap(E.target);
    E.alloc.release(seq);
    return g;
}

int main(int argc, char** argv) {
    std::string tdir = "weights_3b", tcfg = "onnx_3b/config.json";
    std::string ddir = "weights",    dcfg = "onnx/config.json";
    std::string tok_path = "onnx_3b/tokenizer.json", prompts_path = "prompts.txt";
    int n_prompts = 16, n_gen = 128, K = 2, max_len = 1024;
    float gap_eps = 0.125f;   // fp16 logits near 20 have ~0.016 resolution;
                              // a few ulps of accumulation drift fits under this

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&] { return std::string(argv[++i]); };
        if      (a == "--target"        && i+1<argc) tdir = next();
        else if (a == "--target-config" && i+1<argc) tcfg = next();
        else if (a == "--draft"         && i+1<argc) ddir = next();
        else if (a == "--draft-config"  && i+1<argc) dcfg = next();
        else if (a == "--tokenizer"     && i+1<argc) tok_path = next();
        else if (a == "--prompts"       && i+1<argc) prompts_path = next();
        else if (a == "--n-prompts"     && i+1<argc) n_prompts = std::stoi(next());
        else if (a == "--n-gen"         && i+1<argc) n_gen = std::stoi(next());
        else if (a == "--k"             && i+1<argc) K = std::stoi(next());
        else if (a == "--max-len"       && i+1<argc) max_len = std::stoi(next());
        else if (a == "--gap-eps"       && i+1<argc) gap_eps = std::stof(next());
    }

    Tokenizer tok; tok.load(tok_path);
    std::vector<std::vector<int>> prompts;
    {
        std::ifstream f(prompts_path);
        std::string line;
        while ((int)prompts.size() < n_prompts && std::getline(f, line))
            if (!line.empty()) prompts.push_back(tok.encode(line, true));
    }

    Engine E;
    E.max_rows   = max_len;
    E.max_out    = K + 1;
    E.max_blocks = (max_len + E.block_size - 1) / E.block_size;
    E.num_blocks = 4 * E.max_blocks;      // oracle, spec and gap_at never overlap
    E.alloc.configure({E.block_size, E.num_blocks, 0});
    CK(cudaMalloc(&E.d_ids, E.max_rows * sizeof(int)));
    CK(cudaMalloc(&E.d_seq, E.max_rows * sizeof(int)));
    CK(cudaMalloc(&E.d_pos, E.max_rows * sizeof(int)));
    CK(cudaMalloc(&E.d_out, E.max_out  * sizeof(int)));
    CK(cudaMalloc(&E.d_table, (size_t)E.max_blocks * sizeof(int)));
    CK(cudaMemset(E.d_seq, 0, E.max_rows * sizeof(int)));   // batch 1: always row 0
    cublasCreate(&E.blas);

    printf("target:\n"); E.target.load(tdir, tcfg, max_len, E.num_blocks, E.block_size, E.max_rows, E.max_out);
    printf("draft:\n");  E.draft.load (ddir, dcfg, max_len, E.num_blocks, E.block_size, E.max_rows, E.max_out);
    if (E.target.cfg.vocab_size != E.draft.cfg.vocab_size) {
        printf("vocab mismatch: %d vs %d — not a valid draft\n",
               E.target.cfg.vocab_size, E.draft.cfg.vocab_size);
        return 1;
    }

    // Warm-up: first cuBLAS calls pick algorithms and allocate workspace.
    { auto w = oracle(E, prompts[0], 8, tok); (void)w; }

    printf("\n%zu prompts, n_gen %d, K %d\n\n", prompts.size(), n_gen, K);
    printf("  #   verdict              tokens  oracle ms/tok  spec ms/tok  speedup\n");

    SpecStats st;
    st.accepted_hist.assign(K + 1, 0);
    double o_dec = 0, s_dec = 0, o_pre = 0, s_pre = 0;
    long o_tok = 0, s_tok = 0;
    int pass = 0, near = 0, fail = 0;

    for (size_t i = 0; i < prompts.size(); ++i) {
        const GenResult o = oracle(E, prompts[i], n_gen, tok);
        bool implemented = true;
        const GenResult s = speculative(E, prompts[i], n_gen, K, tok, st, implemented);
        if (!implemented) {
            printf("\nspec_round() returned 0 — not implemented yet. Oracle ran fine:\n"
                   "  %zu tokens, %.2f ms/token. Write spec_round() and rerun.\n",
                   o.gen.size(), o.decode_ms / std::max<size_t>(1, o.gen.size() - 1));
            return 0;
        }

        int j = 0;
        const int m = (int)std::min(o.gen.size(), s.gen.size());
        while (j < m && o.gen[j] == s.gen[j]) ++j;
        char verdict[64];
        if (j == m && o.gen.size() == s.gen.size()) {
            snprintf(verdict, sizeof verdict, "PASS identical");
            ++pass;
        } else {
            const float g = gap_at(E, prompts[i], o.gen, j);
            if (g < gap_eps) { snprintf(verdict, sizeof verdict, "PASS tie@%d gap %.3f", j, g); ++near; }
            else             { snprintf(verdict, sizeof verdict, "FAIL @%d gap %.3f", j, g); ++fail; }
        }

        // Decode-only rates; the first token comes from prefill in both.
        const double o_rate = o.decode_ms / std::max<size_t>(1, o.gen.size() - 1);
        const double s_rate = s.decode_ms / std::max<size_t>(1, s.gen.size() - 1);
        printf("  %-3zu %-20s %6zu  %13.2f  %11.2f  %6.2fx\n",
               i, verdict, s.gen.size(), o_rate, s_rate, o_rate / s_rate);
        o_dec += o.decode_ms; s_dec += s.decode_ms;
        o_pre += o.prefill_ms; s_pre += s.prefill_ms;
        o_tok += (long)o.gen.size() - 1; s_tok += (long)s.gen.size() - 1;
    }

    const double o_rate = o_dec / std::max(1L, o_tok), s_rate = s_dec / std::max(1L, s_tok);
    printf("\ncorrectness: %d identical, %d near-tie divergences, %d FAIL\n", pass, near, fail);
    printf("tokens/round (tau): %.3f   over %ld rounds\n",
           st.rounds ? (double)st.emitted / st.rounds : 0.0, st.rounds);
    printf("accepted per round:");
    for (int a = 0; a <= K; ++a)
        printf("  a=%d: %.1f%%", a, st.rounds ? 100.0 * st.accepted_hist[a] / st.rounds : 0.0);
    printf("\ndecode: oracle %.2f ms/tok, spec %.2f ms/tok  ->  %.3fx  (spec_measure predicted, K=2, b1: 1.26x)\n",
           o_rate, s_rate, o_rate / s_rate);
    printf("prefill (TTFT): oracle %.1f ms, spec %.1f ms per prompt  ->  +%.0f%%\n",
           o_pre / prompts.size(), s_pre / prompts.size(), 100.0 * (s_pre / o_pre - 1.0));

    E.target.free_all(); E.draft.free_all();
    cublasDestroy(E.blas);
    return fail ? 1 : 0;
}

// ════════════════════════════════════════════════════════════════════════════
// YOUR CODE
// ════════════════════════════════════════════════════════════════════════════
/**
 * One greedy speculative round for sequence `s`. Appends the emitted tokens to
 * s.tokens, updates s.target_len / s.draft_len, truncates the shared table,
 * updates `st`, and returns how many tokens were appended (always >= 1).
 *
 * Your rules (L = s.target_len at entry, the position of the pending token):
 *   draft writes   L .. L+K-1   (plus any catch-up position before L)
 *   verify writes  L .. L+K
 *   target_len   = L + 1 + a
 *   draft_len    = min(target_len, L + K)
 *   bonus        = g[a]
 *   draft q_len  = target_len - draft_len + 1   (at the START of the next round)
 *
 * Tools:
 *   E.alloc.append_tokens(s.id, cur_len, n)   grow the table (returns false if full)
 *   E.alloc.truncate(s.id, new_len)           your S5.1 function
 *   E.run(E.draft,  s.id, ids, start_pos, false)  -> {next token after ids.back()}
 *   E.run(E.target, s.id, ids, start_pos, true)   -> one prediction per row
 *   s.tokens[p]                               the token at position p
 *
 * Steps:
 *   1. ROOM      make the table cover every position this round will write

 *   2. DRAFT     first pass feeds s.tokens[s.draft_len .. L] (catch-up + pending);
 *                then K-1 single-token passes. Collect d[0..K-1].
 *   3. VERIFY    one target pass over {pending, d...} starting at L, all rows -> g
 *   4. ACCEPT    a = number of leading drafts the target agrees with
 *   5. EMIT      append the accepted drafts and the bonus to s.tokens
 *   6. ROLLBACK  set target_len and draft_len; truncate the table to target_len
 *   7. STATS     st.rounds, st.emitted, st.accepted_hist[a]
 *
 * The harness checks your invariants after every round and throws with a
 * message naming the one you broke.
 */

int spec_round(Engine& E, SpecSeq& s, int K, SpecStats& st) {
    
    const int L = s.target_len;
    //allocate enough room for the round, including the pending token
    if (!E.alloc.append_tokens(s.id, L, K + 1))
        throw std::runtime_error("spec_round: KV pool exhausted");
    //draft, first pass feeds s.tokens[s.draft_len .. L] (catch-up + pending)
    
    std::vector<int> ids;
    for (size_t i = s.draft_len; i <= L; ++i) {
        ids.push_back(s.tokens[i]);
    }
    std::vector<int> d = E.run(E.draft, s.id, ids, s.draft_len, false);
    //then K-1 single-token passes
    for (int i = 1; i < K; ++i) {
        E.alloc.append_tokens(s.id, L + i, 1);
        d.push_back(E.run(E.draft, s.id, {d.back()}, L + i, false)[0]);
    }
    //verify, one target pass over {pending, d...} starting at L, all rows -> g
    std::vector<int> verify_ids = {s.tokens[L]};
    verify_ids.insert(verify_ids.end(), d.begin(), d.end());
    std::vector<int> g = E.run(E.target, s.id, verify_ids, L, true);
    //accept, a = number of leading drafts the target agrees with
    int a = 0;
    while(a < K && g[a] == d[a]){
        a++;
    }
    //emit, append the accepted drafts and the bonus to s.tokens
    for(int i=0; i < a; ++i){
        s.tokens.push_back(d[i]);
    }
    s.tokens.push_back(g[a]);
    //rollback, set target_len and draft_len; truncate the table to target_len
    s.target_len = L + 1 + a;
    s.draft_len = std::min(s.target_len,(L + K));
    E.alloc.truncate(s.id, s.target_len);
    //stats, st.rounds, st.emitted, st.accepted_hist[a]
    st.rounds++;
    st.emitted += a + 1;
    st.accepted_hist[a]++;
    return a + 1;
   
}
