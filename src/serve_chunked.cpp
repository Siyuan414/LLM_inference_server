/**
 * serve_chunked.cpp — HTTP server with CHUNKED PREFILL (S3.4).
 *
 * serve.cpp consumes a prompt one token per forward pass, so TTFT is
 * prompt_len x step_time: measured 1,172 ms for a 256-token prompt against
 * vLLM's 123 ms (benchmarks/VS_VLLM.md). This version batches prompt tokens
 * with decode tokens in a single pass, under a per-step token budget.
 *
 * ── The step ─────────────────────────────────────────────────────────────────
 *   DRAIN    take newly-arrived requests off the HTTP queue
 *   ADMIT    move waiting -> running (lazy: no blocks reserved for the prompt)
 *   PREEMPT  guarantee the decoders can grow; evict newest-first if not
 *   PLAN     ChunkScheduler: 1 token per decode, remainder to prefill, FCFS
 *   COMMIT   append_tokens for every grant  (affordable by construction)
 *   BUILD    flat q_seq / q_pos / token-id arrays + the block table
 *   FORWARD  one varlen pass over total_q tokens
 *   SAMPLE   logits only for sequences whose chunk finished their prompt
 *   RETIRE   EOS, token limit, or client hung up
 *
 * ── Why PREEMPT runs before PLAN ─────────────────────────────────────────────
 * The scheduler grants decode tokens unconditionally (starving a mid-stream
 * decode is worse than a long step) but counts the blocks they need. If the
 * pool cannot cover the decoders at all, no plan is affordable — so free space
 * first, then plan against what is actually available. That ordering is what
 * makes every append_tokens below succeed.
 */

#include "model_config.h"
#include "weights.h"
#include "block_allocator.h"
#include "chunk_scheduler.h"
#include "varlen_layer.h"
#include "tokenizer.h"
#include "http.h"
#include "batched_pick.cuh"

#include <cublas_v2.h>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){                    \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,                      \
            cudaGetErrorString(e)); exit(1);} } while(0)

using Clock = std::chrono::steady_clock;

struct Seq {
    uint64_t id = 0;
    std::vector<int> prompt;
    std::vector<int> output;
    int len     = 0;             // tokens already in the (target) KV cache
    int draft_len = 0;           // S5: positions [0, draft_len) hold valid DRAFT KV.
                                 //     Invariant while decoding: len-1 <= draft_len <= len
    int max_new = 64;
    std::shared_ptr<http::Stream> stream;
    Clock::time_point t_submit, t_first;
    bool header_sent = false;
    bool hit_eos     = false;

    int  prompt_len() const { return (int)prompt.size(); }
    bool prefilling() const { return len < prompt_len(); }

    /// Token fed at absolute position `pos`.
    int token_at(int pos) const {
        return pos < prompt_len() ? prompt[pos] : output[pos - prompt_len()];
    }
};

int main(int argc, char** argv) {
    std::string wdir = "weights", tok_path = "onnx/tokenizer.json",
                cfg_path = "onnx/config.json",
                model_name = "llama-3.2-1b-instruct";
    int    port = 8080, block_size = 16, max_batch = 64, max_seq = 2048;
    int    max_batched_tokens = 512, max_chunk = 512;
    // How many WAITING requests may become RUNNING in one step. Separate knob
    // from the token budget, and measurement showed why they must be separate:
    // a large budget admits many prompts at once, so nearly all in-flight
    // requests prefill simultaneously and finish together — p50 TTFT 123 -> 240
    // ms even though throughput was unchanged (processor sharing across
    // requests). Capping admissions staggers prefill completions instead.
    // 0 = unlimited.
    int    max_admits_per_step = 0;
    double kv_budget_mb = 1024;
    // ── S5 speculative decoding: off unless --spec-k > 0 ──
    std::string draft_wdir, draft_cfg_path;
    int spec_k = 0;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--weights"    && i+1<argc) wdir       = argv[++i];
        else if (a == "--tokenizer"  && i+1<argc) tok_path   = argv[++i];
        else if (a == "--config"     && i+1<argc) cfg_path   = argv[++i];
        else if (a == "--model-name" && i+1<argc) model_name = argv[++i];
        else if (a == "--port"       && i+1<argc) port       = std::stoi(argv[++i]);
        else if (a == "--block-size" && i+1<argc) block_size = std::stoi(argv[++i]);
        else if (a == "--max-batch"  && i+1<argc) max_batch  = std::stoi(argv[++i]);
        else if (a == "--max-seq"    && i+1<argc) max_seq    = std::stoi(argv[++i]);
        else if (a == "--max-batched-tokens" && i+1<argc)
            max_batched_tokens = std::stoi(argv[++i]);
        else if (a == "--max-chunk"  && i+1<argc) max_chunk  = std::stoi(argv[++i]);
        else if (a == "--max-admits-per-step" && i+1<argc)
            max_admits_per_step = std::stoi(argv[++i]);
        else if (a == "--kv-budget-mb" && i+1<argc) kv_budget_mb = std::stod(argv[++i]);
        else if (a == "--draft"        && i+1<argc) draft_wdir     = argv[++i];
        else if (a == "--draft-config" && i+1<argc) draft_cfg_path = argv[++i];
        else if (a == "--spec-k"       && i+1<argc) spec_k = std::stoi(argv[++i]);
    }
    // Decodes are granted unconditionally, so the budget must at least cover a
    // full batch of them or every step overruns it.
    max_batched_tokens = std::max(max_batched_tokens, max_batch);

    // S5: a decoding sequence verifies K+1 tokens per step, and every one of
    // them needs logits — so logits/out_rows are sized per TOKEN of decode,
    // not per sequence. With spec off, decode_tokens == 1 and nothing changes.
    const bool spec = spec_k > 0;
    if (spec && (draft_wdir.empty() || draft_cfg_path.empty())) {
        fprintf(stderr, "--spec-k needs --draft <dir> and --draft-config <config.json>\n");
        return 1;
    }
    const int decode_tokens = spec ? spec_k + 1 : 1;
    const int max_out = max_batch * decode_tokens;          // logits rows per step
    max_batched_tokens = std::max(max_batched_tokens, max_out);
    // Draft pass 1 carries every prefill chunk PLUS each decoder's pending
    // token and possibly one catch-up token.
    const int max_draft_rows = max_batched_tokens + max_batch;

    ModelConfig cfg;  cfg.load(cfg_path, max_seq);  cfg.print();
    Tokenizer   tok;  tok.load(tok_path);
    Weights weights;  weights.load(wdir, cfg);

    ModelConfig dcfg;
    Weights dweights;
    if (spec) {
        dcfg.load(draft_cfg_path, max_seq);  dcfg.print();
        dweights.load(draft_wdir, dcfg);
        if (dcfg.vocab_size != cfg.vocab_size) {   // token ids must mean the same thing
            fprintf(stderr, "draft vocab %d != target vocab %d: not a valid draft\n",
                    dcfg.vocab_size, cfg.vocab_size);
            return 1;
        }
    }

    // ONE allocator and ONE block table serve both models, so a block id exists
    // in both pools at once and costs target bytes + draft bytes.
    const size_t bytes_per_block_t = (size_t)cfg.num_layers * 2 * cfg.num_kv_heads
                                   * block_size * cfg.head_dim * sizeof(__half);
    const size_t bytes_per_block_d = spec
        ? (size_t)dcfg.num_layers * 2 * dcfg.num_kv_heads * block_size * dcfg.head_dim
              * sizeof(__half)
        : 0;
    const size_t bytes_per_block = bytes_per_block_t + bytes_per_block_d;
    const int num_blocks = (int)((kv_budget_mb * 1e6) / bytes_per_block);
    const int max_blocks_per_seq = (max_seq + block_size - 1) / block_size;
    const size_t pool_per_layer = (size_t)num_blocks * cfg.num_kv_heads
                                * block_size * cfg.head_dim;

    // Row-count buffers are sized by TOKENS now, not sequences — a step may
    // carry max_batched_tokens query tokens. Only logits and out_rows stay
    // per-sequence, because only one row per sequence is ever sampled.
    __half *k_pool, *v_pool, *x, *logits;
    CK(cudaMalloc(&k_pool, pool_per_layer * cfg.num_layers * sizeof(__half)));
    CK(cudaMalloc(&v_pool, pool_per_layer * cfg.num_layers * sizeof(__half)));
    CK(cudaMalloc(&x,      (size_t)max_batched_tokens * cfg.hidden_dim * sizeof(__half)));
    CK(cudaMalloc(&logits, (size_t)max_out * cfg.vocab_size * sizeof(__half)));

    int *d_table, *d_q_seq, *d_q_pos, *d_ids, *d_out_rows;
    CK(cudaMalloc(&d_table,    (size_t)max_batch * max_blocks_per_seq * sizeof(int)));
    CK(cudaMalloc(&d_q_seq,    (size_t)max_batched_tokens * sizeof(int)));
    CK(cudaMalloc(&d_q_pos,    (size_t)max_batched_tokens * sizeof(int)));
    CK(cudaMalloc(&d_ids,      (size_t)max_batched_tokens * sizeof(int)));
    CK(cudaMalloc(&d_out_rows, (size_t)max_out * sizeof(int)));

    DecodeScratch scratch;  scratch.alloc(cfg, max_batched_tokens, max_out);
    BlockAllocator alloc;   alloc.configure({block_size, num_blocks, 0});
    BatchedPicker picker;   picker.alloc(cfg.vocab_size, max_out);
    ChunkScheduler sched;   sched.configure({max_batched_tokens, max_chunk, decode_tokens});
    cublasHandle_t blas;    cublasCreate(&blas);

    // ── S5: draft model state. Its own pool, scratch and logits; the SAME
    //    num_blocks, block table (d_table) and allocator as the target. ──────
    __half *dk_pool = nullptr, *dv_pool = nullptr, *dx = nullptr, *dlogits = nullptr;
    int *dd_ids = nullptr, *dd_q_seq = nullptr, *dd_q_pos = nullptr, *dd_out_rows = nullptr;
    DecodeScratch dscratch;
    BatchedPicker dpicker;
    if (spec) {
        const size_t dpool_per_layer = (size_t)num_blocks * dcfg.num_kv_heads
                                     * block_size * dcfg.head_dim;
        CK(cudaMalloc(&dk_pool, dpool_per_layer * dcfg.num_layers * sizeof(__half)));
        CK(cudaMalloc(&dv_pool, dpool_per_layer * dcfg.num_layers * sizeof(__half)));
        CK(cudaMalloc(&dx,      (size_t)max_draft_rows * dcfg.hidden_dim * sizeof(__half)));
        CK(cudaMalloc(&dlogits, (size_t)max_batch * dcfg.vocab_size * sizeof(__half)));
        CK(cudaMalloc(&dd_ids,      (size_t)max_draft_rows * sizeof(int)));
        CK(cudaMalloc(&dd_q_seq,    (size_t)max_draft_rows * sizeof(int)));
        CK(cudaMalloc(&dd_q_pos,    (size_t)max_draft_rows * sizeof(int)));
        CK(cudaMalloc(&dd_out_rows, (size_t)max_batch * sizeof(int)));
        dscratch.alloc(dcfg, max_draft_rows, max_batch);
        dpicker.alloc(dcfg.vocab_size, max_batch);
    }

    /// One forward pass on either model, from host arrays, returning the argmax
    /// of every out row (empty if out_rows is empty). The block table must
    /// already be uploaded to d_table — both models read the same one.
    ///   q_seq[g]  row of the flattened table that query token g belongs to
    ///   q_pos[g]  its absolute position
    /// Plumbing for S5.3c; the non-speculative path below does not use it.
    auto run_pass = [&](bool use_draft,
                        const std::vector<int>& ids, const std::vector<int>& q_seq,
                        const std::vector<int>& q_pos, const std::vector<int>& out_rows)
            -> std::vector<int> {
        const int n = (int)ids.size(), n_out = (int)out_rows.size();
        if (n == 0) return {};
        int max_scan = 0;
        for (int p : q_pos) max_scan = std::max(max_scan, p + 1);
        int *ids_d = use_draft ? dd_ids : d_ids, *seq_d = use_draft ? dd_q_seq : d_q_seq,
            *pos_d = use_draft ? dd_q_pos : d_q_pos, *out_d = use_draft ? dd_out_rows : d_out_rows;
        CK(cudaMemcpy(ids_d, ids.data(),   n * sizeof(int), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(seq_d, q_seq.data(), n * sizeof(int), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(pos_d, q_pos.data(), n * sizeof(int), cudaMemcpyHostToDevice));
        if (n_out) CK(cudaMemcpy(out_d, out_rows.data(), n_out * sizeof(int), cudaMemcpyHostToDevice));
        std::vector<int> picked;
        if (use_draft) {
            forward_varlen(dcfg, dweights, blas, dscratch, ids_d, dx, dlogits, dk_pool, dv_pool,
                           d_table, seq_d, pos_d, out_d, n, n_out, block_size, num_blocks,
                           max_blocks_per_seq, max_scan, 0);
            if (n_out) dpicker.argmax_batched(dlogits, n_out, picked, 0);
        } else {
            forward_varlen(cfg, weights, blas, scratch, ids_d, x, logits, k_pool, v_pool,
                           d_table, seq_d, pos_d, out_d, n, n_out, block_size, num_blocks,
                           max_blocks_per_seq, max_scan, 0);
            if (n_out) picker.argmax_batched(logits, n_out, picked, 0);
        }
        return picked;
    };

    if (spec) {
        printf("Speculative: K=%d (decode_tokens=%d), draft %d layers, "
               "block = %.2f MB target + %.2f MB draft\n",
               spec_k, decode_tokens, dcfg.num_layers,
               bytes_per_block_t / 1e6, bytes_per_block_d / 1e6);
        printf("KV pool:   %.0f MB -> %d blocks of %d tokens (shared table)\n",
               kv_budget_mb, num_blocks, block_size);
    }

    std::mutex mu;
    std::condition_variable cv;
    std::deque<Seq> incoming;
    uint64_t next_id = 1;
    bool shutting_down = false;

    http::Server server;
    server.set_model_name(model_name);
    server.listen_and_serve(port,
        [&](http::Request req, std::shared_ptr<http::Stream> st) {
            Seq s;
            s.prompt   = tok.encode(req.prompt, /*add_bos=*/true);
            s.max_new  = std::max(1, std::min(req.max_tokens, max_seq - 8));
            s.stream   = std::move(st);
            s.t_submit = Clock::now();
            {
                std::lock_guard<std::mutex> lk(mu);
                s.id = next_id++;
                s.stream->set_id("cmpl-" + std::to_string(s.id));
                incoming.push_back(std::move(s));
            }
            cv.notify_one();
        });

    printf("Serving:   http://0.0.0.0:%d   (chunked prefill)\n", port);
    printf("KV pool:   %.0f MB -> %d blocks of %d tokens\n",
           kv_budget_mb, num_blocks, block_size);
    printf("Budget:    %d tokens/step, max chunk %d, max batch %d\n",
           max_batched_tokens, max_chunk, max_batch);
    if (max_admits_per_step > 0)
        printf("Admissions: %d per step\n", max_admits_per_step);
    printf("\n");

    std::vector<Seq> waiting, running;
    long long preemptions = 0;

    /// Append ONE generated token to `s` and stream it. Handles TTFT, the SSE
    /// header, and EOS. Returns true when `s` must stop generating (EOS or
    /// max_new reached) — the caller must not emit further tokens for it this
    /// step. RETIRE removes the sequence afterwards. Plumbing for S5.3c.
    auto emit = [&](Seq& s, int token) -> bool {
        if (s.output.empty()) {
            s.t_first = Clock::now();
            if (s.stream->alive()) { s.stream->begin(); s.header_sent = true; }
        }
        s.output.push_back(token);
        s.hit_eos = (token == tok.eos_id() || token == tok.eot_id());
        if (!s.hit_eos && s.stream->alive()) s.stream->token(tok.decode({token}));
        return s.hit_eos || (int)s.output.size() >= s.max_new;
    };

    // ════════════════════════════════════════════════════════════════════════
    // S5.3c — YOUR CODE: one speculative step for every sequence in `plan`.
    // ════════════════════════════════════════════════════════════════════════
    /// Called after COMMIT: every entry's blocks are appended and the block
    /// table (row r = running[r]) is already uploaded to d_table.
    /// Decoders have e.n_tokens == K+1; prefill entries have their chunk size.
    ///
    /// Positions, for one decoder (K = 2), with L = s.len:
    ///
    ///   pos:      draft_len … L-1     L          L+1    L+2
    ///   token:    (catch-up)          pending    d1     d2
    ///   3B KV:    ✓                   ← target pass writes L .. L+K →
    ///   1B KV:    hole if draft_len < L
    ///
    ///   The target NEVER catches up. The draft's hole is at most one position:
    ///   after an all-accepted round it is d_K, at draft_len.
    ///
    /// Steps (your trace, as code):
    ///   DRAFT 1   prefill entries: their chunk tokens at s.len .. s.len+n-1, no out row
    ///             decoders: tokens[draft_len .. L] (token_at), out row = the last one -> d1
    ///   DRAFT j   (j = 2..K) decoders only: d_{j-1} at L+j-1, out row each -> d_j
    ///   TARGET    decoders: [pending, d1..dK] at L .. L+K, ALL K+1 rows out
    ///                       verify_start[r] = out_rows.size() BEFORE pushing them
    ///             prefill:  chunk tokens; out row only on the last token if the
    ///                       chunk reaches the end of the prompt
    ///   ACCEPT    a = leading matches of d[i] vs g[verify_start + i]
    ///             emit d1..da then g[verify_start + a] — stop early if emit() says so
    ///             s.len = L+1+a; s.draft_len = min(s.len, L+K); alloc.truncate(s.id, s.len)
    ///   PREFILL   s.len += n; s.draft_len += n; if the prompt is now complete,
    ///             emit its first token from its (packed!) index in g
    ///
    /// Reminders from the trace:
    ///   * out_rows index rows of THIS pass — they restart at 0 every pass
    ///   * g is packed: g[i] answers out_rows[i], not row i
    ///   * q_seq is r, the index in `running` (= the row of the uploaded table)
    ///
    /// Tools: run_pass(use_draft, ids, q_seq, q_pos, out_rows) -> argmax per out row
    ///        s.token_at(pos), emit(s, tok), alloc.truncate(id, len), spec_k


    auto spec_step = [&](const ChunkPlan& plan) {
        std::vector<int> dec_r;               // running index of each decoder
        std::vector<int> pre_r, pre_n;        // prefill: running index + chunk size
        for (const auto& e : plan.entries) {
            // 1. find r such that running[r].id == e.id
            // 2. prefilling? → pre_r / pre_n,  else → dec_r
            int r = -1;
            for(int i=0; i<(int)running.size(); ++i)
                if(running[i].id == e.id){ r = i; break; }
            if(r < 0){
                fprintf(stderr, "BUG: spec_step() plan entry id=%llu not in running\n",
                        (unsigned long long)e.id);
                exit(1);   
            }
            if(running[r].prefilling()){
                pre_r.push_back(r);
                pre_n.push_back(e.n_tokens);
            } else {
                dec_r.push_back(r);
            }
        }
        // ════ DRAFT pass 1 (1B): prefill chunks + each decoder's catch-up and pending ════
        std::vector<int> dr_ids, dr_seq, dr_pos, dr_out;   // rows of THIS pass
        for (size_t i = 0; i < pre_r.size(); ++i) {         // prefill: KV only, no out row
            const int r = pre_r[i], n = pre_n[i];
            const Seq& s = running[r];
            for (int j = 0; j < n; ++j) {
                dr_ids.push_back(s.token_at(s.len + j));
                dr_seq.push_back(r);
                dr_pos.push_back(s.len + j);
            }
        }
        for (size_t i = 0; i < dec_r.size(); ++i) {         // decoders: draft_len .. L
            const int r = dec_r[i];
            const Seq& s = running[r];
            const int n = s.len - s.draft_len;               // catch-up (0 or 1)
            for (int p = 0; p <= n; ++p) {
                dr_ids.push_back(s.token_at(s.draft_len + p));
                dr_seq.push_back(r);
                dr_pos.push_back(s.draft_len + p);
            }
            dr_out.push_back((int)dr_ids.size() - 1);        // last row -> d1
        }
        const std::vector<int> d1 = run_pass(/*use_draft=*/true, dr_ids, dr_seq, dr_pos, dr_out);
        std::vector<std::vector<int>> drafts(dec_r.size());  // drafts[i] = {d1, d2, …}
        for (size_t i = 0; i < dec_r.size(); ++i) drafts[i].push_back(d1[i]);

        // ════ DRAFT passes 2..K (1B): decoders only, one row each ════
        for (int j = 2; j <= spec_k; ++j) {
            std::vector<int> ids, seq, pos, out;
            for (size_t i = 0; i < dec_r.size(); ++i) {
                const Seq& s = running[dec_r[i]];
                ids.push_back(drafts[i].back());             // previous guess
                seq.push_back(dec_r[i]);
                pos.push_back(s.len + j - 1);                // d_{j-1} sits at L+j-1
                out.push_back((int)out.size());              // rows restart at 0
            }
            const std::vector<int> dj = run_pass(/*use_draft=*/true, ids, seq, pos, out);
            for (size_t i = 0; i < dec_r.size(); ++i) drafts[i].push_back(dj[i]);
        }

        // ════ TARGET pass (3B): prefill chunks + each decoder's K+1 verify rows ════
        std::vector<int> t_ids, t_seq, t_pos, t_out;
        std::vector<int> first_g(pre_r.size(), -1);          // prefill: index into g, or -1
        for (size_t i = 0; i < pre_r.size(); ++i) {
            const int r = pre_r[i], n = pre_n[i];
            const Seq& s = running[r];
            for (int j = 0; j < n; ++j) {
                t_ids.push_back(s.token_at(s.len + j));
                t_seq.push_back(r);
                t_pos.push_back(s.len + j);
            }
            if (s.len + n >= s.prompt_len()) {               // chunk finishes the prompt
                first_g[i] = (int)t_out.size();
                t_out.push_back((int)t_ids.size() - 1);
            }
        }
        std::vector<int> verify_start(dec_r.size(), -1);     // decoder: where its K+1 answers start in g
        for (size_t i = 0; i < dec_r.size(); ++i) {
            const int r = dec_r[i];
            const Seq& s = running[r];
            // ── PIECE (d), YOURS: rows [pending, d1..dK] at positions L .. L+K ──
            t_ids.push_back(s.token_at(s.len));  t_seq.push_back(r);  t_pos.push_back(s.len);   // pending
            for (int k = 0; k < spec_k; ++k) {
                t_ids.push_back(drafts[i][k]);  t_seq.push_back(r);  t_pos.push_back(s.len+k+1);   // d_{k+1}
            }
            verify_start[i] = (int)t_out.size();              // BEFORE pushing its out rows
            for (int k = 0; k <= spec_k; ++k)
                t_out.push_back((int)t_ids.size() - (spec_k +1) + k);       // all K+1 rows just pushed
        }
        const std::vector<int> g = run_pass(/*use_draft=*/false, t_ids, t_seq, t_pos, t_out);

        // ════ UPDATE prefill sequences ════
        for (size_t i = 0; i < pre_r.size(); ++i) {
            Seq& s = running[pre_r[i]];
            s.len       += pre_n[i];
            s.draft_len += pre_n[i];                         // the 1B saw the same chunk
            if (first_g[i] >= 0) emit(s, g[first_g[i]]);     // first generated token
        }

        // ════ PIECE (e), YOURS: accept, emit, roll back — per decoder ════
        for (size_t i = 0; i < dec_r.size(); ++i) {
            Seq& s = running[dec_r[i]];
            const int L = s.len;                             // pending token's position
            int a = 0;
            while (a < spec_k && drafts[i][a] == g[verify_start[i] + a]) ++a;

            // EMIT: the a accepted drafts, then the 3B's bonus token.
            // emit() returns true at EOS / max_new: emit nothing more for this seq.
            bool stop = false;
            for (int k = 0; k < a && !stop; ++k) stop = emit(s, drafts[i][k]);
            if (!stop) emit(s, g[verify_start[i] + a]);

            // ROLLBACK
            s.len       = L + 1 + a;   // pending + accepted drafts
            s.draft_len = std::min(L+1+a,L+spec_k);   // the 1B saw the same accepted drafts
            alloc.truncate(s.id, s.len);
        }
    };

    while (true) {
        // ── DRAIN ────────────────────────────────────────────────────────────
        {
            std::unique_lock<std::mutex> lk(mu);
            if (incoming.empty() && waiting.empty() && running.empty())
                cv.wait(lk, [&] { return !incoming.empty() || shutting_down; });
            if (shutting_down) break;
            while (!incoming.empty()) {
                waiting.push_back(std::move(incoming.front()));
                incoming.pop_front();
            }
        }

        // ── ADMIT (lazy: no blocks reserved for the whole prompt) ─────────────
        int admitted = 0;
        while (!waiting.empty() && (int)running.size() < max_batch) {
            if (max_admits_per_step > 0 && admitted >= max_admits_per_step) break;
            Seq& front = waiting.front();

            // Lazy allocation lost the guard eager allocation had for free: an
            // oversized prompt would be admitted, prefill until blocks run out,
            // be preempted, requeued, admitted again — forever. Reject it.
            if (!alloc.fits_ever(front.prompt_len())) {
                if (front.stream->alive()) {
                    front.stream->begin();
                    front.stream->token("[error: prompt exceeds KV cache capacity]");
                    front.stream->done(0, 0.0, 0.0, /*stop=*/true);
                }
                waiting.erase(waiting.begin());
                continue;
            }
            // Room for at least the first chunk?
            const int first = std::min(front.prompt_len(), max_chunk);
            if (!alloc.can_admit(first)) break;

            Seq s = std::move(front);
            waiting.erase(waiting.begin());
            alloc.allocate(s.id, 0);              // empty table; grows per chunk
            running.push_back(std::move(s));
            ++admitted;
        }
        if (running.empty()) continue;

        {
            auto decode_blocks_needed = [&] {
                int n = 0;
                for (const auto& s : running)
                    if (!s.prefilling())
                        n += alloc.blocks_to_append(s.id, s.len, decode_tokens);
                return n;
            };
            while (alloc.num_free() < decode_blocks_needed() && running.size() > 1) {
                Seq victim = std::move(running.back());
                running.pop_back();
                alloc.release(victim.id);
                victim.len = 0;                   // discard partial prefill too
                victim.draft_len = 0;
                victim.output.clear();
                alloc.allocate(victim.id, 0);
                waiting.push_back(std::move(victim));   // re-admitted last
                ++preemptions;
            }
        }

        // ── PLAN ─────────────────────────────────────────────────────────────
        std::vector<SeqView> views;
        views.reserve(running.size());
        for (const auto& s : running)
            views.push_back({s.id, s.len, s.prompt_len()});
        const ChunkPlan plan = sched.plan(views, alloc);
        if (plan.entries.empty()) continue;

        // Map sequence id -> index in `running`, so plan entries (which carry
        // ids, not positions) can be applied.
        std::vector<int> slot_of(running.size());
        auto index_of = [&](uint64_t id) {
            for (size_t i = 0; i < running.size(); ++i)
                if (running[i].id == id) return (int)i;
            return -1;
        };

        // ── COMMIT + BUILD ───────────────────────────────────────────────────
        std::vector<int> h_ids, h_q_seq, h_q_pos, h_out_rows;
        std::vector<int> sample_of;               // out_row index -> running index
        std::vector<uint64_t> ids_of_running;
        for (const auto& s : running) ids_of_running.push_back(s.id);

        h_ids.reserve(plan.total_tokens);
        h_q_seq.reserve(plan.total_tokens);
        h_q_pos.reserve(plan.total_tokens);

        bool commit_failed = false;
        for (const auto& e : plan.entries) {
            const int r = index_of(e.id);
            if (r < 0) continue;
            Seq& s = running[r];

            // Affordable by construction — the scheduler planned against the
            // same allocator state. A failure here means the plan and the pool
            // disagree, which is a bug rather than back-pressure.
            if (!alloc.append_tokens(s.id, s.len, e.n_tokens)) {
                fprintf(stderr, "BUG: unaffordable plan (seq %llu, n=%d, free=%d)\n",
                        (unsigned long long)s.id, e.n_tokens, alloc.num_free());
                commit_failed = true;
                break;
            }
            if (spec) continue;                   // S5.3c builds its own passes

            for (int i = 0; i < e.n_tokens; ++i) {
                const int pos = s.len + i;
                h_ids.push_back(s.token_at(pos));
                h_q_seq.push_back(r);             // row of the flattened table
                h_q_pos.push_back(pos);
            }

            // Sample only if this chunk reaches the end of the prompt. A
            // sequence still mid-prefill predicts nothing we keep — its
            // interior tokens exist to populate the KV cache.
            if (s.len + e.n_tokens >= s.prompt_len()) {
                h_out_rows.push_back((int)h_ids.size() - 1);   // last query token
                sample_of.push_back(r);
            }
        }
        if (commit_failed) continue;

        if (spec) {
            // One table upload; every draft pass and the target pass reuse it.
            const std::vector<int> table =
                alloc.flatten(ids_of_running, max_blocks_per_seq);
            CK(cudaMemcpy(d_table, table.data(), table.size()*sizeof(int),
                          cudaMemcpyHostToDevice));
            spec_step(plan);
        } else {

        const int total_q = (int)h_ids.size();
        const int n_out   = (int)h_out_rows.size();
        if (total_q == 0) continue;

        const std::vector<int> table =
            alloc.flatten(ids_of_running, max_blocks_per_seq);
        int max_scan = 0;
        for (int p : h_q_pos) max_scan = std::max(max_scan, p + 1);

        CK(cudaMemcpy(d_ids,   h_ids.data(),   total_q*sizeof(int), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_q_seq, h_q_seq.data(), total_q*sizeof(int), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_q_pos, h_q_pos.data(), total_q*sizeof(int), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_table, table.data(), table.size()*sizeof(int),
                      cudaMemcpyHostToDevice));
        if (n_out)
            CK(cudaMemcpy(d_out_rows, h_out_rows.data(), n_out*sizeof(int),
                          cudaMemcpyHostToDevice));

        // ── FORWARD ──────────────────────────────────────────────────────────
        forward_varlen(cfg, weights, blas, scratch, d_ids, x, logits,
                       k_pool, v_pool, d_table, d_q_seq, d_q_pos, d_out_rows,
                       total_q, n_out, block_size, num_blocks,
                       max_blocks_per_seq, max_scan, 0);

        // Advance lengths for every granted token, sampled or not.
        for (const auto& e : plan.entries) {
            const int r = index_of(e.id);
            if (r >= 0) running[r].len += e.n_tokens;
        }

        // ── SAMPLE + stream ──────────────────────────────────────────────────
        if (n_out) {
            std::vector<int> picked;
            picker.argmax_batched(logits, n_out, picked, 0);

            for (int o = 0; o < n_out; ++o) {
                Seq& s = running[sample_of[o]];
                if (s.output.empty()) {
                    s.t_first = Clock::now();
                    if (s.stream->alive()) { s.stream->begin(); s.header_sent = true; }
                }
                s.output.push_back(picked[o]);
                s.hit_eos = (picked[o] == tok.eos_id() || picked[o] == tok.eot_id());
                if (!s.hit_eos && s.stream->alive())
                    s.stream->token(tok.decode({picked[o]}));
            }
        }

        }   // !spec

        // ── RETIRE ───────────────────────────────────────────────────────────
        for (int r = (int)running.size() - 1; r >= 0; --r) {
            Seq& s = running[r];
            const bool at_limit = (int)s.output.size() >= s.max_new;
            const bool gone     = !s.stream->alive();
            if (!s.hit_eos && !at_limit && !gone) continue;

            if (!gone) {
                const double ttft = std::chrono::duration<double,std::milli>(
                    s.t_first - s.t_submit).count();
                const double gen = std::chrono::duration<double,std::milli>(
                    Clock::now() - s.t_first).count();
                if (!s.header_sent) s.stream->begin();
                s.stream->done((int)s.output.size(), ttft,
                               gen > 0 ? s.output.size() / (gen / 1000.0) : 0.0,
                               s.hit_eos);
            }
            alloc.release(s.id);
            running.erase(running.begin() + r);
        }
    }

    server.stop();
    printf("preemptions: %lld\n", preemptions);
    scratch.free(); weights.free(); picker.free();
    if (spec) { dscratch.free(); dweights.free(); dpicker.free(); }
    cublasDestroy(blas);
    return 0;
}
