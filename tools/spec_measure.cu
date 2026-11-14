/**
 * spec_measure.cu — should speculative decoding exist here, and at what K? (S5.0)
 *
 * Measure before building. Speculative decoding's payoff is fixed by three
 * numbers, all measurable WITHOUT writing any speculative machinery:
 *
 *   c_t       target step time, q_len = 1           (what we are trying to beat)
 *   c_d       draft  step time, q_len = 1           (paid K times per round)
 *   c_v(K)    target step time, q_len = K+1, logits on every row (one verify)
 *   tau(K)    tokens emitted per round (accepted drafts + 1 bonus)
 *
 *   speedup(K) = tau(K) * c_t / (K * c_d + c_v(K))
 *
 * tau is measured by TEACHER FORCING, which is exact for greedy decoding:
 *   1. the target generates N tokens greedily (ground truth)
 *   2. the draft reads prompt+generation in ONE pass, logits on every row
 *   3. match[j] = (draft argmax at j == target token j)
 * Every accepted prefix in greedy spec decoding IS the target's own output, so
 * the draft's context in a real round equals its teacher-forced context. The
 * round simulation over match[] therefore gives exactly the tokens/round a real
 * implementation would see (modulo fp16 near-ties) — not the i.i.d. formula.
 *
 * Free correctness check, same mechanism: the TARGET is also teacher-forced
 * over its own output. Multi-row logits must reproduce token-by-token decode;
 * that equivalence is precisely what one verify pass relies on. Expect ~100%.
 *
 * Models are loaded one at a time (3B then 1B) so this runs even if both
 * would not fit alongside their pools.
 *
 * Build:
 *   nvcc -std=c++17 -O3 -I src -I kernels tools/spec_measure.cu \
 *        -o build/spec_measure -lcublas --extended-lambda -arch=sm_120
 * Run (idle GPU — check nvidia-smi):
 *   ./build/spec_measure --target weights_3b --target-config onnx_3b/config.json \
 *       --draft weights --draft-config onnx/config.json \
 *       --tokenizer onnx_3b/tokenizer.json --prompts prompts.txt \
 *       --n-prompts 16 --n-gen 128 --batch 1 --context 256 --k-max 8
 */

#include "model_config.h"
#include "weights.h"
#include "varlen_layer.h"
#include "tokenizer.h"
#include "batched_pick.cuh"

#include <cublas_v2.h>
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){                   \
    printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));      \
    exit(1);} } while(0)

// ── One model with its own pool, laid out contiguously per sequence ─────────
struct Model {
    ModelConfig cfg;
    Weights w;
    DecodeScratch s;
    BatchedPicker pick;
    __half *k_pool = nullptr, *v_pool = nullptr, *x = nullptr, *logits = nullptr;
    int *d_ids, *d_seq, *d_pos, *d_tab, *d_out;
    int block_size = 16, max_len = 0, max_seqs = 0, max_rows = 0, max_out = 0;
    int max_blocks = 0, num_blocks = 0;

    void load(const std::string& wdir, const std::string& cfg_path,
              int max_len_, int max_seqs_, int max_out_) {
        max_len = max_len_; max_seqs = max_seqs_; max_out = max_out_;
        max_rows = std::max(max_len, max_out);
        cfg.load(cfg_path, max_len);
        w.load(wdir, cfg);
        max_blocks = (max_len + block_size - 1) / block_size;
        num_blocks = max_seqs * max_blocks;
        const size_t ppl = (size_t)num_blocks * cfg.num_kv_heads * block_size * cfg.head_dim;
        CK(cudaMalloc(&k_pool, ppl * cfg.num_layers * sizeof(__half)));
        CK(cudaMalloc(&v_pool, ppl * cfg.num_layers * sizeof(__half)));
        CK(cudaMalloc(&x,      (size_t)max_rows * cfg.hidden_dim * sizeof(__half)));
        CK(cudaMalloc(&logits, (size_t)max_out  * cfg.vocab_size * sizeof(__half)));
        CK(cudaMalloc(&d_ids, max_rows * sizeof(int)));
        CK(cudaMalloc(&d_seq, max_rows * sizeof(int)));
        CK(cudaMalloc(&d_pos, max_rows * sizeof(int)));
        CK(cudaMalloc(&d_out, max_out  * sizeof(int)));
        CK(cudaMalloc(&d_tab, (size_t)max_seqs * max_blocks * sizeof(int)));
        std::vector<int> tab((size_t)max_seqs * max_blocks);
        for (size_t i = 0; i < tab.size(); ++i) tab[i] = (int)i;   // seq s owns a contiguous run
        CK(cudaMemcpy(d_tab, tab.data(), tab.size() * sizeof(int), cudaMemcpyHostToDevice));
        s.alloc(cfg, max_rows, max_out);
        pick.alloc(cfg.vocab_size, max_out);
    }

    void free_all() {
        cudaFree(k_pool); cudaFree(v_pool); cudaFree(x); cudaFree(logits);
        cudaFree(d_ids); cudaFree(d_seq); cudaFree(d_pos); cudaFree(d_out); cudaFree(d_tab);
        s.free(); w.free(); pick.free();
    }

    /// Upload a batch description and run one forward. No sync.
    void forward(cublasHandle_t blas, const std::vector<int>& ids,
                 const std::vector<int>& seq, const std::vector<int>& pos,
                 const std::vector<int>& out) {
        const int n = (int)ids.size(), n_out = (int)out.size();
        if (n > max_rows || n_out > max_out) { printf("batch too large\n"); exit(1); }
        int max_scan = 0;
        for (int p : pos) max_scan = std::max(max_scan, p + 1);
        CK(cudaMemcpy(d_ids, ids.data(), n * sizeof(int), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_seq, seq.data(), n * sizeof(int), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_pos, pos.data(), n * sizeof(int), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_out, out.data(), n_out * sizeof(int), cudaMemcpyHostToDevice));
        forward_varlen(cfg, w, blas, s, d_ids, x, logits, k_pool, v_pool, d_tab,
                       d_seq, d_pos, d_out, n, n_out, block_size, num_blocks,
                       max_blocks, max_scan, 0);
    }

    std::vector<int> argmax(int n_out) {
        std::vector<int> r;
        pick.argmax_batched(logits, n_out, r, 0);
        return r;
    }

    /// Greedy generation for one sequence: one prefill pass, then q_len = 1.
    std::vector<int> generate(cublasHandle_t blas, const std::vector<int>& prompt,
                              int n_gen, int eos, int eot) {
        const int P = (int)prompt.size();
        std::vector<int> ids = prompt, seq(P, 0), pos(P);
        for (int i = 0; i < P; ++i) pos[i] = i;
        forward(blas, ids, seq, pos, {P - 1});
        std::vector<int> gen = {argmax(1)[0]};
        while ((int)gen.size() < n_gen && P + (int)gen.size() < max_len) {
            if (gen.back() == eos || gen.back() == eot) break;
            forward(blas, {gen.back()}, {0}, {P + (int)gen.size() - 1}, {0});
            gen.push_back(argmax(1)[0]);
        }
        return gen;
    }

    /// One pass over prompt + gen[0..n-2]; returns argmax at every position that
    /// predicts a generated token. pred[j] is this model's guess for gen[j].
    std::vector<int> teacher_force(cublasHandle_t blas, const std::vector<int>& prompt,
                                   const std::vector<int>& gen) {
        const int P = (int)prompt.size(), n = (int)gen.size();
        std::vector<int> ids = prompt;
        ids.insert(ids.end(), gen.begin(), gen.end() - 1);
        const int T = (int)ids.size();
        std::vector<int> seq(T, 0), pos(T), out(n);
        for (int i = 0; i < T; ++i) pos[i] = i;
        for (int j = 0; j < n; ++j) out[j] = P - 1 + j;
        forward(blas, ids, seq, pos, out);
        return argmax(n);
    }

    /// Min-of-trials time for one forward: n_seqs sequences, q_len tokens each,
    /// starting at `context`. all_rows: logits for every row (a verify pass)
    /// rather than the last row of each sequence (a decode step).
    double time_step(cublasHandle_t blas, int n_seqs, int q_len, int context,
                     bool all_rows, int trials = 5, int iters = 20) {
        std::vector<int> ids, seq, pos, out;
        for (int s_ = 0; s_ < n_seqs; ++s_)
            for (int i = 0; i < q_len; ++i) {
                ids.push_back(1000 + (int)ids.size() % 5000);
                seq.push_back(s_);
                pos.push_back(context + i);
                if (all_rows || i == q_len - 1) out.push_back((int)ids.size() - 1);
            }
        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
        for (int i = 0; i < 3; ++i) forward(blas, ids, seq, pos, out);  // warmup
        CK(cudaDeviceSynchronize());
        double best = 1e30;
        for (int t = 0; t < trials; ++t) {
            cudaEventRecord(a);
            for (int i = 0; i < iters; ++i) forward(blas, ids, seq, pos, out);
            cudaEventRecord(b); cudaEventSynchronize(b);
            float ms = 0; cudaEventElapsedTime(&ms, a, b);
            best = std::min(best, (double)ms / iters);
        }
        cudaEventDestroy(a); cudaEventDestroy(b);
        return best;
    }
};

/// Greedy speculative rounds over a teacher-forced match vector: each round
/// accepts the longest run of matches (at most K), then emits one bonus token
/// from the target. Returns {tokens, rounds}.
static std::pair<long, long> simulate(const std::vector<char>& match, int K) {
    long i = 0, rounds = 0;
    const long n = (long)match.size();
    while (i < n) {
        int a = 0;
        while (a < K && i + a < n && match[i + a]) ++a;
        i += a + 1;
        ++rounds;
    }
    return {n, rounds};
}

int main(int argc, char** argv) {
    std::string tdir = "weights_3b", tcfg = "onnx_3b/config.json";
    std::string ddir = "weights",    dcfg = "onnx/config.json";
    std::string tok_path = "onnx_3b/tokenizer.json", prompts_path = "prompts.txt";
    int n_prompts = 16, n_gen = 128, batch = 1, context = 256, k_max = 8;

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
        else if (a == "--batch"         && i+1<argc) batch = std::stoi(next());
        else if (a == "--context"       && i+1<argc) context = std::stoi(next());
        else if (a == "--k-max"         && i+1<argc) k_max = std::stoi(next());
    }

    Tokenizer tok; tok.load(tok_path);
    std::vector<std::vector<int>> prompts;
    {
        std::ifstream f(prompts_path);
        std::string line;
        while ((int)prompts.size() < n_prompts && std::getline(f, line))
            if (!line.empty()) prompts.push_back(tok.encode(line, true));
    }
    printf("%zu prompts, %d generated tokens each, batch %d, context %d\n\n",
           prompts.size(), n_gen, batch, context);

    const int max_len = std::max(1024, context + k_max + 2);
    const int max_out = std::max(n_gen, batch * (k_max + 1));
    cublasHandle_t blas; cublasCreate(&blas);

    // ── Phase 1: target — ground truth, self-consistency, timings ───────────
    std::vector<std::vector<int>> gens;
    long self_ok = 0, self_n = 0;
    double c_t = 0;
    std::vector<double> c_v(k_max + 1, 0);
    {
        Model m; m.load(tdir, tcfg, max_len, std::max(1, batch), max_out);
        for (const auto& p : prompts) {
            gens.push_back(m.generate(blas, p, n_gen, tok.eos_id(), tok.eot_id()));
            const auto pred = m.teacher_force(blas, p, gens.back());
            for (size_t j = 0; j < pred.size(); ++j) self_ok += pred[j] == gens.back()[j];
            self_n += (long)pred.size();
        }
        c_t = m.time_step(blas, batch, 1, context, false);
        for (int K = 1; K <= k_max; ++K)
            c_v[K] = m.time_step(blas, batch, K + 1, context, true);
        m.free_all();
    }

    // ── Phase 2: draft — teacher forcing, timing ────────────────────────────
    std::vector<char> match;            // concatenated across prompts
    std::vector<std::pair<int,int>> spans;
    double c_d = 0;
    {
        Model m; m.load(ddir, dcfg, max_len, std::max(1, batch), max_out);
        for (size_t i = 0; i < prompts.size(); ++i) {
            const auto pred = m.teacher_force(blas, prompts[i], gens[i]);
            spans.push_back({(int)match.size(), (int)pred.size()});
            for (size_t j = 0; j < pred.size(); ++j) match.push_back(pred[j] == gens[i][j]);
        }
        c_d = m.time_step(blas, batch, 1, context, false);
        m.free_all();
    }
    cublasDestroy(blas);

    long hits = 0;
    for (char c : match) hits += c;
    printf("target self-consistency (multi-row == incremental): %ld/%ld = %.2f%%\n",
           self_ok, self_n, 100.0 * self_ok / std::max(1L, self_n));
    printf("draft per-token match (alpha):                      %ld/%zu = %.3f\n\n",
           hits, match.size(), (double)hits / std::max<size_t>(1, match.size()));
    printf("step times, batch %d, context %d (min of 5):\n", batch, context);
    printf("  c_t target q=1   %.3f ms\n  c_d draft  q=1   %.3f ms   (c_d/c_t = %.3f)\n\n",
           c_t, c_d, c_d / c_t);

    printf("  K   c_v(K) ms  c_v/c_t   tau(K)   round ms   predicted speedup\n");
    for (int K = 1; K <= k_max; ++K) {
        long tok_sum = 0, rnd_sum = 0;
        for (auto [off, len] : spans) {      // rounds never cross prompt boundaries
            std::vector<char> m(match.begin() + off, match.begin() + off + len);
            auto [t, r] = simulate(m, K);
            tok_sum += t; rnd_sum += r;
        }
        const double tau = (double)tok_sum / std::max(1L, rnd_sum);
        const double round_ms = K * c_d + c_v[K];
        printf("  %-3d %9.3f  %7.3f  %7.3f  %9.3f   %6.3fx\n",
               K, c_v[K], c_v[K] / c_t, tau, round_ms, tau * c_t / round_ms);
    }
    printf("\nPrediction ignores host overhead, the draft catch-up row after a\n"
           "fully-accepted round, and rollback bookkeeping. Real <= predicted.\n");
    return 0;
}
