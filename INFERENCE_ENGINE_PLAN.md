# Mini Inference Server — Senior ML Inference Engineer Plan

**Goal:** Build a production-style LLM inference server ("mini-vLLM"), demonstrating
end-to-end ownership from kernel to serving layer.

**Timeline:** ~3–4 months | **Platform:** RTX 5070 Ti (WSL) for development; Jetson Orin
Nano as the edge-deployment story.

**Why this plan:** At 3 YOE + MS, the gap to senior isn't breadth — it's one deep,
coherent system whose every design decision you can defend. Continuous batching, paged
KV cache, and speculative decoding are what senior inference interviews probe; almost no
mid-level candidate has built them.

**Working mode (from Stage 2 onward):** coaching, not delivery. For each piece: design
and tradeoffs get explained, comprehension gets checked with questions, then *you* write
the code and it gets reviewed. Stage 1 was written for you, and the lesson learned was
that code you didn't write isn't code you can defend.

The split that emerged in practice: anything an interviewer would ask about — kernels,
scheduler policy, cache design — you write. Plumbing — CLI parsing, buffer allocation,
test harnesses, benchmark scripts — gets written for you.

**From Stage 5 on, predictions are a gate:** every run is preceded by a written
prediction (a number and one line of reasoning), and the result is compared against
it. Explaining a number after seeing it is not the same skill as predicting it.

**Ordering principle (learned the hard way at the vLLM comparison):** measure before
building. The comparison was planned for Stage 4 and run early instead; it replaced the
planned next feature (speculative decoding) with the one the numbers demanded (chunked
prefill), and gave a before-number to prove the after against. Every stage below now
ends with a measurement, and the next stage is chosen from it rather than from the plan.

---

## Baseline before this project

- Quantization study: GPTQ, AWQ, SmoothQuant, mixed precision, QAT fundamentals
- CUDA kernels: tiled matmul, flash attention (Triton), Nsight profiling
- Jetson Orin Nano: TRT Edge-LLM (28.2 tok/s decode, Qwen3-4B) vs llama.cpp (18.1 tok/s)
- Single-request TRT runtime in `../llama_jetson` — the seed for this project

---

## Stage 1 — Single-request engine ✅ COMPLETE

→ [benchmarks/STAGE1.md](benchmarks/STAGE1.md)

| | Delivered |
|---|---|
| S1.1 | One engine, two TRT optimization profiles: prefill (seq 1–2048, past=0), decode (seq=1, past 0–2047) |
| S1.2 | `KVCache` class — ping-pong buffers, zero copies per step |
| S1.3 | True batched prefill — whole prompt in one `enqueueV3` |
| S1.4 | GPU sampling (temperature / top-k / top-p), 1.7% overhead |
| S1.5 | Dimensions from `config.json`; device-resident attention mask |
| S1.6 | Verified vs HF: cosine 1.0000, 5/5 top-1; benchmark sweep with warmup |

**Results:** 41k tok/s prefill (512 tokens), 235 tok/s decode, 2048 context.

**Two bugs worth remembering:**

1. **KV must be contiguous.** TRT reads `past_key_values` as `[1, H, past, D]` with head
   *h* at `h·past·D` — a stride that changes every step. A fixed-capacity buffer strided
   by `max_seq` misreads every head after head 0. Symptom: fluent, wrong output. Fix:
   ping-pong, since `present` is already the layout `past` needs next.
2. **The pre-tokenizer is a regex with ordered alternatives**, not a categorizer.
   `1969` → `196`+`9` (digits capped at 3), `(n` → one chunk (rule 2 precedes rule 4).
   Getting this wrong shifted logits by KL 0.52 while still producing fluent text.

---

## Stage 2 — Continuous batching ← the core of the project

**Split into 2A and 2B deliberately.** Paged KV requires a custom attention kernel,
because TRT's attention demands a contiguous `past` and paging scatters KV across blocks.
Building the scheduler and that kernel simultaneously means every bug is ambiguous, with
no reference to compare against. So:

- **2A** — fixed slots, TRT's stock attention → a working multi-request server and a
  correctness baseline. Teaches continuous batching.
- **2B** — swap in paged KV + custom attention as a pure memory optimization, validated
  against 2A. Teaches paged attention.

### S2.0 — Batched engine spike ✅ COMPLETE

Rebuilt the engine with a batch range (min 1 / opt 4 / max 32) on every input, and
verified ragged batched decode with `lens = [500, 30, 200, 1]`.

**Verification design.** `isfinite()` is a worthless gate — it passes even when masking
leaks. Three invariance tests instead:

| Test | Question | Result |
|---|---|---|
| A | slot 1 batched vs alone | 0.234 |
| B | same batch, padding poisoned +1e4 vs −1e4 | **0.000000** |
| C | slot 1 at past=500 (padded) vs past=30 (not) | 0.234 |

B is the correctness gate and it is *exact*: flipping 470 positions of masked memory
changed nothing, so padding provably never reaches the output. A and C being identical
showed batch size doesn't affect numerics at all — only reduction length does. 0.234 on a
signal of max 14.66 is 1.6%: fp16 rounding from a 501-term softmax vs a 31-term one.

**Lesson: "numerically different" is not "incorrect."** A and C measure batch-composition
reproducibility, a real property of every batched server (vLLM included), not correctness.

**What the spike established for the design:**

- Per-row `position_ids` are mandatory — a slot's new token is at *its own* next position,
  not at `past`; passing `past` misrotates RoPE.
- The mask has three regions per row: `[1 × len_i][0 × (past − len_i)][1 × seq]`.
  TRT appends new KV at index `past`, so the new columns are always last.
- **The shared `past` is the fundamental limitation.** One `past = max(lens)` for the whole
  batch means every slot carries the longest sequence's length, and `past` grows by 1 each
  step for every slot regardless of its real length. One 2000-token request forces all 8
  slots to carry 2000 tokens of KV. *This is the motivation for 2B, derived rather than
  asserted.*

### Stage 2A — Continuous batching, fixed slots ✅ COMPLETE

→ [benchmarks/STAGE2.md](benchmarks/STAGE2.md)

| | Delivered |
|---|---|
| S2.1 | `BatchKVCache` — N slots with independent lengths, scatter kernel, per-row mask and positions |
| S2.2 | `Request` + `Scheduler` — waiting queue, running set, admission control, eviction, batch rebuilt every step so requests join mid-flight |
| S2.3 | Batched sampling — one segmented argmax for B sequences (`batched_pick.cuh`) |
| S2.4 | HTTP front-end — OpenAI-compatible SSE streaming, cancellation on disconnect |
| S2.5 | Verify + benchmark — 24/24 outputs identical to sequential |

**Results:** 947 tok/s peak at B=16 (5.2× single-request).

**Unresolved:** throughput collapses above B=16 (152 tok/s at B=32). Suspected cause
is TRT context memory sized for profile maxima (batch 32 × seq 2048) across two
execution contexts; `cudaMemGetInfo` reports 16302/16302 MB at every batch size on
WSL, making it useless for confirming this. Superseded by 2B rather than fixed.

### Stage 2B — Paged KV + custom decode path ✅ COMPLETE

→ [benchmarks/STAGE2B.md](benchmarks/STAGE2B.md)

| | Delivered |
|---|---|
| S2.6 | `BlockAllocator` — 16-token blocks, free list, block tables, no CUDA, 10 tests |
| S2.7 | Custom CUDA decode path instead of a TRT plugin: RMSNorm/RoPE/SiLU-mul/embedding kernels + 7 cuBLAS GEMMs per layer, verified layer-by-layer vs HF |
| S2.8 | Paged attention kernel — block-table indirection, 3 invariance tests, block-major restructure after profiling |
| S2.9 | `paged_runtime` — preemption (newest-first, recompute), KV budget sweep |

**Results:** 15,090 tok/s (**15.9×** Stage 2A's 947), TTFT p50 524 → 83 ms,
128 concurrent requests in 512 MB. KV utilization 86% vs 11%.

**Why a custom decode path rather than a TRT plugin (S2.7):** replacing attention
inside the engine needs ONNX graph surgery plus `IPluginV3` — a fiddly,
version-fragile API where a bug looks like a TRT error rather than a math error.
Writing the forward pass directly means everything is unit-testable and verifiable
against HF layer by layer. Cost: ~16% slower per step at equal batch (no kernel
fusion), which is what buys the ability to page.

**Findings worth carrying forward:**

- Paged attention is at **parity per call**, not faster — indirection costs about
  what the reduced scanning saves. The win is 8–40× memory. (Matches vLLM.)
- The naive Phase-3 loop was **2.3× slower** than the kernel it replaced, from
  per-token integer division; block-major iteration fixed it. Only the benchmark
  revealed this.
- **Raw throughput overstates under preemption** — at 64 MB, 32% of generated
  tokens were recompute. Goodput is the honest metric.
- Prefill and decode are the **same operation** in this path (no admission stall),
  at the cost of P steps for a P-token prompt. Chunked prefill is the fix.
  *This was recorded as a note here and cost 1,172 ms of TTFT in the vLLM
  comparison below — the gap between noticing a weakness and measuring it.*

## vLLM comparison ✅ COMPLETE — *ran early, and it rewrote the plan*

→ [benchmarks/VS_VLLM.md](benchmarks/VS_VLLM.md)

Originally scheduled as Stage 4, after speculative decoding. Doing it first was the
best decision in the project, because it replaced a guess about what to build next
with a measurement.

Same client, same GPU, same model, two prompt lengths:

| prompt | | tok/s | TTFT p50 | TPOT |
|---|---|---:|---:|---:|
| ~10 tok | llm_server | 5,755 | 61 ms | **4.57** |
| | vLLM | 5,633 | **22 ms** | 5.49 |
| 256 tok | llm_server | 1,294 | 1,172 ms | 6.64 |
| | vLLM | **4,182** | **123 ms** | **5.61** |

**Parity at short prompts, 3.2× behind at realistic ones.** Quoting the first row
alone would have been dishonest. TTFT proved to be exactly `prompt_len × step_time`
— predicted 256 × 4.6 ms = 1.18 s, measured 1.172 s — which identified the missing
feature precisely rather than by intuition.

TPOT also *crosses over*: ahead at short context (framework dispatch overhead
dominates, and a C++ loop beats a Python scheduler) and behind at long context
(the attention kernel dominates, and scalar fp16 loads lose to FlashAttention).
A win in one regime does not survive into the other.

**This reordered the roadmap.** Speculative decoding was planned next; the
measurement said chunked prefill, and said so with a number. Speculative decoding
also targets TPOT, which at 256-token prompts was the *smallest* gap (0.84×) while
TTFT sat at 0.10×.

## Stage 3 — Chunked prefill ✅ COMPLETE

→ [benchmarks/STAGE3.md](benchmarks/STAGE3.md)

| | Delivered |
|---|---|
| S3.1 | `paged_attention_varlen.cuh` — attention for `q_len >= 1`. Mask is a loop bound (`n_keys = q_pos[g] + 1`), not a branch; `lens[]` disappears entirely. Decode is the degenerate case. |
| S3.2 | `BlockAllocator::append_tokens` — multi-token growth, all-or-nothing, plus `fits_ever` to reject prompts that can never fit |
| S3.3 | `chunk_scheduler.h` — decodes first, prefill FCFS, per-sequence cap, block-aware. No CUDA, 23 tests. |
| S3.4 | `varlen_layer.h` + `serve_chunked.cpp` — mixed-batch forward pass and engine loop |
| S3.5 | Chunk-size invariance + configuration sweep |

**Results at 256-token prompts:** TTFT 1,172 → **240 ms (4.9×)**, throughput
1,294 → **3,261 tok/s (2.5×)**, TPOT 6.64 → 5.97 ms. Tuned for latency,
**96.8 ms TTFT p50 — better than vLLM's 123 ms.**

**Why varlen came first, and what it also unlocks.** Speculative decoding verifies
K draft tokens in one forward pass, which *is* `q_len = K` with a causal mask among
the drafts — the same kernel chunked prefill needs. Building it for prefill leaves
speculative decoding as acceptance sampling plus cache rollback, not a second kernel
project. That's why the order changed rather than the scope.

**Findings worth carrying forward:**

- **Chunk-size invariance is the cheap correctness gate.** The same prompt served
  with `--max-chunk 512` (one pass) and `--max-chunk 2` (three passes) must produce
  token-identical greedy output. No reference model needed; catches any off-by-one
  in `q_pos` or the mask at a chunk boundary.
- **Scheduler policy redistributes latency but not throughput.** Across five
  configurations throughput varied 3,105–3,261 tok/s — a 5% spread — while TTFT p50
  ranged 97–246 ms. This localizes the residual gap to vLLM as *kernel* work and
  makes further scheduler tuning provably pointless.
- **A bigger token budget made p50 TTFT worse** (123 → 240 ms) while tightening the
  tail. Admitting ~8 prompts per step means they all prefill together and finish
  together; a smaller budget staggers completions. The processor-sharing result from
  the FCFS analysis, reappearing across requests instead of within a step.
- **Budget and admission rate are not independent knobs.** Capping admissions gave
  the best p50 (96.8 ms) and the *worst* TPOT (8.26 ms): staggering doesn't reduce
  prefill interference, it stretches it over more steps. The real axis is
  concentrated vs spread interference, and it's an SLO choice.
- **`max_chunk` never bound** at a 2048 budget (240 vs 246 ms). Correct in
  principle, irrelevant at this operating point — worth knowing which knobs are
  live before tuning them.

## Stage 4 — Attention kernel: profiled, scoped, deliberately not built

→ [benchmarks/STAGE4_FINDINGS.md](benchmarks/STAGE4_FINDINGS.md)

Investigated as the measurement indicated, then stopped on purpose. What the
profiling established:

- Attention is **47.4%** of a decode step (gemm 36.9%, elementwise 7.5%, lm_head 8.1%)
- **11.6% load efficiency** — 3.7 useful bytes per 32-byte sector, an 8.8× overfetch
  verified by hand against the ideal sector count
- **Not** bandwidth-bound (15–19% of peak on every shape) and **not** compute-bound
  (20% SM throughput). Latency-bound, stalling on requests it mostly wastes.
- Cause: Phase 1 assigns one thread per token, so adjacent lanes read addresses
  128 bytes apart. The loop *looks* strided and is still uncoalesced — the stride is
  over the wrong axis.
- Amdahl ceiling: 2× faster attention → 31% faster step → ~4,200 tok/s, which is
  vLLM's number.

The fix (cooperative coalesced load of a K tile into shared memory, then
thread-per-token math out of shared) is specified in the findings doc. Left
unimplemented: the remaining work is implementation grind, and this project exists
to understand the concepts rather than to win a benchmark. The concepts — profile
before optimizing, compute the ceiling first, and why tiling *forces* online softmax
— are recorded.

Two methodology rules came out of it and now apply to everything: **report min of N
trials, not a mean** (the same binary measured 113 µs idle and 179 µs with a game
open; minima reproduce across sessions to ~1%, medians drift 20–30% from thermal
droop within one run), and **verify the binary changed before explaining a result**
(an entire mechanism was theorized for a slowdown that turned out to be an
unrebuilt kernel plus a background game).

## Stage 5 — Speculative decoding ← in progress

A low-concurrency latency feature, not a throughput feature: it spends idle
arithmetic to buy single-stream TPOT. Target **Llama-3.2-3B**, draft **Llama-3.2-1B**
(the smallest model sharing LLaMA-3's 128,256-token vocabulary). Greedy only —
`serve_chunked` has no sampling path yet; the probabilistic acceptance rule moves to
Stage 11 with sampling.

| | Status | Delivered |
|---|---|---|
| S5.0 | ✅ | `tools/spec_measure.cu` — predicts the speedup *before* building anything: `c_t`, `c_d`, `c_v(K)` timed; `tau(K)` from teacher forcing (exact for greedy) |
| S5.1 | ✅ | `BlockAllocator::truncate()` — the first operation that shrinks a sequence. Validates before mutating; 5 isolated tests, each naming the bug it catches |
| S5.2 | ✅ | `tools/spec_offline.cu` — greedy spec loop at batch 1: two KV pools, one allocator, one block table; checked against plain 3B greedy |
| S5.3a | ✅ | `ChunkScheduler::Config::decode_tokens` — decoders take K+1 tokens; block-boundary test (`prefill got 64, not 384`) |
| S5.3b | ✅ | `serve_chunked` plumbing: `--draft/--draft-config/--spec-k`, shared table, per-token logits, `run_pass()` |
| S5.3c | ⏳ | the batched speculative step inside the server: draft pass 1 (prefill chunks + catch-up + pending), draft passes 2..K, one target pass (verify rows + prefill chunks), per-sequence accept and rollback |
| S5.4 | ☐ | benchmark at concurrency **1 and 4** against the spec-off baselines below |

### S5.0 — the prediction

1B → 3B, batch 1, context 256, min of 5:

| | batch 1 | batch 4 |
|---|---:|---:|
| `c_t` (3B, q=1) | 11.46 ms | 11.86 ms |
| `c_d` (1B, q=1) | 5.36 ms | 4.83 ms |
| `c_d / c_t` | **0.467** | 0.407 |
| `c_v(2) / c_t` | 1.02 | 1.07 |
| α (teacher-forced match rate) | **0.808** | 0.808 |
| `tau(2)` | 2.465 | 2.465 |
| best predicted speedup | **1.26× at K=2** | 1.31× at K=2 |

`speedup(K) = tau(K) · c_t / (K · c_d + c_v(K))`. **The draft is too expensive:**
α is good, but at `c ≈ 0.47` the ceiling is ~1.3×; 2× needs `c ≲ 0.1`. The 1B is
also *less* bandwidth-efficient than the 3B (~460 vs ~560 GB/s, 51% vs 62% of peak),
which is why `c` exceeds the 0.39 bytes ratio. Kept anyway — the mechanism is the
point, and a measured, explained 1.27× is a valid result. Ways to lower `c`, recorded
for later: INT4 draft + vocabulary-trimmed draft lm_head (~1.8–2.0× predicted);
INT8 8B target with INT8 1B draft (~1.65×, Stage 8); EAGLE-style heads (need training).

### S5.2 — the measurement

16 prompts × 128 tokens, K = 2, against plain 3B greedy decoding:

| | predicted | measured |
|---|---:|---:|
| decode speedup | 1.26× | **1.272×** (11.22 → 8.82 ms/token) |
| tokens per round | 2.465 | 2.444 |
| accepted a = 0 / 1 / 2 | 19.2 / 15.5 / 65.3 % | 19.1 / 17.3 / 63.5 % |
| TTFT increase (both models prefill) | ~40% | **+42%** |
| prompts with a near-tie divergence | 1 − 0.9941¹²⁸ = 53% | 8 / 16 |

**Correctness: 16/16 pass, 0 FAIL** — 8 token-identical, 8 diverging only at fp16
near-ties (top-2 gap 0.016–0.047).

**Findings worth carrying forward:**

- **Exact in theory, not bit-exact in practice.** A K+1-row verify GEMM and a 1-row
  decode GEMM accumulate in different orders; target self-consistency is 99.41% per
  token. The correctness test is therefore "identical, or diverges at a near-tie"
  (top-2 logit gap below a threshold), and the near-tie rate is *predicted* from the
  per-token flip rate. A broken implementation diverges at positions 1–3 with gaps up
  to 8.6 — the two failure signatures are unmistakable.
- **τ alone decides per-prompt speedup.** Per-prompt speedup ranged 1.08–1.36×, but
  `tau × ms/token` (ms per round) was constant at ~21.7 ms for every prompt: the cost
  side is fixed, acceptance varies with how predictable the text is. Near-tie prompts
  cluster at low τ — uncertain text causes both.
- **Rollback is bookkeeping.** Rejected KV stays in the pool, never read (the mask is
  `n_keys = q_pos + 1`) and overwritten next round. `truncate()` only returns whole
  blocks.
- **The draft lags the target by at most one position.** After a fully accepted round
  the draft never fed `d_K`; the next round's first draft pass feeds `[d_K, bonus]`
  (q_len 2). The varlen kernel makes catch-up free — no special case.
- **Invariant checks verify counts; the oracle verifies content.** The first
  implementation emitted the rejected draft `d[a]` instead of the target's `g[a]`
  (and read out of bounds when `a == K`). Every count was right; the oracle caught it
  on the first run.
- **Memory:** one table means each block costs target + draft bytes — 868 blocks vs
  1,116 at 2 GB, **22% less KV capacity**.

### Spec-off 3B baselines for S5.4

`serve_chunked`, 3B, ~203-token prompts:

| | concurrency 1 | concurrency 32 |
|---|---:|---:|
| throughput | 81.8 tok/s | 528 tok/s |
| TTFT p50 | **62.7 ms** | 2,984 ms |
| TPOT p50 | **11.82 ms** | 12.85 ms |

- **TTFT at concurrency 32 is ~98% queueing.** Same prefill, same GPU: 62.7 ms alone,
  2,984 ms behind 31 others with a 512-token budget per step.
- **The server adds ~0.7 ms per forward pass** over the offline harness (11.82 vs
  ~11.1 ms). Speculation runs ~1.23 passes per token, so this overhead costs it slightly
  more than it costs plain decoding.
- **The 3B is prefill-bound at high concurrency, and the cause is the attention
  kernel.** TPOT scales with weight bytes (2.2× the 1B), but TTFT is 12× worse:
  28 layers, `head_dim` 128 through the same 11.6%-efficient access pattern. At
  concurrency 32 speculation would make this worse (a second prefill, ~+40%), which
  is why S5.4 measures at 1 and 4. **This raised Stage 6's priority** — a
  measure-first reordering, like the vLLM comparison before it.

## Roadmap after Stage 5 — slimmed (2026-11-18)

Reordered for the goal (senior inference roles), not for feature count: depth in a
few things that can be defended completely, plus fluent concepts for the rest.
Multi-GPU moved up because "no distributed experience" is a bigger gap for
NVIDIA/CoreWeave-type roles than any single-GPU feature. CUDA graphs and prefix
caching shrank to short experiments: their interview value is mostly conceptual.

| # | Stage | Size | Why it is here |
|---|---|---|---|
| 6 | Paged varlen attention in CUDA (FA-2 → serving) | ~1 week | most-asked kernel topic; the measured 3B bottleneck |
| 7 | Tensor parallelism, 2 rented GPUs | ~1 week | closes the multi-GPU gap |
| 8 | Weight-only quantized GEMV (W8A16 / W4A16) | ~1 week | needed for the edge comparison; lowers spec `c` |
| 9 | Jetson Orin Nano 8 GB vs llama.cpp | days (reuses `llama_jetson`) | edge story, bandwidth-bound regime |
| 10 | Short experiments: CUDA graphs, prefix caching | 1–2 + 3–4 days | concept plus one number each |
| 11 | Production completeness (sampling, chat, RoPE scaling) | only if a role needs it | |
| 12 | Write-up + mock system-design interviews | ongoing | turning the work into offers |

## Stage 6 — Paged varlen attention in CUDA

**Prior work, not redone:** `../Triton/flash-attention-triton` is a FlashAttention-2
forward pass — online softmax with fp32 running max/sum and a rescaled accumulator,
`exp2` with the scale folded into Q, tensor-core `tl.dot`, causal early exit,
autotuned tiles. The algorithm is known. What that kernel does *not* handle is
exactly what a serving engine needs, and that is the scope here:

| | Triton FA-2 | this stage |
|---|---|---|
| layout | contiguous `(B, H, N, D)` | **paged**: K/V tiles loaded through `block_table`; a 16-token block is a natural tile |
| batch | one N for all sequences | **varlen**: mixed prefill chunks + decode tokens via `q_seq` / `q_pos`; causal mask from `q_pos` per tile |
| heads | Hq == Hkv | **GQA**: the Q heads sharing a KV head load its tile once (ratio 3 on the 3B, 4 on the 1B) |
| decode, q_len = 1 | one query row idles most of a tile | **split-K (FlashDecoding)**: keys split across thread blocks, partial `(m, l, acc)` merged — online softmax across blocks instead of across iterations |
| engine | Python | CUDA, called from `varlen_layer.h` |

Fixes the Stage 4 finding: Phase 1 puts one thread per token, so adjacent lanes read
128 bytes apart — **11.6% load efficiency**.

- **Gates:** ncu load efficiency 11.6% → 90%+; matches the current kernel on
  `test_paged_attention_varlen` (shuffled blocks, ragged batch, prefill vs N decodes);
  `spec_offline` and chunk-size invariance still pass.
- **Measure:** `bench_varlen_attention` per shape; 3B TTFT at concurrency 32
  (2,984 ms now); re-run the vLLM comparison.
- **Predict first:** the Stage 4 Amdahl ceiling (2× attention → ~31% faster step on the
  1B). Attention's share is larger on the 3B in prefill — compute the ceiling there too.

## Stage 7 — Tensor parallelism (2 rented GPUs)

Megatron-style splits: column-parallel QKV / gate / up, row-parallel o_proj / down,
one NCCL all-reduce after attention and one after the MLP; KV heads sharded across
ranks (8 KV heads → 4 per GPU). One process per GPU, the scheduler on rank 0
broadcasting each step's batch description.

- **Predict first:** per-layer all-reduce bytes = 2 × tokens × hidden × 2 B; against the
  interconnect bandwidth (PCIe vs NVLink), when does communication dominate a decode
  step?
- **Measure:** TPOT and throughput at TP=1 vs TP=2 on the same rented machine; the
  fraction of the step spent in NCCL (nsys).
- Rent for a weekend; record the instance type so the numbers can be reproduced.

## Stage 8 — Weight-only quantized GEMV (W8A16 / W4A16)

Not quantization research — a kernel: unpack, dequantize in registers, dot product,
for 1 to K+1 rows. Prefill dequantizes one layer into fp16 scratch and uses cuBLAS
(compute-bound, so the extra traffic is fine). Per-output-channel (W8) or group-wise
(W4, g=128) scales. Verification changes from cosine 1.0 to top-1 agreement and
perplexity delta.

Then re-run Stage 5's unchanged speculative code with an INT4 draft (and optionally
an INT8 Llama-3.1-8B target): a controlled experiment where the measured speedup
should track `c` — predicted ~1.26× (now) → ~1.65–1.8×.

## Stage 9 — Edge: Jetson Orin Nano 8 GB vs llama.cpp

Same engine where the bottleneck is bytes per token and nothing else
(102 GB/s vs 896 GB/s, concurrency 1, 8 GB unified memory). Reuses
`../llama_jetson` (Orin setup, reference numbers: TRT Edge-LLM 28.2 vs llama.cpp
18.1 tok/s on Qwen3-4B) and `../llama_benchmark` (llama.cpp benchmark scripts).

- **Port:** `sm_87` (already in `CMakeLists.txt`), no TensorRT needed; alias the tied
  lm_head to the embedding table (saves 0.5 GB on the 1B); KV budget 256–512 MB.
- **Fits:** 1B fp16 yes; 3B only as INT4 (Stage 8); 1B + 3B speculation does not fit in fp16.
- **Two comparisons, both reported:** (1) fp16 vs llama.cpp fp16 — runtime quality at
  equal bytes; (2) W4A16 vs llama.cpp Q4_K_M — best vs best at equal bits.
- **Method:** pin `nvpmodel` MAXN SUPER + `jetson_clocks`, watch thermals, min of N.
- **Predict first:** ceiling = bandwidth ÷ bytes per token; then at the 5070 Ti's
  measured efficiency (~51% for the 1B).

## Stage 10 — Short experiments

- **CUDA graphs (1–2 days):** capture one fixed-shape decode step (batch 1, in
  `spec_offline`) and measure the drop in `c_d` — launch overhead hurts the small
  draft most, so graphs also raise the speculative speedup. Compare with
  `../llama_benchmark`'s llama.cpp graphs-on/off numbers. Write up why production
  engines capture per batch-size bucket and pad.
- **Prefix caching, allocator only (3–4 days):** hash full blocks by
  (parent hash, tokens), refcount shared blocks, copy-on-write on the last partial
  block, LRU eviction of unreferenced cached blocks — in `BlockAllocator` with host
  tests, no server integration. Write up the design against vLLM's automatic prefix
  caching and SGLang's RadixAttention.

## Stage 11 — Production completeness (only if a target role needs it)

Sampling path in the server (segmented top-k/top-p for B rows), then the speculative
acceptance rule for sampling — accept with `min(1, p_target/p_draft)`, resample from
`max(0, p_target − p_draft)`, provably the same distribution as plain sampling.
`/v1/chat/completions` with the chat template. LLaMA-3 RoPE frequency scaling
(`rope_type: llama3`), which short-context verification cannot catch.

## Stage 12 — Write-up

- One deep technical post: the scheduler, the paged cache, speculative decoding's
  cost model, and the measure-then-build ordering that reshaped the plan three times.
- Repo polish: architecture diagram, consolidated benchmark tables, datacenter vs edge.
- Mock system-design interviews defending every decision above.

---

## Open questions

- Stage 2A throughput collapse above B=16 — never root-caused; superseded by 2B.
- Prefill throughput peaked at 512 tokens (41k tok/s) and fell to 10k at 1900 in the
  TRT path. The same O(n²) question applies to `paged_attention_varlen`; Stage 6
  should measure it.
- TTFT p95 (253 ms) remained 1.6× vLLM's (161 ms) in every configuration tried.
  The 3B concurrency-1 vs 32 runs show TTFT under load is dominated by queueing; a
  concurrency sweep would confirm whether that explains the p95 gap too.
- The 1B draft step is *faster* at batch 4 (4.83 ms) than at batch 1 (5.36 ms).
  Hypothesis: cuBLAS chooses a poor kernel for `n = 1`. Confirm with nsys kernel names.
- The 1B reaches only ~51% of peak bandwidth at batch 1 vs ~62% for the 3B — launch
  overhead per layer is a larger share of a small model's step (Stage 10, CUDA graphs).
- Every benchmark is a single run. No repeats, no variance, no error bars — a 1%
  difference (1.272× vs 1.26× predicted) is inside the noise.
- Only ~200–256-token prompts were swept. The gap to vLLM should widen at longer contexts.

## Explicitly deprioritized

- Quantization *research* (GPTQ/AWQ variants, QAT) — enough depth already
  (`../Quantization`). The quantized GEMV kernel (Stage 8) is in scope because the
  edge comparison needs it.
- Re-deriving FlashAttention — done in Triton (`../Triton/flash-attention-triton`);
  Stage 6 builds only the serving-specific parts.
- Full CUDA-graph integration and full prefix-caching integration in the server —
  reduced to Stage 10 experiments.
- MLIR/TVM (different career track)
- k8s (learn on the job if a role requires it)
- Full Unicode pre-tokenizer (needs PCRE2/ICU; ASCII approximation verified 8/8)
- Trained draft heads (EAGLE/Medusa) — the right production answer, but a training
  project, not an inference one
