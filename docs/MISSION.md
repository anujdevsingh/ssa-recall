---
type: analysis
created: 2026-07-06
query: "Write the full five-year mission plan: from the current SSA-recall study to the first India-based sub-quadratic LLM with 10-12M context, reasoning and coding capability — with measurable expected results per stage so future progress can be compared against it."
sources-consulted: [subq-subquadratic-startup-deep-dive, ssa-recall-research-6month-plan, zoology-recall, subquadratic-architectures, associative-recall, hybrid-architectures]
output-format: prose
tags: [mission, roadmap, subquadratic, long-context, ssa-recall, career, india, five-year-plan]
---

# The Five-Year Mission: A Verified Sub-Quadratic Long-Context LLM, Built in India

> **Mission statement.** Make reading 10M+ tokens ~100× cheaper without losing recall — proven in public at every stage — and grow that result, stage by stage, into the first India-based sub-quadratic LLM with multi-million-token context, credible reasoning, and coding ability, that real people can use.

This document is the north star. Every strategic decision gets checked against it. It is written the way a senior researcher writes a program grant: each stage has an objective, a duration, prerequisites, concrete workstreams, deliverables, **quantified expected results** (so we can compare reality against the plan later), and explicit pivot/kill criteria. The stages are strictly sequential — each one *funds* the next with evidence, credibility, or resources. Skipping a stage is the defined failure mode of this mission.

**Canonical copies:** this page (KnowledgeBank wiki) and `docs/MISSION.md` in the [ssa-recall repo](https://github.com/anujdevsingh/ssa-recall) (git-versioned, never lost).

---

## Operating principles (apply to every stage)

1. **One-axis bet.** We compete on *context length × cost × recall*, never on general capability head-to-head with frontier labs. Reasoning/coding enter only when a team exists (Stage 3+).
2. **Honesty is the moat.** Our differentiation vs SubQ and every closed lab is verifiability: public repo, public logs, reproducible-for-dollars experiments, caveats printed in the paper. A modest airtight claim always beats a flashy shaky one.
3. **Evidence before scale.** No stage starts until the previous stage's exit criteria are met. The plan bends (timelines slip, methods change); the gates do not.
4. **Altitude discipline.** At each stage, do only the work that stage can support. Solo = experiments and one novel component. Team-scale work (post-training, serving, data pipelines) waits for the team.
5. **Everything ships.** Every stage ends with a public artifact (paper, repo, model, demo) even if the scientific result is negative. Negative results on recall-at-length are publishable and still advance the mission.
6. **Compare against this document.** At each stage boundary, write a review entry: expected vs achieved, and update this plan *explicitly* (logged in wiki/log.md) rather than silently drifting.

---

## Stage 0 — The proof (now → ~2026-10) — IN PROGRESS

**Objective.** Complete and publish the small-scale study: *content-based sparse attention with a cheap hierarchical selector preserves associative recall sub-quadratically.*

**Status when this plan was written (2026-07-06):**
- M1 done — reproduced the Zoology MQAR recall frontier (attention ~0.99 vs gated-conv ~0.10-0.13).
- M2a done — NSA rides the attention ceiling at seq128 (0.99+ recall), confirming half of SubQ's thesis; NSA's selector shown to be quadratic ("complexity moved, not removed").
- M2b headline in hand — at L512, our hierarchical selector (`hier`) hits **0.983 recall** vs attention 0.988 vs NSA 0.978, at **1.8× fewer scored-pairs/query**, converging 2-4× faster (epoch 14 vs 32/52).
- L2048 gather-OOM fix committed (`bfeda39`); the L2048 sweep is the immediate next action.

**Remaining workstreams.**
1. **L2048 sweep** — attention/nsa/hier at lr 3.2e-4, 100 epochs, batch 16, subprocess-per-config, on Kaggle. This is the 3.8×-cheaper point; two points make the cost-scaling trend.
2. (Stretch) **L8192 point** if T4 memory allows — three points make it undeniable.
3. **Parse + figures** — m2b RESULTS.md, the recall-vs-length frontier figure, the cost-vs-length figure.
4. **Write-up** — arXiv preprint + efficient-ML/long-context workshop submission (NeurIPS/ICLR workshop cycle). Framing: *measurement + one new cheap selector*, with all caveats (LR sensitivity, scored-pairs metric not wall-clock, gather+mask prototype) inside the paper.
5. **Public launch** — flip repo public simultaneously with the write-up; blog post; X/HN post. Repo README must let a stranger regenerate every figure for <$10 of compute.

**Deliverables.** arXiv preprint · workshop submission · public GitHub repo · blog post.

**Expected results (the comparison baseline for the future).**
- hier recall at L2048 within **0.02 of full attention** (predicted ≥0.96) while NSA is ≤ hier or OOM/slow. If hier *fails* at L2048, that is itself the finding — publish the failure length and its analysis.
- Cost curve: scored-pairs/query advantage grows with length (1.8× @512 → ~3.8× @2048 → ~8× @8192 if run).
- Workshop acceptance (target ≥1 of 2 submissions) OR ≥50 GitHub stars / meaningful researcher engagement as the fallback credibility signal.

**Exit criteria (gate to Stage 1).** Paper on arXiv + repo public + L2048 result (positive or negative) in the write-up.

**Pivot/kill criteria.** None — this stage completes regardless. Only the *framing* changes with the L2048 outcome.

---

## Stage 1 — The real-LM demonstration (~2026-10 → ~2027-08)

**Objective.** Show the selector survives contact with a *real language model at real lengths*: a ~340M-1.3B hybrid LM (mostly hier layers + a few full-attention layers) pretrained at short context and length-extended to **1-4M tokens**, with recall-at-depth measured at every extension stage. This is where the "first India-based long-context sub-quadratic LM" flag gets planted.

**Prerequisites.** Stage 0 exit criteria; GPU grant (IndiaAI Mission compute program / academic cloud credits / sponsor) — the Stage 0 paper *is* the grant application's core evidence.

**Workstreams.**
1. **GPU access (month 1-2).** Apply to IndiaAI Mission compute, Google TRC, Modal/Lambda research credits — in parallel. Budget target: the equivalent of ~2-8 × A100/H100 for ~4-8 weeks total across the stage.
2. **Stack, not scratch (month 1-3).** Port `HierSparseAttention` into a flash-linear-attention-compatible layer; adopt a proven lean pretraining harness (FLA training stack / litgpt-class trainer). Our novel code stays ~one file; everything else is reused, exactly as in Stage 0.
3. **Twin pretrain (month 3-5).** Pretrain two matched models on a public corpus (FineWeb-class) at 4-32K context: (a) the hier-hybrid, (b) a full-attention twin of identical size/data/schedule. Sizes: start 340M; scale to 1.3B only after 340M validates.
4. **Length extension (month 5-8).** Staged fine-tuning 32K → 128K → 1M → 4M with synthetic + natural long-context curricula; at each stage measure needle/MQAR-style recall-at-depth and perplexity-at-length against the twin.
5. **Triton kernel v1 (parallel track, month 3-8).** Fused kernel for the hierarchical selector so Stage 1 cost claims are **wall-clock**, not scored-pairs. Escalation ladder: Triton → ThunderKittens → CUDA/CUTLASS, only as far as needed.

**Deliverables.** Second paper (main-track attempt; workshop fallback) · open weights for both twins · kernel in the public repo · technical blog.

**Expected results (comparison baseline).**
- Hybrid matches the attention twin's perplexity within **~3%** at 4-32K (hybrids routinely do — Kimi Linear precedent).
- Recall-at-depth at 1M context: hybrid retrieves needles at **≥90%** of the attention twin's rate at equal depth. (The attention twin itself may not reach 1M — then compare against published full-attention baselines.)
- Wall-clock: hier kernel prefill **≥3× faster** than FlashAttention at 1M+ on the grant hardware.
- One accepted paper; measurable community adoption (external users of the kernel or weights).

**Exit criteria (gate to Stage 2).** Open weights at ≥1M context with recall-at-depth evidence + wall-clock kernel numbers published.

**Pivot criteria.** If recall collapses at length in a real LM (and debugging attributes it to the selector, not the recipe): pivot the research to *why* — that diagnosis is a strong paper, and the mission re-plans around the fix (e.g., hybrid ratio, SR-TTT-style exact-memory residual). If no GPU grant lands by month 4: shrink to 160M/1M-context on rented spot instances (~$2-5K) and continue; the mission slows, it doesn't stop.

---

## Stage 2 — The headline number (~2027-08 → ~2028-06)

**Objective.** Push the *same* model family's verified context to **8-12M tokens** — matching SubQ's headline as **the first independently verifiable demonstration at that length** — and build the eval harness that makes "verified" mean something.

**Prerequisites.** Stage 1 exit criteria; renewed/larger compute grant (Stage 1 paper + weights are the application).

**Workstreams.**
1. **Eval first.** Public long-context evals stop near 128K-1M (RULER-class). Build and release an open recall/reasoning-at-depth harness that runs to 12M. This is a first-class contribution: whoever defines the measurement owns the conversation — and it is precisely what the field demanded of SubQ and never got.
2. **Sequence-parallel training.** Adopt (not build) sequence/context parallelism from an existing stack (ring-attention-style / DeepSpeed-Ulysses-class) adapted to the hier selector's block structure.
3. **Extension to 8-12M.** Staged 4M → 8M → 12M on the 1.3B (or a fresh ~3B if compute allows), with the eval harness run at every stage, all logs public.
4. **Kernel v2.** Multi-GPU-aware selector kernel; wall-clock scaling curve to 12M published.

**Deliverables.** The eval harness (standalone repo) · 8-12M verified model + report · third paper.

**Expected results (comparison baseline).**
- Verified needle/recall-at-depth at **8M**: ≥80% retrieval at uniform depths (any credible number at 8M+ is a first — the *verified* qualifier is the product).
- End-to-end cost per 10M-token read: **≥50× cheaper** than dense-attention extrapolation on the same hardware.
- Eval harness adopted by ≥2 external groups (papers or repos citing/using it).

**Exit criteria (gate to Stage 3).** Public, reproducible 8M+ demonstration + the eval harness in external use.

**Pivot criteria.** If 8M+ is compute-infeasible on grant hardware: stop at the longest verified length achieved (even 2-4M verified beats 12M claimed) and proceed to Stage 3 on the strength of "longest independently verified context in the world."

---

## Stage 3 — The team (~2028 → ~2029)

**Objective.** Convert three papers, open weights, a used eval standard, and a world-first verified demonstration into **resources**: a funded lab/startup or a major institutional program — because Stages 4's work (reasoning, coding, serving) is structurally team-work. DeepSeek had High-Flyer's ~10K A100s before it shipped anything; Moonshot raised $1B+ in year one. The lesson taken: *iterate in public for years* (copyable) on top of *serious resources* (must be acquired — this stage acquires them).

**Workstreams.** Founding story + pitch built directly from Stages 0-2 artifacts · funding routes in parallel: IndiaAI Mission institutional partnership, Indian VC (sovereign-AI thesis), global efficient-inference VC, or a lab partnership · first hires: post-training lead, data-pipeline lead, infra/serving lead, kernels engineer (4-6 people) · governance: keep the open-verification identity contractually protected — it is the brand.

**Expected results.** Funding/compute commitment sufficient for a 3-7B-class long-context model program (order $2-10M or equivalent sovereign compute) · founding team of ≥4 senior people · the open-verification identity intact.

**Exit criteria.** Signed compute + team in seat.

**Pivot criteria.** If funding doesn't land in ~12 months: join forces instead — bring the stack to an existing Indian lab (Sarvam-class, IIT consortium, IndiaAI model program) as the long-context lead. The mission survives inside a bigger vessel; ego is not a stage gate.

---

## Stage 4 — The public model (~2029 → ~2031)

**Objective.** Ship the model the mission promises: **a 3-7B-class sub-quadratic LLM, 10-12M verified context, competent reasoning and coding, served cheaply, used by the public** — the first of its kind from India.

**Workstreams (team-scale).** Base pretrain with the mature hybrid architecture · post-training program: instruction tuning, RL for reasoning (R1-style recipes will be commodity by then), code-data pipeline · long-context serving: paged/hierarchical KV management on the selector's block structure — the selector's sparsity is a *serving* advantage, not just a training one · product surface: API + one flagship Indian use case executed deeply (courts/legislation, healthcare records, or govt-scheme corpus) in partnership with a domain institution · trust: publish the eval harness results for every release; independent red-team.

**Expected results (comparison baseline).**
- Reasoning/coding within the **top open-weights tier of its size class** at release (not frontier parity — size-class parity).
- Long-context: **10M+ verified** on the public harness; ≥95% needle at 1M, graceful degradation curve published to 10M.
- Cost: 10M-token session priced **~100× below** dense-attention incumbents' long-context pricing.
- Adoption: one flagship deployment with a named Indian institution + public API with real third-party usage.

**Exit criteria.** The mission statement, satisfied: cheap, long, honest, public, Indian-built.

---

## The scoreboard (fill in at every stage boundary)

| Stage | Planned exit | Key expected number | Actual result | Date closed |
|---|---|---|---|---|
| 0 — Proof | ~2026-10 | hier ≥0.96 recall @L2048; paper + public repo | *L512: hier 0.983 vs nsa 0.978, 1.8× cheaper (done)* | — |
| 1 — Real LM | ~2027-08 | ≥90% relative recall-at-depth @1M; ≥3× wall-clock @1M+ | | — |
| 2 — Headline | ~2028-06 | ≥80% recall @8M verified; harness in external use | | — |
| 3 — Team | ~2029 | Funding + 4-6 senior hires | | — |
| 4 — Public model | ~2031 | 10M+ verified, size-class-tier reasoning, ~100× cost | | — |

---

## Risk register (top 6, with mitigations)

1. **Recall collapses at length in real LMs** (the scientific risk). *Mitigation:* it surfaces in Stage 1 where it is cheap; the diagnosis is a paper; fixes exist to try (hybrid ratio, exact-memory residual à la [[sr-ttt]]). This risk is the *research*, not a failure of it.
2. **SubQ (or a frontier lab) publishes and open-sources first.** *Mitigation:* our identity is *verification + openness*, which their success amplifies rather than destroys; our eval harness and independent numbers stay valuable in a world where they win. Watch item, not a kill.
3. **No GPU grant.** *Mitigation:* every stage has a shrunk-scope fallback (Stage 1: 160M/1M on ~$2-5K spot). Slower, not dead.
4. **Solo-founder burnout / life events.** *Mitigation:* stage gates are natural pause points; every stage ends shipped, so a pause never strands unpublished work. The repo + this document make the program resumable by future-us or collaborators.
5. **Kernel engineering exceeds solo capability.** *Mitigation:* escalation ladder stops at "fast enough for the claim" (Triton usually suffices for research-grade wall-clock); a perfect kernel is Stage 3+ hire's job.
6. **The field moves past sparse attention** (e.g., a new paradigm wins recall-at-length outright). *Mitigation:* the mission's true asset is *recall-at-length measurement expertise + public trust*, which transfers to whatever architecture wins. Re-platform, keep the mission.

## Review cadence

- **Monthly (solo stages):** one log entry in wiki/log.md — progress vs current stage workstreams.
- **Every stage boundary:** fill the scoreboard row, write expected-vs-actual, amend this document explicitly.
- **Every 6 months:** re-read the risk register; check whether the one-axis bet is still the right axis against the field ([[subquadratic-architectures]] page kept current).

## Relationships

- [[ssa-recall-research-6month-plan|Stage 0 is this plan, executed @part-of]]
- [[subq-subquadratic-startup-deep-dive|SubQ's unverified 12M claim is the open door @inspired-by]]
- [[associative-recall|the recall wall is the scientific core of every stage @extends]]
- [[hybrid-architectures|Stage 1+ models are hier-attention hybrids @uses]]
- [[zoology-recall|MQAR/Zoology is the Stage 0 measurement instrument @uses]]
- [[attention-complexity-theory|sparse selection is the loophole the hardness result leaves open @uses]]
