# Eumnesia Labs — company context & brand brief

*Written 2026-07-24. Two parts. **Part A is safe to paste into any design tool** (Claude,
Figma AI, a human designer's inbox). **Part B is internal only** — never paste it anywhere
public, never into third-party tools.*

---

## PART A — PUBLIC BRAND BRIEF (safe to share / paste for design work)

### Who we are

**Eumnesia Labs** — an AI research lab founded in 2026, based in India, building toward one
goal: **the end of AI amnesia.**

The name: *eu·mne·sia*, from Greek *eu* (good, healthy) + *mnēsis* (memory). In clinical
language, eumnesia is memory working as it should — nothing lost, nothing blurred. It is
the opposite of amnesia. That word is our standard for machine intelligence.

### The problem we exist to solve

Today's AI models forget. To handle long inputs cheaply, efficient models compress the past
into a small fixed-size memory — and provably lose exact facts they read moments ago
(the field calls this the recall wall). The alternative, full attention, remembers
everything but its cost grows quadratically until long context becomes unaffordable.
Every AI system today picks a side of that wall.

### What we build

Attention architectures that keep a model's memory **exact** while reading only the small
part of the past that matters — recall of full attention at a fraction of the cost.
We are research-first: every claim ships with open, reproducible experiments. First paper
expected 2026.

### Brand personality

- **Honest science.** Claims with receipts. No hype, no vague promises. Plain words.
- **Precision.** We care about exactness — it is literally our product.
- **Calm confidence.** A lab, not a gadget company. Think Anthropic's tone, not a crypto launch.
- **Preservation.** What enters memory stays perfect. Nothing fades.

### Visual identity (already established — extend, don't replace)

- **Ink**: `#0D1219` — deep blue-black ground (never pure black).
- **Amber**: `#DFAC4C` — THE accent. Chosen because amber preserves things perfectly for
  millions of years. Used sparingly: one amber element per composition.
- **Slate**: `#232D3B` (quiet surfaces/lines) and `#8C97A6` (muted text).
- **Type**: serif display (Iowan/Palatino/Georgia family) for headlines; clean system sans
  for body; monospace for small uppercase labels.
- **Existing logo mark v1**: a 3×3 grid of rounded memory blocks, eight in faint slate,
  **one in amber** — "nine blocks, one remembered." Also echoes selecting the one block of
  the past that matters.
- **Existing motif**: the word "amnesia" dissolving/fading; solid amber = what is kept.
- **Tagline**: *The end of AI amnesia.*

### Design directions worth exploring (for logo/brand work)

- Memory blocks / grids with a single preserved element (current mark's family).
- Amber as material: inclusion in amber, a solid drop, light through amber.
- Dissolve vs. permanence: fading letterforms against one crisp element.
- Precise geometry: monospaced grids, exact alignment — precision as an aesthetic.

### Explicitly avoid

- Brains, neurons, robots, circuit boards, owls/elephants ("memory animal" clichés).
- Purple-blue gradient "AI startup" look; neon acid-green on black; glowing wireframes.
- Busy marks that die at favicon size — everything must survive at 16px.
- Any imagery implying products/scale we don't have (server racks, dashboards, teams).

---

## PART B — INTERNAL CONTEXT (never paste outside; strip before sharing)

### Confidential items

- **Saqade** (S-a-q-a-d-e): codename of the model/architecture line ("Saqade-1"). From
  *saccade* — the eye's jump straight to what matters — respelled with Q (the quadratic
  cost we remove). **Not public until the paper.** Do not put in design briefs, the site,
  or any tool.
- The technical mechanism (hierarchical sub-quadratic selector, budget results, etc.) —
  lives in the research repo, never in brand materials.
- The research's relationship to any specific company's claims — the paper stays
  independent; the brand never mentions other companies.

### Founder

Anuj Dev Singh (anujdev9928@gmail.com / anuj@eumnesia.com) — solo founder. Name kept off
public materials for now by choice; revisit at paper release.

### The 5-year arc (aspiration, revised as reality arrives)

- **Year 1 — 2026: Prove it.** Independent reproducible research on recall-preserving
  sub-quadratic attention → arXiv paper + workshop submission + open repo. Brand exists
  (site, email, logo). Free-tier compute + rented GPUs. Outcome: credibility.
- **Year 2 — 2027: Open-source gravity.** Release the kernel/benchmark as open source;
  Saqade-1 small-scale preview models; build the niche audience that cares about long
  context. Join an accelerator / raise pre-seed on the back of the paper. Outcome: users
  of our code, first funding.
- **Year 3 — 2028: First product.** Saqade-1 as a real trained model + API for
  long-context workloads (document/codebase/agent memory). Seed round. First 3–5 hires
  (research + infra). Outcome: revenue exists.
- **Year 4 — 2029: The memory layer.** From model to infrastructure: the memory backbone
  for AI agents — exact recall over months of history at flat cost. Enterprise pilots.
  Series A. Outcome: "Eumnesia inside" other people's products.
- **Year 5 — 2030: Category owner.** When the industry says "AI that doesn't forget,"
  they mean us. Frontier-scale partnerships or the standalone platform — decided by how
  Years 2–4 actually go.

### Standing rules

1. **Paper before product claims.** Nothing goes on the site the paper hasn't earned.
2. **Research repo (`ssa-recall`) and company stay separate** until public release day.
3. All setup logistics: see `startup/SETUP_PLAN.md` (domain, Zoho, Pages, AWS credits).
