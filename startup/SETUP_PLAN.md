# Eumnesia Labs — startup setup plan (do not forget!)

*Written 2026-07-24. Practical checklist for getting the company's basic identity + free
cloud credits in place. Total cash cost: ~₹1,000/year (just the domain). Everything else free.*

> ⚠️ **Before the research repo goes public (M6): move this whole `startup/` folder out of
> `ssa-recall`.** The paper must stay independent research — no company branding near it.

## The locked names (exact spellings!)

- **Company:** Eumnesia Labs — `E-u-m-n-e-s-i-a` (say "yoom-NEE-zia").
  Greek: *eu* (good) + *mnesia* (memory) — the opposite of amnesia. Real but unclaimed
  clinical term; zero companies/brands using it (checked 2026-07-24 — re-check before buying).
- **Model/product:** Saqade — `S-a-q-a-d-e` (say "sah-KAHD"). First release: **Saqade-1**.
  From *saccade* (the eye's jump straight to what matters) respelled with Q (the quadratic
  cost we remove). Zero web presence (checked 2026-07-24).
- Tagline drafts: *"Eumnesia Labs — the end of AI amnesia."* /
  *"Saqade-1 — attention that knows where to look."*
- ❌ Rejected names, do not revisit: SubQAI (collides with SubQ, the company our research
  audits — legal + credibility disaster), SmritiQ (live AI diary app — smritiq.com, caught
  by manual Google check after automated search missed it), Hiracle, Exactal, Anamna,
  Fractall, Nexact, Trescall, Smaraq, Mnema, Remara (all taken).

## Step 1 — Buy the domain (~₹800–1,200/yr) — THE ONLY MANDATORY COST

- [ ] Final manual check first (the SmritiQ lesson): Google `"Eumnesia"` + `"eumnesia ai"`
      + Play Store search. Automated search ≠ enough.
- [ ] Buy **eumnesia.com** (Namecheap / Porkbun / GoDaddy). Triple-check spelling in cart.
- [ ] Skip `.ai` for now (~₹6–8k/yr) — add after funding.
- [ ] Same hour: grab free handles — GitHub org, X, LinkedIn company page.

## Step 2 — Free domain email — Zoho Mail Forever Free

https://www.zoho.com/mail/custom-domain-email.html — free forever: 5 users × 5GB, one
custom domain. (Indian company, works fine from India.)

- [ ] Add domain in Zoho Mail → verify via TXT record in registrar DNS.
- [ ] Set Zoho's MX records.
- [ ] Create `anuj@eumnesia.com` (+ `founder@` alias).
- Limitation: webmail/mobile app only on free plan (no IMAP/Outlook) — fine for now.

## Step 3 — Free landing page — GitHub Pages (or Cloudflare Pages)

- [ ] Single-page site: name, tagline, what we're building, research note, contact email.
      (Claude drafts the HTML — just ask.)
- [ ] Repo → GitHub Pages → connect custom domain → free HTTPS.
- [ ] Keep it minimal and credible; no fake claims, no product screenshots that don't exist.

## Step 4 — AWS Activate Founders — $1,000 free credits

https://aws.amazon.com/startups (Founders tier, for self-funded startups)

- Eligibility (we qualify): <10 years old, <10 employees, <$1M revenue/funding, no VC or
  accelerator, no prior Activate credits.
- Requirements: AWS account on **paid tier** (card on file; nothing to pay upfront),
  company **website** (step 3) + **domain email** (step 2).
- [ ] Apply with `anuj@eumnesia.com`.
- Note: applying to Founders now does NOT disqualify the bigger **Portfolio tier
  ($5k–$100k)** later via an accelerator/VC. Both tiers add ~$350 developer-support credits.

## Bonus — stack these too (all free, no funding required)

- [ ] **Microsoft for Startups Founders Hub** — Azure credits, famously easy approval.
- [ ] **Google for Startups Cloud Program** — GCP credits for self-funded startups.
- [ ] **NVIDIA Inception** — no credits, but discounts/perks + looks good on the deck.

## Expectations (so we don't fool ourselves)

- These credits fund the **startup's** infra (site, demos, API) — NOT the research GPUs.
  New cloud accounts get tight GPU quotas. Research stays on Kaggle/Colab + rented L40S.
- Order of operations stays: **paper first, company second.** Repo stays `ssa-recall`;
  no company name in commits, the arXiv submission, or the public release.
