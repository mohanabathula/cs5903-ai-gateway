# Architecture Decision Records

Project 17 | Group 40 | Mohana Bathula | CS5903 2026
Repository: https://github.com/mohanabathula/cs5903-ai-gateway (final tag: cs5903-final-2026, commit SHA: [FILL IN])

Entries marked [FILL IN] need information only the author has. Remove this line once none remain.

---

## ADR-001 — Put LiteLLM in front of the vLLM Semantic Router

- **Date, status, scope:** Decided between 20 Sep and 6 Nov 2026 (exact day not recorded); Accepted; whole gateway.
- **Context and drivers:** Send easy requests to a small model and hard ones to a large model to cut inference cost. Also needed: API keys, rate limits, spend tracking, and an OpenAI-compatible endpoint for clients.
- **Options considered:**
  1. LiteLLM only (static model choice per request).
  2. Semantic Router only (no access control or spend tracking).
  3. LiteLLM in front, Semantic Router behind it.
- **Decision and rationale:** Option 3. LiteLLM handles access and accounting; the router handles model choice. LiteLLM sees the router as one model named `auto`. This is a judgment based on the two tools' feature sets, not a measurement.
- **Consequences and trade-offs:** Two hops per request, two sets of logs, and two places where failures appear (see ADR-007). Easy to swap either part.
- **Evidence:** `config/litellm_config.yaml`, `config/docker-compose.yml`; end-to-end run on 12 Nov 2026 (easy request answered by gemma through the full path).
- **Revisit trigger:** If router latency (about 2.3-2.6 s per decision in the logs) is unacceptable for the target workload, or LiteLLM adds native complexity routing.

## ADR-002 — Use the router's example-based complexity signal, with no trained classifier

- **Date, status, scope:** Decided between 20 Sep and 6 Nov 2026 (exact day not recorded); Accepted; router configuration.
- **Context and drivers:** Need an easy/hard label per request without a labelled training pipeline. The router learns only from example prompts in its config.
- **Options considered:**
  1. Example-based complexity signal (embedding similarity to easy and hard examples).
  2. Trained text classifier (TF-IDF logistic regression was tried offline).
  3. LLM-based classification (e.g. AnyJev), listed as future work.
- **Decision and rationale:** Option 1 inside the router. Measured: with 40 real examples per class and two cut-offs, easy vs hard AUC was 1.000 on 300 test cases. The offline TF-IDF model reached 3-class accuracy 97%, but its strongest cues were procedure and diagnosis words and a "20% co-pay" phrase, so it may be using generator patterns. It was used only as a side experiment.
- **Consequences and trade-offs:** No training step, but the router measures how a text looks, not how much reasoning it needs. It cannot separate medium from hard (ADR-005).
- **Evidence:** `evaluation/routing_accuracy.py`, `evaluation/results/`.
- **Revisit trigger:** Medium-vs-hard AUC stays near 0.5 after context reduction, or a clean (untuned) evaluation falls well below the reported 97%.

## ADR-003 — Build router examples from real calibration cases

- **Date, status, scope:** Decided between 20 Sep and 6 Nov 2026 (exact day not recorded); Accepted; `config_medium.yaml` example prompts.
- **Context and drivers:** The first configuration used generic example prompts (code and claims lines).
- **Options considered:** Generic examples; 20 real examples per tier; 40 real examples (hard list = 20 hard + 20 medium).
- **Decision and rationale:** Real examples, 40 easy and 40 hard, each about 2,000 characters, from calibration cases only (no example appears in the test set). Measured: hard-vs-easy AUC went from 0.497 (generic) to 0.851 (20 real) to 1.000 (40 real).
- **Consequences and trade-offs:** Examples are tied to this dataset's templates, so the result shows the method works on this kind of claim file, not that it generalises. Start-up takes about 6 minutes because the examples are embedded on CPU.
- **Evidence:** `evaluation/results/baseline_generic_examples.csv`, `run3_threshold_0.049.csv`, `run4_bands_90.csv`, `routing_accuracy.csv`; `evaluation/build_examples.py`.
- **Revisit trigger:** A different dataset or product wording is added, or the 6-minute start-up becomes a deployment problem.

## ADR-004 — Replace the single cut-off with two bands (-0.01 and 0.027)

- **Date, status, scope:** Decided between 20 Sep and 6 Nov 2026 (exact day not recorded); Accepted; supersedes the single `threshold: 0.049` setting.
- **Context and drivers:** With a single cut-off of 0.049, only 47% of cases were routed as intended and 79% of medium/hard cases were missed (sent to gemma), which is the costly error.
- **Options considered:** Single cut-off 0.049; two cut-offs `easy_below: -0.01`, `hard_above: 0.027` (medium in between).
- **Decision and rationale:** Two bands. Measured on 300 test cases: policy routing accuracy 97%, missed 6 of 200 (3%), wasted 4 of 100 (4%), 34% of traffic to gemma.
- **Consequences and trade-offs:** **The cut-offs were chosen from results on the same test cases, so 97% is optimistic.** A half-and-half check on an earlier setup gave 82% on the tuning half and 78% on the unseen half. A clean evaluation (tune on non-example calibration cases, run test cases once) is not done. The meaning of the `easy_below` / `hard_above` fields is the author's reading of the schema, which lists them without descriptions; the test run confirmed that all three labels are produced.
- **Evidence:** `config/config_medium.yaml` (lines 42-43), `evaluation/results/run4_bands_90.csv`, `routing_accuracy.csv`.
- **Revisit trigger:** A clean evaluation gives accuracy more than 5 points below 97%.

## ADR-005 — Route medium cases to the large model for now

- **Date, status, scope:** Decided between 20 Sep and 6 Nov 2026 (exact day not recorded); Accepted (temporary); decision `medium-route`.
- **Context and drivers:** Routing is what is being tested. Measured: the router labels 59 of 100 medium cases medium and 35 hard, and 42 of 100 hard cases hard and 58 medium; mean margins medium +0.027, hard +0.028. Medium and hard are not separated.
- **Options considered:** Medium to gemma; medium to Qwen; a separate medium model.
- **Decision and rationale:** Medium to Qwen3.6-27B, so no harder case is under-served. A separate medium model is planned.
- **Consequences and trade-offs:** Cost saving comes only from easy cases (about a third of traffic). Three-class accuracy is 197 of 300 (66%).
- **Evidence:** `config/config_medium.yaml` decisions `hard-route` (200), `medium-route` (175), `easy-route` (150), `default-route` (100).
- **Revisit trigger:** A method that separates medium from hard (context reduction, LLM-based classification, feature extraction) reaches AUC well above 0.5.

## ADR-006 — Port offset 100 and an SSH tunnel on the Docker bridge

- **Date, status, scope:** Decided between 20 Sep and 6 Nov 2026 (exact day not recorded); Accepted; deployment on netx3.
- **Context and drivers:** A host Redis already used port 6379. Qwen runs on slurm-comp01 and netx3 reaches it only through SSH.
- **Options considered:** Stop the host Redis; offset the router stack's ports; expose Qwen directly.
- **Decision and rationale:** `VLLM_SR_PORT_OFFSET=100` (routed listener 8899 becomes 8999). Qwen is reached through an SSH tunnel bound to the Docker bridge address 172.17.0.1:9000, so the router container can reach it without exposing it to the LAN.
- **Consequences and trade-offs:** The offset must be set for every `vllm-sr` command. The tunnel dropped several times during testing (not self-restarting), so a manual restart or `autossh` is needed.
- **Evidence:** `config/config_medium.yaml` (`base_url: http://172.17.0.1:9000/v1`).
- **Revisit trigger:** The tunnel causes a failed demo or evaluation run, or Qwen gets a reachable address.

## ADR-007 — Strip Qwen's `metrics` field with a small proxy (workaround for the 502)

- **Date, status, scope:** 12 Nov 2026; Accepted as a workaround; Qwen backend path.
- **Context and drivers:** Requests routed to Qwen returned 502 "The selected model returned an invalid response". Router log: `chat execution field must be null`.
- **Options considered:**
  1. Report upstream and wait.
  2. Try a newer `vllm-sr` (config format risk; 6-minute restart).
  3. A proxy between the router and Qwen that removes the field.
  4. Restart Qwen's vLLM with different settings (not tried).
- **Decision and rationale:** Option 3. Measured: comparing field names of raw gemma and raw Qwen replies, the only fields Qwen sends that gemma does not are the five under `metrics`. With the proxy removing `metrics`, hard requests return a normal Qwen reply through client, LiteLLM, router and Qwen. Disabling thinking per request did not fix the 502 earlier.
- **Consequences and trade-offs:** The proxy works around a router bug and does not fix it. It does not handle streaming replies. One request failed during testing because the SSH tunnel dropped, not because of the proxy.
- **Evidence:** `demo/qwen_shim.py`; `evaluation/results/metrics_finding/` (raw replies, shim log; if a file is missing, it was lost from /tmp and is not reconstructed).
- **Revisit trigger:** A newer `vllm-sr` accepts the unmodified reply, or the upstream issue is resolved.

## ADR-008 — Methodology status at 9 October 2026 (mandatory record)

- **Date, status, scope:** Written on **20 November 2026** (not backdated); Accepted.
- **Status:** **Not changed.** The core approach has stayed the same since before 9 October 2026: LiteLLM in front, the vLLM Semantic Router behind it, and an example-based complexity signal that sends easy requests to a small model and harder ones to a large model. This record is written on 20 November 2026, as the guide asks. [FILL IN: say whether a reaffirmation was sent to the mentor or course team on or before 9 October 2026. If none was sent, write "no separate record was made at the time".]
- **Tuning within the same approach (see ADR-003, ADR-004, ADR-007 for their dates):** generic examples replaced by real calibration examples, one cut-off replaced by two bands, and a proxy added to work around the Qwen 502. The architecture, the router, the dataset and the evaluation question did not change.
- **Project track:** 2-month project (Sept-Nov 2026), per the evaluation plan.
- **Evidence:** [FILL IN: mentor message, weekly log, or commit showing the approach at or before 9 October]
- **Revisit trigger:** Any later fundamental change needs the documented mentor approval from the evaluation framework.
