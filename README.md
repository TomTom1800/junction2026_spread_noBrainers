# junction2026_spread_noBrainers
Our code for the Junction 2026 main event. Our challenge is spread and tech challenges __. Junction is a hackathon where you try solving challenges in under 48 hours.

## LLM optimization loop

`src/algorithms/iterativeModels/iterate_llm.py` asks the signed-in Codex CLI for a small patch to
`src/algorithms/iterativeModels/dispatch_iterative.py`, evaluates the candidate and current best on the same
deterministic practice seeds, and saves only candidates that improve the score. Ranking
eligibility across the seeds is the first objective, followed by total margin and then
average on-time share. Codex requests use your signed-in ChatGPT/Codex allowance; practice
evaluations use the team's Spread practice API quota.

Install and sign in to Codex CLI with your ChatGPT account (`codex login`), and set
`SPREAD_URL` and `SPREAD_KEY`, then run:

```sh
python src/algorithms/iterativeModels/iterate_llm.py --seed sweep-a --seed sweep-b --iterations 10
```

It defaults to level 8, five training seeds (`iterate-0` through `iterate-4`), and two
separate holdout seeds (`iterate-holdout-0` and `iterate-holdout-1`). Candidates must improve
the training score without regressing the holdout score. Use repeated `--seed` and
`--holdout-seed` arguments to set those groups, or `--no-holdout` to skip validation.
Each unusable or no-op patch is retried twice by default; change that with `--patch-retries`.
The evaluator fetches each completed world's event log and reveal to provide failure,
late-delivery, quote, and driver-booking diagnostics to Codex. Those reads also use the
Spread practice quota. Accepted changes
are written directly to `src/algorithms/iterativeModels/dispatch_iterative.py`. The original `dispatch.py` remains unchanged
as a comparison baseline.

Codex CLI uses its configured model by default; pass `--model <name>` to override it.
The optimizer removes `OPENAI_API_KEY` and `OPENAI_BASE_URL` from the Codex subprocess
environment so the request uses the signed-in Codex account rather than API billing.
