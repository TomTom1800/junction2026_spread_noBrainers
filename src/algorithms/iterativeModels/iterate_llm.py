"""Ask the signed-in Codex CLI for dispatcher improvements and keep measured gains.

Sign in to Codex CLI with your ChatGPT account, then run from the repository root:
    python src/algorithms/iterativeModels/iterate_llm.py --seed sweep-a --seed sweep-b --iterations 10

Each candidate is compared with the current best on the same practice seeds. Only
single-file patches to the selected dispatcher are accepted. Practice worlds use the
team's normal API quota and are created by the dispatcher.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional


ROOT = Path(__file__).resolve().parents[3]
DISPATCH = ROOT / "src" / "algorithms" / "dispatch_upgraded.py"
DISPATCH_NAME = DISPATCH.name
DISPATCH_RELATIVE = DISPATCH.relative_to(ROOT).as_posix()
RULES = ROOT / "docs" / "RULES.md"
SCORE_RE = re.compile(r"End\s+margin €\s*([\d,.-]+)\s+revenue €\s*[\d,.-]+\s+\d+ orders\s+(\d+(?:\.\d+)?)% on time")
WORLD_ID_RE = re.compile(r"(?:Practice world|Ranking run)\s+([A-Za-z0-9_-]+)")
TRANSIENT_ERROR_RE = re.compile(
    r"(?:timeout|timed out|temporarily unavailable|connection reset|connection "
    r"aborted|connection error|\b502\b|\b503\b|\b504\b|\b429\b)",
    re.IGNORECASE,
)


def is_transient_server_error(error: BaseException, output: str = "") -> bool:
    """Identify failures that are safe to retry while the service recovers."""
    details = f"{type(error).__name__}: {error}\n{output}"
    return isinstance(error, (TimeoutError, ConnectionError, subprocess.TimeoutExpired)) or bool(
        TRANSIENT_ERROR_RE.search(details)
    )


def wait_before_retry(operation: str, attempt: int, retry_delay: float) -> None:
    delay = retry_delay * (2 ** (attempt - 1))
    print(
        f"{operation} hit a transient server error; retrying in {delay:g}s "
        f"(attempt {attempt})...",
        file=sys.stderr,
        flush=True,
    )
    time.sleep(delay)


def collect_diagnostics(world_id: str, server_retries: int, retry_delay: float) -> dict[str, Any]:
    """Fetch compact events and reveal details for one completed practice world."""
    from spread.client import Client

    world = Client().world(world_id)
    for attempt in range(server_retries + 1):
        try:
            events = world.events()
            reveal = world.reveal()
            break
        except Exception as error:
            if not is_transient_server_error(error) or attempt == server_retries:
                raise
            wait_before_retry("Fetching world diagnostics", attempt + 1, retry_delay)
    event_counts = Counter(event["kind"] for event in events)
    ledger_totals = Counter()
    failures, drops, refunds = [], [], []
    deadlines: dict[str, dict[str, Any]] = {}
    cancelled: set[str] = set()
    for event in events:
        kind = event["kind"]
        if kind == "order_won":
            order = event["order"]
            deadlines[order["id"]] = order
        elif kind == "order_cancelled":
            cancelled.add(event["orderId"])
        elif kind == "visit_failed":
            failures.append({key: event.get(key) for key in ("at", "orderId", "leg", "reason")})
        elif kind in ("gig_dropped", "no_show"):
            drops.append({key: event.get(key) for key in ("at", "kind", "gigId", "driverId")})
        elif kind == "ledger":
            line = event["line"]
            ledger_totals[line["kind"]] += line["cents"]
            if line["kind"] in ("late_refund", "undelivered_penalty", "fine", "failed_visit"):
                refunds.append({key: line.get(key) for key in ("at", "kind", "cents", "orderId", "detail")})

    outcomes = Counter()
    option_outcomes = Counter()
    quote_ratios: dict[str, list[float]] = {}
    for item in reveal["requests"]:
        outcome, option = item.get("outcome", "unknown"), item.get("option")
        outcomes[outcome] += 1
        if option:
            option_outcomes[f"{outcome}:{option}"] += 1
        quotes = item.get("quote") or {}
        request = item["request"]
        for offer in request.get("options", []):
            quote = quotes.get(offer["kind"])
            list_price = offer.get("listPriceCents", 0)
            if quote and list_price:
                quote_ratios.setdefault(offer["kind"], []).append(quote / list_price)
    quote_summary = {kind: {"count": len(values), "mean_vs_list": round(sum(values) / len(values), 3)}
                     for kind, values in quote_ratios.items()}

    problem_orders = []
    for revealed in reveal["orders"]:
        order_id = revealed["orderId"]
        if order_id in cancelled:
            continue
        order = deadlines.get(order_id, {})
        deadline = order.get("deadline")
        legs = []
        has_problem = False
        for index, truth in enumerate(revealed["legs"]):
            delivered = truth.get("deliveredAt")
            late = deadline is not None and (delivered is None or delivered > deadline)
            has_problem |= late
            planned_leg = order.get("legs", [{}] * len(revealed["legs"]))[index]
            legs.append({
                "pickup": planned_leg.get("pickup", {}).get("placeId"),
                "delivery": planned_leg.get("delivery", {}).get("placeId"),
                "readyAt": truth.get("readyAt"), "releasedAt": truth.get("releasedAt"),
                "pickedUpAt": truth.get("pickedUpAt"), "deliveredAt": delivered,
                "noteDeadline": truth.get("noteDeadline"), "manualGearbox": truth.get("manualGearbox"),
                "needsPlates": truth.get("needsPlates"), "summerTyres": truth.get("summerTyres"),
                "late": late,
            })
        if has_problem:
            problem_orders.append({"orderId": order_id, "option": order.get("option"),
                                  "deadline": deadline, "priceCents": order.get("priceCents"), "legs": legs})

    views = reveal.get("views", [])
    bookings = [event.get("feeCents", 0) for event in events if event["kind"] == "gig_booked"]
    return {
        "worldId": world_id,
        "eventCounts": dict(event_counts),
        "requestOutcomes": dict(outcomes), "wonOptions": dict(option_outcomes),
        "meanQuotedPriceVsList": quote_summary,
        "ledgerCentsByKind": dict(ledger_totals),
        "visitFailures": failures[:12], "dropsAndNoShows": drops[:12],
        "refundsFinesAndFailedVisits": refunds[:12], "lateOrUndeliveredOrders": problem_orders[:12],
        "gigViewCount": len(views),
        "bookings": {"count": len(bookings), "meanFeeCents": round(sum(bookings) / len(bookings)) if bookings else None},
    }


def request_patch(
    source: str,
    history: str,
    model: Optional[str],
    feedback: str = "",
    server_retries: int = 0,
    retry_delay: float = 5.0,
) -> str:
    rules = RULES.read_text(encoding="utf-8")
    # Send the game rules and the full strategy so the model can reason from the contract.
    prompt = f"""    Improve this Spread dispatcher using the measured results and game rules below.
    Return ONLY a unified diff that changes {DISPATCH_RELATIVE}. Keep changes focused;
preserve API correctness and the 90% on-time ranking minimum. Do not add dependencies.
    The patch must use standard diff headers (--- a/{DISPATCH_RELATIVE} / +++ b/{DISPATCH_RELATIVE}).
You MUST propose a concrete source-code improvement. Do not return an empty or no-op diff.
Prefer an evidence-based change over speculative rewrites. The caller tests each patch
on the same deterministic practice seeds and retains it only when its score tuple improves:
ranked seed count first, then total margin, then average on-time share.

PREVIOUS PATCH ISSUE (if any):
{feedback or "None; this is the first attempt for this iteration."}

MEASURED RUNS:
{history}

GAME RULES:
{rules}

CURRENT {DISPATCH_RELATIVE}:
```python
{source}
```
"""
    codex = shutil.which("codex")
    if not codex:
        raise RuntimeError("Codex CLI was not found on PATH. Install it and sign in with `codex login`.")
    with tempfile.TemporaryDirectory(prefix="spread-codex-llm-") as temp:
        output = Path(temp) / "answer.txt"
        command = [codex, "exec", "--ephemeral", "--sandbox", "read-only",
                   "--output-last-message", str(output)]
        if model:
            command += ["--model", model]
        command.append("-")  # read the prompt on stdin
        # Prevent an API key or custom API endpoint in the shell environment from
        # overriding Codex's ChatGPT sign-in and charging the API account.
        child_env = os.environ.copy()
        child_env.pop("OPENAI_API_KEY", None)
        child_env.pop("OPENAI_BASE_URL", None)
        for attempt in range(server_retries + 1):
            try:
                result = subprocess.run(command, input=prompt, text=True, capture_output=True,
                                        cwd=ROOT, env=child_env, timeout=600)
            except subprocess.TimeoutExpired as error:
                if attempt == server_retries:
                    raise RuntimeError("Codex CLI request timed out after 10 minutes") from error
                wait_before_retry("Codex CLI request", attempt + 1, retry_delay)
                continue
            if result.returncode:
                detail = result.stderr.strip() or result.stdout.strip() or "no error details"
                if is_transient_server_error(RuntimeError(detail), detail) and attempt < server_retries:
                    wait_before_retry("Codex CLI request", attempt + 1, retry_delay)
                    continue
                raise RuntimeError(f"Codex CLI failed (exit {result.returncode}): {detail[-3000:]}")
            break
        if not output.exists():
            raise RuntimeError("Codex CLI completed without writing its final response")
        return output.read_text(encoding="utf-8")


def extract_diff(response: str) -> str:
    fenced = re.search(r"```(?:diff)?\s*\n(.*?)\n```", response, re.S)
    diff = fenced.group(1) if fenced else response.strip()
    # Refuse edits outside the dispatcher before applying anything.
    headers = re.findall(r"^(?:---|\+\+\+) (.+)$", diff, re.M)
    if len(headers) < 2 or any(Path(header.split("\t")[0]).name != DISPATCH_NAME for header in headers):
        raise ValueError(f"Model response must be a unified diff touching only {DISPATCH_NAME}")
    diff = re.sub(
        r"^(---|\+\+\+) [^\n]+$",
        lambda m: f"{m.group(1)} {'a' if m.group(1) == '---' else 'b'}/{DISPATCH_NAME}",
        diff,
        flags=re.M,
    )
    if not diff.startswith("--- "):
        raise ValueError("No unified diff found in model response")
    return diff + "\n"


def apply_candidate(source: str, diff: str) -> str:
    with tempfile.TemporaryDirectory(prefix="spread-candidate-") as temp:
        directory = Path(temp)
        target = directory / DISPATCH_NAME
        target.write_text(source, encoding="utf-8")
        patch_file = directory / "candidate.patch"
        patch_file.write_text(diff, encoding="utf-8")
        applied = subprocess.run(["patch", "-s", "-p1", "-i", str(patch_file)], cwd=directory,
                                 text=True, capture_output=True)
        if applied.returncode:
            raise ValueError(f"Patch did not apply: {applied.stderr.strip() or applied.stdout.strip()}")
        candidate = target.read_text(encoding="utf-8")
        # Reject syntactically invalid source before spending practice worlds.
        compile(candidate, str(DISPATCH), "exec")
        return candidate


def evaluate(
    source: str,
    seeds: list[str],
    level: int,
    server_retries: int,
    retry_delay: float,
    evaluation_timeout: float,
) -> tuple[tuple[int, int, float], list[dict[str, Any]]]:
    runs: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="spread-eval-") as temp:
        candidate = Path(temp) / "dispatch.py"
        candidate.write_text(source, encoding="utf-8")
        for seed in seeds:
            command = [sys.executable, str(candidate), "--level", str(level), "--seed", seed]
            for attempt in range(server_retries + 1):
                try:
                    result = subprocess.run(
                        command, cwd=ROOT, text=True, capture_output=True, timeout=evaluation_timeout
                    )
                except subprocess.TimeoutExpired as error:
                    if attempt == server_retries:
                        raise RuntimeError(f"Evaluation timed out for seed {seed!r}") from error
                    wait_before_retry("Evaluation", attempt + 1, retry_delay)
                    continue
                output = f"{result.stderr}\n{result.stdout}"
                if result.returncode and is_transient_server_error(RuntimeError(output), output):
                    if attempt == server_retries:
                        raise RuntimeError(
                            f"Evaluation failed after {server_retries + 1} attempts for seed {seed!r}:\n{output}"
                        )
                    wait_before_retry("Evaluation", attempt + 1, retry_delay)
                    continue
                break
            match = SCORE_RE.search(result.stdout)
            world_match = WORLD_ID_RE.search(result.stdout)
            ranked = "Ranked." in result.stdout
            if result.returncode or not match or not world_match:
                raise RuntimeError(f"Evaluation failed for seed {seed!r}:\n{result.stderr}\n{result.stdout[-3000:]}")
            try:
                diagnostics = collect_diagnostics(world_match.group(1), server_retries, retry_delay)
            except Exception as error:
                if not is_transient_server_error(error):
                    raise
                print(
                    f"Diagnostics unavailable for seed {seed!r} after retries: {error}",
                    file=sys.stderr,
                    flush=True,
                )
                diagnostics = {
                    "worldId": world_match.group(1),
                    "error": f"{type(error).__name__}: {error}",
                }
            runs.append({"seed": seed, "margin_cents": round(float(match.group(1).replace(",", "")) * 100),
                         "on_time": float(match.group(2)) / 100, "ranked": ranked,
                         "diagnostics": diagnostics})
    score = (sum(run["ranked"] for run in runs), sum(run["margin_cents"] for run in runs),
             sum(run["on_time"] for run in runs) / len(runs))
    return score, runs


def main() -> int:
    parser = argparse.ArgumentParser(
        description=f"Iteratively improve {DISPATCH_RELATIVE} with Codex and seeded practice evaluations."
    )
    parser.add_argument("--seed", action="append", dest="seeds", help="training seed; repeat to set the optimization seed set")
    parser.add_argument("--holdout-seed", action="append", dest="holdout_seeds", help="separate validation seed; repeat to set holdout worlds")
    parser.add_argument("--no-holdout", action="store_true", help="skip independent validation worlds")
    parser.add_argument("--level", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--patch-retries", type=int, default=2, help="re-ask Codex after a no-op or unusable patch (default: 2 retries)")
    parser.add_argument("--server-retries", type=int, default=3, help="retries for transient server failures (default: 3)")
    parser.add_argument("--retry-delay", type=float, default=5.0, help="initial delay between server retries in seconds (default: 5)")
    parser.add_argument("--evaluation-timeout", type=float, default=600.0, help="maximum seconds for one simulated world (default: 600)")
    parser.add_argument("--model", help="Codex model override; defaults to the model configured in Codex CLI")
    args = parser.parse_args()
    if importlib.util.find_spec("spread") is None:
        parser.error(
            "Spread client package is not installed for this Python interpreter. "
            "From the project root, run `python3 -m venv .venv`, "
            "`. .venv/bin/activate`, then `pip install $SPREAD_URL/kit/spread-0.1.0-py3-none-any.whl`; "
            "rerun the optimizer with that environment active."
        )
    if args.iterations < 1:
        parser.error("--iterations must be at least 1")
    if args.patch_retries < 0:
        parser.error("--patch-retries must be 0 or greater")
    if args.server_retries < 0:
        parser.error("--server-retries must be 0 or greater")
    if args.retry_delay < 0:
        parser.error("--retry-delay must be 0 or greater")
    if args.evaluation_timeout <= 0:
        parser.error("--evaluation-timeout must be greater than 0")
    seeds = args.seeds or [f"iterate-{i}" for i in range(5)]
    holdout_seeds = [] if args.no_holdout else args.holdout_seeds or ["iterate-holdout-0", "iterate-holdout-1"]
    if len(set(seeds)) != len(seeds) or len(set(holdout_seeds)) != len(holdout_seeds):
        parser.error("Seed names must be unique within each set")
    overlap = set(seeds) & set(holdout_seeds)
    if overlap:
        parser.error(f"Training and holdout seeds must be separate; overlap: {', '.join(sorted(overlap))}")

    best_source = DISPATCH.read_text(encoding="utf-8")
    best_score, best_runs = evaluate(best_source, seeds, args.level, args.server_retries, args.retry_delay, args.evaluation_timeout)
    best_holdout_score, _ = evaluate(best_source, holdout_seeds, args.level, args.server_retries, args.retry_delay, args.evaluation_timeout) if holdout_seeds else (None, [])
    compact = lambda runs: [{"seed": run["seed"], "margin_cents": run["margin_cents"],
                             "on_time": run["on_time"], "ranked": run["ranked"]} for run in runs]
    print(f"Initial training score {best_score}; runs: {json.dumps(compact(best_runs))}", flush=True)
    if holdout_seeds:
        print(f"Initial holdout score {best_holdout_score} on {len(holdout_seeds)} separate seeds.", flush=True)
    history = [f"Baseline training score {best_score}; detailed diagnostics: {json.dumps(best_runs, ensure_ascii=False)}"]
    fatal_codex_error = False
    for iteration in range(1, args.iterations + 1):
        feedback = ""
        for attempt in range(1, args.patch_retries + 2):
            print(f"Iteration {iteration}/{args.iterations}, patch attempt {attempt}/{args.patch_retries + 1}: requesting Codex"
                  f"{f' ({args.model})' if args.model else ''}", flush=True)
            try:
                response = request_patch(
                    best_source, "\n\n".join(history[-5:]), args.model, feedback,
                    args.server_retries, args.retry_delay,
                )
                candidate = apply_candidate(best_source, extract_diff(response))
                if candidate == best_source:
                    raise ValueError("The patch applied but made no source-code change. Return a substantive code change.")
            except (RuntimeError, ValueError) as error:
                feedback = str(error)
                print(f"Patch attempt {attempt} unusable: {feedback}", file=sys.stderr, flush=True)
                history.append(f"Iteration {iteration}, patch attempt {attempt}: unusable response: {feedback}")
                if str(error).startswith(("Codex CLI was not found", "Codex CLI failed", "Codex CLI request timed out")):
                    print("Stopping because Codex could not return a candidate. Check Codex CLI installation and ChatGPT sign-in before rerunning.", file=sys.stderr, flush=True)
                    fatal_codex_error = True
                    break
                continue

            score, runs = evaluate(candidate, seeds, args.level, args.server_retries, args.retry_delay, args.evaluation_timeout)
            print(f"Candidate training score {score}; runs: {json.dumps(compact(runs))}", flush=True)
            if score <= best_score:
                feedback = f"The valid patch scored {score}, which did not beat the current training score {best_score}. Try a different evidence-based change."
                history.append(f"Iteration {iteration}: rejected training candidate {score}; diagnostics: {json.dumps(runs, ensure_ascii=False)}")
                break

            holdout_score, holdout_runs = evaluate(candidate, holdout_seeds, args.level, args.server_retries, args.retry_delay, args.evaluation_timeout) if holdout_seeds else (None, [])
            holdout_ok = not holdout_seeds or holdout_score >= best_holdout_score
            if holdout_seeds:
                print(f"Candidate holdout score {holdout_score} (current best {best_holdout_score}); "
                      f"{'passed' if holdout_ok else 'regressed'}.", flush=True)
            if holdout_ok:
                best_source, best_score, best_runs = candidate, score, runs
                if holdout_seeds:
                    best_holdout_score = holdout_score
                DISPATCH.write_text(best_source, encoding="utf-8")
                print(f"Accepted and saved to {DISPATCH}.", flush=True)
                holdout_note = "passed; score details withheld from Codex" if holdout_seeds else "disabled"
                history.append(f"Iteration {iteration}: accepted training score {score}; diagnostics: {json.dumps(runs, ensure_ascii=False)}; holdout={holdout_note}")
            else:
                feedback = "The patch improved training but regressed on the separate holdout seeds. Try a more general change."
                history.append(f"Iteration {iteration}: rejected by holdout; training score {score}; holdout details withheld; training diagnostics: {json.dumps(runs, ensure_ascii=False)}")
            break
        if fatal_codex_error:
            break
    print(f"Finished. Best training score {best_score}; runs: {json.dumps(compact(best_runs))}; "
          f"best holdout score: {best_holdout_score}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
