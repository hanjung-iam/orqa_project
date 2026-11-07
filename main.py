"""
ORQA experiment runner (v2)

新實驗
    python main.py --config config/experiment.yaml
    python main.py --config config/experiment.yaml --method skill_rag --provider transformers
    python main.py --config config/experiment.yaml --method basic --limit 5          # smoke test
    python main.py --config config/experiment.yaml --method rag --dry-run            # 只印出 prompt，不呼叫 LLM

續跑 / 分階段
    python main.py --resume <experiment_id>                         # 從中斷處繼續（使用 experiment.json 的設定）
    python main.py --resume <experiment_id> --limit 1468            # 把 smoke test 擴大成完整實驗
    python main.py --resume <experiment_id> --stage post_analysis   # 只做 Haiku 後處理（已完成的不重做）
    python main.py --resume <experiment_id> --stage post_analysis --reanalyze
    python main.py --resume <experiment_id> --stage evaluate        # 只重新計算 evaluation

列出實驗
    python main.py --list
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from pathlib import Path

from tqdm import tqdm

from orqa.config import Config, ConfigError, apply_overrides, load_config, validate_config
from orqa.dataset import load_dataset
from orqa.experiment import (STATUS_EVALUATED, STATUS_FAILED, STATUS_INFERENCE_DONE, STATUS_INTERRUPTED,
                             STATUS_POST_DONE, STATUS_RUNNING, Experiment, ExperimentError)
from orqa.prompts import PROMPT_VERSION, build_prompt, build_system_prompt, render_prompt
from orqa.storage import EVALUATION_POST, EVALUATION_RAW, EXPERIMENT, RAW_PREDICTIONS, read_json

STAGES = ("all", "inference", "post_analysis", "evaluate")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ORQA experiment runner",
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--config", default="config/experiment.yaml")
    p.add_argument("--method", choices=["basic", "skill", "rag", "skill_rag"])
    p.add_argument("--provider", choices=["anthropic", "transformers"])
    p.add_argument("--model-id")
    p.add_argument("--limit", type=int, help="只跑前 N 題")
    p.add_argument("--experiment-id", help="新實驗的 ID（預設自動產生）")
    p.add_argument("--resume", metavar="EXPERIMENT_ID", help="續跑既有實驗")
    p.add_argument("--stage", choices=STAGES, default="all")
    p.add_argument("--reanalyze", action="store_true", help="post-analysis 重新分析所有 invalid（不覆寫舊紀錄，新增一筆）")
    p.add_argument("--dry-run", action="store_true", help="只建立並印出第一題的 prompt，不呼叫 LLM")
    p.add_argument("--list", action="store_true", help="列出 results/ 中的實驗")
    return p.parse_args(argv)


# --------------------------------------------------------------------- helpers

def sha256_file(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def fmt(x) -> str:
    return "-" if x is None else (f"{x:.4f}" if isinstance(x, float) else str(x))


def load_inputs(config: Config):
    dataset = load_dataset(config.dataset.path, config.dataset.name).head(config.experiment.limit)
    skill = None
    if config.experiment.use_skill:
        from orqa.skill import load_skill
        skill = load_skill(config.skill.path, config.skill.source_type)
    rag = None
    if config.experiment.use_rag:
        from orqa.rag.pipeline import RAGPipeline
        rag = RAGPipeline(config.rag)
        rag.open()  # dense / hybrid：index 未建立時在這裡就報錯（尚未建立實驗目錄）
    system_prompt = build_system_prompt(config.experiment.method, skill)
    prompt_info = {"version": PROMPT_VERSION,
                   "system_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()}
    return dataset, skill, rag, prompt_info


def list_experiments(output_dir: Path) -> None:
    rows = []
    for d in sorted(output_dir.glob("*/")):
        meta_path = d / EXPERIMENT
        if not meta_path.exists():
            continue
        meta = read_json(meta_path)
        if "config" not in meta:
            rows.append((d.name, "legacy (v1)", meta.get("method", "-"), meta.get("model", "-"), "-", "-"))
            continue
        raw = read_json(d / EVALUATION_RAW)["accuracy"] if (d / EVALUATION_RAW).exists() else None
        post = read_json(d / EVALUATION_POST)["accuracy"] if (d / EVALUATION_POST).exists() else None
        rows.append((d.name, meta["experiment_status"], meta["inference_mode"], meta["model_id"], fmt(raw), fmt(post)))
    print(f"{'experiment_id':<52} {'status':<24} {'mode':<10} {'model':<26} {'raw_acc':<8} post_acc")
    for r in rows:
        print(f"{r[0]:<52} {r[1]:<24} {r[2]:<10} {r[3]:<26} {r[4]:<8} {r[5]}")


# ---------------------------------------------------------------------- stages

def stage_inference(exp: Experiment, config: Config, provider, dataset, skill, rag) -> tuple[bool, bool]:
    """回傳 (是否全部完成, 是否遇到 fatal error)。"""
    from orqa.engine import InferenceAborted, InferenceEngine
    from orqa.llm.base import FatalLLMError

    engine = InferenceEngine(provider, config.experiment.method, skill=skill, rag=rag, experiment_id=exp.id)
    done_before = len(set(dataset.indices) & exp.storage.completed_indices())
    run = exp.start_run("inference", questions=len(dataset), already_completed=done_before)
    exp.set_status(STATUS_RUNNING)
    bar = tqdm(total=len(dataset), initial=done_before, desc="Inference", unit="q")
    status, summary, error = STATUS_INTERRUPTED, None, None
    try:
        summary = engine.run(dataset, exp.storage,
                             max_consecutive_errors=config.experiment.max_consecutive_errors,
                             progress=bar.update)
    except KeyboardInterrupt:
        error = "KeyboardInterrupt"
    except InferenceAborted as exc:
        error = f"aborted: {exc}"
    except FatalLLMError as exc:
        status, error = STATUS_FAILED, f"fatal: {exc}"
    finally:
        bar.close()
        complete = set(dataset.indices) <= exp.storage.completed_indices()
        if complete:
            status = STATUS_INFERENCE_DONE
        exp.meta["raw_predictions_sha256"] = sha256_file(exp.storage.path(RAW_PREDICTIONS))
        exp.finish_run(run, summary=summary.as_dict() if summary else None, error=error,
                       usage=provider.get_usage(), model_info=_safe_info(provider))
        exp.set_status(status)

    if error:
        print(f"\n[!] Inference stopped ({error}).")
    if not complete:
        missing = len(set(dataset.indices) - exp.storage.completed_indices())
        print(f"[!] {missing} question(s) not completed. Resume with:  python main.py --resume {exp.id}")
    return complete, status == STATUS_FAILED


def _safe_info(provider) -> dict:
    try:
        return provider.get_model_info()
    except Exception as exc:  # GPU 資訊等非必要欄位失敗不影響主流程
        return {"error": str(exc)}


def stage_post_analysis(exp: Experiment, config: Config, dataset, *, reanalyze: bool) -> None:
    from orqa.analysis import InvalidAnswerAnalyzer, run_post_analysis

    analyzer, provider = None, None
    pa = config.post_analysis
    invalid = sum(1 for r in exp.storage.raw_predictions().values() if r.get("prediction") is None)
    if pa.enabled and invalid:
        from orqa.llm import create_post_analysis_provider
        provider = create_post_analysis_provider(pa)
        analyzer = InvalidAnswerAnalyzer(provider, pa.model_id)

    run = exp.start_run("post_analysis", enabled=pa.enabled, model_id=pa.model_id, reanalyze=reanalyze)
    bar = tqdm(total=0, desc="Post-analysis", unit="q", disable=analyzer is None)

    def progress(done, total):
        bar.total = total
        bar.n = done
        bar.refresh()

    try:
        summary = run_post_analysis(exp.storage, dataset, analyzer, reanalyze=reanalyze,
                                    max_consecutive_errors=config.experiment.max_consecutive_errors,
                                    progress=progress)
    finally:
        bar.close()
    if exp.meta.get("raw_predictions_sha256") not in (None, sha256_file(exp.storage.path(RAW_PREDICTIONS))):
        raise RuntimeError("predictions_raw.jsonl changed during post-analysis — this must never happen")
    exp.finish_run(run, summary=summary.as_dict(), usage=provider.get_usage() if provider else None)
    print(f"Post-analysis: invalid={summary.original_invalid} resolved={summary.resolved} "
          f"unresolved={summary.unresolved} failed={summary.failed} (API calls this run: {summary.analyzed_now})")
    if summary.failed:
        print(f"[!] {summary.failed} analysis call(s) failed; rerun --stage post_analysis to retry them.")


def stage_evaluate(exp: Experiment, dataset) -> dict:
    from orqa.evaluation import run_evaluation

    recorded = exp.meta.get("raw_predictions_sha256")
    current = sha256_file(exp.storage.path(RAW_PREDICTIONS))
    integrity = "ok" if recorded in (None, current) else "MODIFIED_AFTER_INFERENCE"
    result = run_evaluation(exp.storage, dataset)
    raw, post = result["raw"], result["post"]
    m = post["post_analysis_metrics"]
    exp.meta["results_summary"] = {
        "total_questions": raw["total_questions"],
        "raw_accuracy": raw["accuracy"],
        "raw_invalid_count": raw["invalid_count"],
        "post_analysis_accuracy": post["accuracy"],
        "remaining_invalid_count": post["remaining_invalid_count"],
        "missing_count": raw["missing_count"],
        "haiku_resolution_rate": m["haiku_resolution_rate"],
        "accuracy_improvement": m["accuracy_improvement"],
        "raw_predictions_integrity": integrity,
    }
    exp.save()

    print("\n" + "=" * 64)
    print(f"Experiment {exp.id}")
    print("=" * 64)
    print(f"{'':<22}{'Raw':>12}{'Post-analysis':>16}")
    print(f"{'Accuracy':<22}{fmt(raw['accuracy']):>12}{fmt(post['accuracy']):>16}")
    print(f"{'Correct':<22}{raw['correct_count']:>12}{post['correct_count']:>16}")
    print(f"{'Incorrect':<22}{raw['incorrect_count']:>12}{post['incorrect_count']:>16}")
    print(f"{'Invalid':<22}{raw['invalid_count']:>12}{post['invalid_count']:>16}")
    print(f"{'Missing':<22}{raw['missing_count']:>12}{post['missing_count']:>16}")
    print(f"Haiku resolution rate : {fmt(m['haiku_resolution_rate'])}   "
          f"checked_invalid: {m['checked_invalid']}   improvement: {fmt(m['accuracy_improvement'])}")
    print(f"Parse methods (raw)   : {raw['parse_method_counts']}")
    print("\nPer question type (raw -> post):")
    for qt, s in raw["per_question_type"].items():
        ps = post["per_question_type"][qt]
        print(f"  {qt:<4} n={s['total']:<5} {fmt(s['accuracy'])} -> {fmt(ps['accuracy'])}")
    if integrity != "ok":
        print("[!] predictions_raw.jsonl differs from the hash recorded after inference.")
    print(f"\nResults: {exp.dir}")
    return result


# ------------------------------------------------------------------------ main

def main(argv=None) -> int:
    args = parse_args(argv)
    base = load_config(args.config, validate=False)
    output_dir = Path(base.experiment.output_dir)

    if args.list:
        list_experiments(output_dir)
        return 0

    if args.resume:
        if any(getattr(args, k) is not None for k in ("method", "provider", "model_id", "experiment_id")):
            print("--resume uses the settings stored in experiment.json; "
                  "--method/--provider/--model-id/--experiment-id cannot be combined with it.")
            return 2
        exp = Experiment.load(output_dir, args.resume)
        config = exp.config
        if args.limit is not None:
            config = dataclasses.replace(config, experiment=dataclasses.replace(config.experiment, limit=args.limit))
            exp.meta["config"] = config.to_dict()
            exp.meta["dataset"]["limit"] = args.limit
        validate_config(config)
    else:
        if args.stage in ("post_analysis", "evaluate"):
            print(f"--stage {args.stage} requires --resume <experiment_id>")
            return 2
        exp = None
        config = apply_overrides(base, method=args.method, provider=args.provider, model_id=args.model_id,
                                 limit=args.limit, experiment_id=args.experiment_id)

    dataset, skill, rag, prompt_info = load_inputs(config)
    print(f"Dataset: {config.dataset.path} ({len(dataset)} questions, sha256={dataset.sha256[:12]})")
    print(f"Method : {config.experiment.method} | Provider: {config.llm.provider} | Model: {config.llm.active.model_id}")
    if skill:
        print(f"Skill  : {skill.name} {list(skill.file_names)} sha256={skill.sha256[:12]}")
    if rag:
        print(f"RAG    : {rag.collection_name} mode={config.rag.retrieval.mode} chunks={len(rag.chunks)}")

    if args.dry_run:
        q = dataset[0]
        retrieval = rag.retrieve_for_question(q) if rag else None
        prompt = build_prompt(q, config.experiment.method, skill=skill,
                              retrieved_chunks=retrieval.texts if retrieval else None)
        print(render_prompt(prompt))
        if retrieval:
            print(json.dumps(retrieval.to_record(include_content=False), indent=2, ensure_ascii=False))
        return 0

    rag_key = rag.index_key if rag else None
    run_inference = args.stage in ("all", "inference")
    provider = None
    if run_inference:
        from orqa.llm import create_provider
        provider = create_provider(config.llm)

    if exp is None:
        exp = Experiment.create(config, dataset=dataset, skill=skill, prompt_info=prompt_info,
                                provider_info=_safe_info(provider),
                                rag_info=rag.describe() if rag else None,
                                experiment_id=config.experiment.experiment_id)
        print(f"Created experiment: {exp.dir}")
    else:
        exp.verify(dataset=dataset, skill=skill, prompt_sha256=prompt_info["system_sha256"], rag_index_key=rag_key)
        print(f"Resuming experiment: {exp.dir} (status={exp.meta['experiment_status']})")

    try:
        if run_inference:
            complete, fatal = stage_inference(exp, config, provider, dataset, skill, rag)
            if fatal:
                print("Stopped because of a non-retryable error (check API key / model id / config). "
                      f"Fix it, then resume with: python main.py --resume {exp.id}")
                return 1
            if not complete and args.stage == "all":
                print("Inference incomplete; post-analysis/evaluation will run on the completed questions only.")
        if args.stage in ("all", "post_analysis"):
            stage_post_analysis(exp, config, dataset, reanalyze=args.reanalyze)
            if exp.meta["experiment_status"] == STATUS_INFERENCE_DONE:
                exp.set_status(STATUS_POST_DONE)
        if args.stage in ("all", "evaluate", "post_analysis"):
            stage_evaluate(exp, dataset)
            if exp.meta["experiment_status"] in (STATUS_INFERENCE_DONE, STATUS_POST_DONE):
                exp.set_status(STATUS_EVALUATED)
    finally:
        if provider is not None:
            provider.close()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ConfigError, ExperimentError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)
