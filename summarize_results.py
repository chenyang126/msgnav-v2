#!/usr/bin/env python
"""Aggregate GOAT-Bench eval pkl results into human-readable CSVs.

Reads the per-split pickle files written by src/logger_goatbench.py and produces:
  - <output_dir>/summary_subtasks.csv : one row per finished subtask
  - <output_dir>/summary_overall.csv  : overall + per-task-type success / SPL

Usage:
  python summarize_results.py                        # defaults to results/example_goatbench
  python summarize_results.py results/example_goatbench
"""
import csv
import glob
import os
import pickle
import sys
from collections import defaultdict


def load_merged_dict(output_dir, prefix):
    """Merge every success_by_distance_*.pkl style dict across splits."""
    merged = {}
    # match per-split files (with suffix) but skip the already-aggregated bare one
    for path in sorted(glob.glob(os.path.join(output_dir, f"{prefix}_*.pkl"))):
        try:
            with open(path, "rb") as f:
                merged.update(pickle.load(f))
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] {os.path.basename(path)}: {e}")
    return merged


def load_json_like(output_dir, prefix):
    import json

    merged = {}
    for path in sorted(glob.glob(os.path.join(output_dir, f"{prefix}_*.json"))):
        try:
            with open(path) as f:
                merged.update(json.load(f))
        except Exception:  # noqa: BLE001
            pass
    return merged


def merge_records_jsonl(output_dir):
    """Merge all per-worker records_*.jsonl into one list (live debug detail)."""
    import json

    records = []
    for path in sorted(glob.glob(os.path.join(output_dir, "records_*.jsonl"))):
        if os.path.basename(path) == "records_all.jsonl":
            continue
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] {os.path.basename(path)}: {e}")
    return records


def write_merged_records(output_dir, records):
    """Write a single merged records CSV + JSONL across all workers."""
    import csv
    import json

    if not records:
        return
    fields = list(records[0].keys())
    # union of all keys, preserving first-seen order
    for r in records:
        for k in r:
            if k not in fields:
                fields.append(k)
    merged_csv = os.path.join(output_dir, "records_all.csv")
    merged_jsonl = os.path.join(output_dir, "records_all.jsonl")
    with open(merged_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in sorted(records, key=lambda x: x.get("subtask_id", "")):
            row = dict(r)
            v = row.get("goal_obj_ids")
            if isinstance(v, (list, tuple)):
                row["goal_obj_ids"] = ";".join(str(x) for x in v)
            w.writerow(row)
    with open(merged_jsonl, "w") as f:
        for r in sorted(records, key=lambda x: x.get("subtask_id", "")):
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    return merged_csv, merged_jsonl


def main(output_dir):
    if not os.path.isdir(output_dir):
        sys.exit(f"Not a directory: {output_dir}")

    success = load_merged_dict(output_dir, "success_by_distance")
    spl = load_merged_dict(output_dir, "spl_by_distance")
    n_total = load_json_like(output_dir, "n_total_frames")
    n_filtered = load_json_like(output_dir, "n_filtered_frames")

    # merge the rich live records (JSONL) written per subtask, if present
    live_records = merge_records_jsonl(output_dir)
    if live_records:
        paths = write_merged_records(output_dir, live_records)

    if not success:
        print(f"No finished subtasks found in {output_dir} yet.")
        return

    # ---- per-subtask detail ----
    subtask_csv = os.path.join(output_dir, "summary_subtasks.csv")
    rows = []
    for sid in sorted(success):
        scene, episode, sub = (sid.split("_") + ["", "", ""])[:3]
        rows.append(
            {
                "subtask_id": sid,
                "scene": scene,
                "episode": episode,
                "subtask": sub,
                "success": success.get(sid, ""),
                "spl": round(spl.get(sid, float("nan")), 4) if sid in spl else "",
                "n_total_frames": n_total.get(sid, ""),
                "n_filtered_frames": n_filtered.get(sid, ""),
            }
        )
    with open(subtask_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- overall + per task type ----
    succ_task = load_merged_list(output_dir, "success_by_task")
    spl_task = load_merged_list(output_dir, "spl_by_task")

    overall_csv = os.path.join(output_dir, "summary_overall.csv")
    summary_rows = []
    n = len(success)
    overall_sr = 100.0 * sum(success.values()) / n
    overall_spl = 100.0 * sum(spl.values()) / len(spl) if spl else 0.0
    summary_rows.append(
        {"task_type": "OVERALL", "n": n, "success_rate_%": round(overall_sr, 2),
         "spl_%": round(overall_spl, 2)}
    )
    for tt in sorted(succ_task):
        vals = succ_task[tt]
        svals = spl_task.get(tt, [])
        summary_rows.append(
            {
                "task_type": tt,
                "n": len(vals),
                "success_rate_%": round(100.0 * sum(vals) / len(vals), 2) if vals else 0.0,
                "spl_%": round(100.0 * sum(svals) / len(svals), 2) if svals else 0.0,
            }
        )
    with open(overall_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task_type", "n", "success_rate_%", "spl_%"])
        w.writeheader()
        w.writerows(summary_rows)

    # ---- console print ----
    print(f"\nFinished subtasks: {n}")
    print(f"{'task_type':<14}{'n':>6}{'success%':>12}{'spl%':>10}")
    print("-" * 42)
    for r in summary_rows:
        print(f"{r['task_type']:<14}{r['n']:>6}{r['success_rate_%']:>12}{r['spl_%']:>10}")

    # ---- list failed subtasks for quick debugging ----
    failed = [sid for sid, v in success.items() if not v]
    if failed:
        print(f"\nFailed subtasks ({len(failed)}):")
        for sid in sorted(failed):
            info = ""
            for rec in live_records:
                if rec.get("subtask_id") == sid:
                    info = (f"  [{rec.get('task_type')}] q={rec.get('question','')[:50]!r} "
                            f"gt={rec.get('gt_answer','')} "
                            f"dist={rec.get('final_distance_to_goal')} "
                            f"steps={rec.get('n_steps')}")
                    break
            print(f"  {sid}{info}")

    print("\nWrote:")
    print(f"  {subtask_csv}")
    print(f"  {overall_csv}")
    if live_records:
        print(f"  {os.path.join(output_dir, 'records_all.csv')}")
        print(f"  {os.path.join(output_dir, 'records_all.jsonl')}")


def load_merged_list(output_dir, prefix):
    merged = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(output_dir, f"{prefix}_*.pkl"))):
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            for k, v in data.items():
                merged[k].extend(v)
        except Exception:  # noqa: BLE001
            pass
    return merged


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "results/example_goatbench_gpt-5.4-mini"
    main(out)
