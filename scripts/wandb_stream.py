#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def _load_config(path: str) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


_MIN_FREE_BYTES = 1 * 1024 * 1024 * 1024  # 1 GiB — warn below this threshold

_STEP_METRIC = "global_step"
_WANDB_SECTIONS_KEY = "_wandb_sections"
_WANDB_SECTION_ORDER_KEY = "_wandb_section_order"


def _disable_auto_panel_generation(wandb: Any) -> None:
    """Keep W&B from auto-adding duplicate panels for every logged scalar.

    Pulsar emits its workspace section metadata with each metrics payload.
    Marking streamed metrics as hidden prevents W&B's automatic panel generator
    from creating another line plot for the same key on later runs. The values
    are still logged normally and existing/manual panels keep working.
    """
    auto_panels = os.environ.get("PULSAR_WANDB_AUTO_PANELS", "").strip().lower()
    hidden = auto_panels not in {"1", "true", "yes", "on"}

    wandb.define_metric(_STEP_METRIC, hidden=True, summary="last")
    wandb.define_metric("update", step_metric=_STEP_METRIC, hidden=hidden, summary="last")
    wandb.define_metric("*", step_metric=_STEP_METRIC, hidden=hidden, summary="last")


def _env_enabled(name: str, default: bool) -> bool:
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


def _stable_id(prefix: str, value: str) -> str:
    return hashlib.sha1(f"pulsar:{prefix}:{value}".encode("utf-8")).hexdigest()[:9]


def _workspace_view_query(gql: Any) -> Any:
    return gql(
        """
        query Views($entityName:String,$name:String,$viewType:String="project-view") {
          project(name:$name, entityName:$entityName) {
            allViews(viewType:$viewType) {
              edges { node { id name displayName spec } }
            }
          }
        }
        """
    )


def _workspace_upsert_query(gql: Any) -> Any:
    return gql(
        """
        mutation UpsertView2(
          $id: ID,
          $entityName: String,
          $projectName: String,
          $type: String,
          $name: String,
          $displayName: String,
          $description: String,
          $spec: String
        ) {
          upsertView(input: {
            id: $id,
            entityName: $entityName,
            projectName: $projectName,
            name: $name,
            displayName: $displayName,
            description: $description,
            type: $type,
            spec: $spec,
            createdUsing: WANDB_SDK
          }) {
            view { id name }
            inserted
          }
        }
        """
    )


def _panel_metrics(panel: dict[str, Any]) -> tuple[str, ...]:
    metrics = panel.get("config", {}).get("metrics", []) or []
    return tuple(metric for metric in metrics if isinstance(metric, str))


def _panel_key(panel: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    return panel.get("viewType", ""), _panel_metrics(panel)


def _line_panel(metric: str) -> dict[str, Any]:
    return {
        "viewType": "Run History Line Plot",
        "config": {
            "metrics": [metric],
            "groupBy": "None",
            "legendFields": ["run:displayName"],
        },
        "__id__": _stable_id("panel", metric),
        "isAuto": False,
    }


def _classify_panel(panel: dict[str, Any], original_section: str, metric_sections: dict[str, str]) -> str:
    metrics = _panel_metrics(panel)
    for metric in metrics:
        section = metric_sections.get(metric)
        if section:
            return section
    return original_section or "Charts"


def _base_workspace_section(name: str, existing: dict[str, dict[str, Any]]) -> dict[str, Any]:
    section = dict(existing.get(name, {}))
    section["__id__"] = section.get("__id__") or _stable_id("section", name)
    section["name"] = name
    section["isOpen"] = section.get("isOpen", name != "Hidden Panels")
    section["sorted"] = section.get("sorted", 0)
    section["pinned"] = section.get("pinned", False)
    section["isPanelsAuto"] = False
    section["defaultName"] = section.get("defaultName", name)
    section["panels"] = []
    return section


def _organize_workspace_spec(
    spec: dict[str, Any],
    metric_sections: dict[str, str],
    section_order: list[str],
) -> tuple[dict[str, Any], Counter[str]]:
    panel_bank = spec["section"]["panelBankConfig"]
    existing_sections = {
        section.get("name", ""): section
        for section in panel_bank.get("sections", [])
        if section.get("name")
    }
    ordered_names = []
    for name in section_order:
        if name and name not in ordered_names:
            ordered_names.append(name)
    for name in existing_sections:
        if name not in ordered_names:
            ordered_names.append(name)
    if "Charts" not in ordered_names:
        ordered_names.append("Charts")
    if "System" not in ordered_names:
        ordered_names.append("System")

    organized = {
        name: _base_workspace_section(name, existing_sections)
        for name in ordered_names
    }

    counts: Counter[str] = Counter()
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for section in panel_bank.get("sections", []):
        original_name = section.get("name", "Charts")
        for panel in section.get("panels", []):
            key = _panel_key(panel)
            if key[1] and key in seen:
                continue
            if key[1]:
                seen.add(key)
            target = _classify_panel(panel, original_name, metric_sections)
            if target not in organized:
                organized[target] = _base_workspace_section(target, existing_sections)
                ordered_names.append(target)
            organized[target]["panels"].append(panel)
            counts[target] += 1

    existing_metric_panels = {
        metric
        for section in organized.values()
        for panel in section.get("panels", [])
        for metric in _panel_metrics(panel)
    }
    for metric, target in metric_sections.items():
        if metric in existing_metric_panels:
            continue
        if target not in organized:
            organized[target] = _base_workspace_section(target, existing_sections)
            ordered_names.append(target)
        organized[target]["panels"].append(_line_panel(metric))
        counts[target] += 1
        existing_metric_panels.add(metric)

    panel_bank["sections"] = [
        organized[name]
        for name in ordered_names
        if organized[name]["panels"] or name in {"Charts", "System"}
    ]
    return spec, counts


def _organize_wandb_workspace(
    wandb: Any,
    entity: str | None,
    project: str,
    mode: str,
    metric_sections: dict[str, str],
    section_order: list[str],
) -> None:
    if not _env_enabled("PULSAR_WANDB_ORGANIZE_WORKSPACE", True):
        return
    if mode == "offline":
        return
    if not entity or not metric_sections:
        return

    try:
        from wandb_gql import gql

        api = wandb.Api()
        node = api.client.execute(
            _workspace_view_query(gql),
            {"entityName": entity, "name": project, "viewType": "project-view"},
        )["project"]["allViews"]["edges"][0]["node"]
        spec, counts = _organize_workspace_spec(json.loads(node["spec"]), metric_sections, section_order)
        api.client.execute(
            _workspace_upsert_query(gql),
            {
                "id": node["id"],
                "entityName": entity,
                "projectName": project,
                "type": "project-view",
                "name": node["name"],
                "displayName": node["displayName"],
                "description": "",
                "spec": json.dumps(spec, separators=(",", ":")),
            },
        )
        count_text = ", ".join(f"{name}={count}" for name, count in counts.items())
        print(f"wandb workspace organized: {count_text}", file=sys.stderr)
    except Exception as exc:
        print(f"wandb workspace organization skipped: {exc}", file=sys.stderr)


def _check_disk_space(dir_path: str) -> None:
    """Verify at least _MIN_FREE_BYTES are available on the filesystem containing dir_path.

    Exits gracefully with a clear message when disk is too full, avoiding the cascade of
    nested OSErrors that happens when logging / wandb operations fail mid-flight.
    """
    usage = shutil.disk_usage(os.path.dirname(os.path.abspath(dir_path)) or dir_path)
    free_gb = usage.free / (1024**3)
    if usage.free < _MIN_FREE_BYTES:
        print(
            f"FATAL: Disk is full on {dir_path}. "
            f"Free: {free_gb:.1f} GiB, required: {_MIN_FREE_BYTES / (1024**3):.0f} GiB. "
            f"Free up space and retry.",
            file=sys.stderr,
        )
        sys.exit(1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stream JSON metrics from Pulsar C++ loops into Weights & Biases.")
    parser.add_argument("--project", required=True)
    parser.add_argument("--entity", default="")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--group", default="")
    parser.add_argument("--job-type", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--wandb-dir", required=True)
    parser.add_argument("--mode", default="online")
    parser.add_argument("--config-path", default="")
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--id", default="")
    parser.add_argument("--resume", default="")
    args = parser.parse_args()

    _check_disk_space(args.wandb_dir)

    try:
        import wandb
    except ImportError as exc:
        raise SystemExit(
            "wandb is required for Pulsar W&B logging. Install it with `pip install wandb` or `pip install -e .[offline]`."
        ) from exc

    run = wandb.init(
        project=args.project,
        entity=args.entity or None,
        name=args.run_name,
        group=args.group or None,
        job_type=args.job_type,
        dir=args.wandb_dir,
        mode=args.mode,
        tags=args.tag or None,
        config=_load_config(args.config_path) if args.config_path else None,
        id=args.id or None,
        resume=args.resume or "allow",
        reinit="finish_previous",
    )
    if run is None:
        return 0
    _disable_auto_panel_generation(wandb)

    run_id_file = os.path.join(args.run_dir, "wandb_run_id.txt")
    try:
        with open(run_id_file, "w") as f:
            f.write(run.id)
    except OSError:
        pass

    workspace_signature: tuple[tuple[tuple[str, str], ...], tuple[str, ...]] | None = None
    try:
        for line in sys.stdin:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            metric_sections_raw = payload.pop(_WANDB_SECTIONS_KEY, None)
            section_order_raw = payload.pop(_WANDB_SECTION_ORDER_KEY, None)
            if isinstance(metric_sections_raw, dict):
                metric_sections = {
                    str(metric): str(section)
                    for metric, section in metric_sections_raw.items()
                    if isinstance(metric, str) and isinstance(section, str) and metric and section
                }
                section_order = [
                    str(section)
                    for section in (section_order_raw if isinstance(section_order_raw, list) else [])
                    if isinstance(section, str) and section
                ]
                signature = (tuple(sorted(metric_sections.items())), tuple(section_order))
                if signature != workspace_signature:
                    _organize_wandb_workspace(
                        wandb,
                        args.entity or getattr(run, "entity", None),
                        args.project,
                        args.mode,
                        metric_sections,
                        section_order,
                    )
                    workspace_signature = signature
            step = payload.pop("_step", None)
            wandb.log(payload, step=step)
    finally:
        wandb.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
