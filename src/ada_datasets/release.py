"""Resumable, deterministic curation jobs on shared private Linux storage."""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import ctypes
from dataclasses import asdict
import fcntl
import json
from itertools import repeat
import multiprocessing
import os
from pathlib import Path
import re
import tempfile

from . import __version__
from .core import CurationConfig, digest, file_digest, write_jsonl
from .lineage import read_tombstones
from .release_dedup import CorpusIndex


FORMAT = "ada-training-release"
SPLITS = ("train", "validation", "test")
TASK_ARTIFACTS = ("candidates.jsonl", "audit.jsonl", "suppression.jsonl")
AUXILIARY_PINS = ("policy", "holdout", "denylist", "tokenizer", "language_model", "quality_model")
_SHA = re.compile(r"[a-f0-9]{64}\Z")
_SOURCE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


def _sha(value: object) -> bool:
    return isinstance(value, str) and _SHA.fullmatch(value) is not None


def _json(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                       separators=(",", ":")) + "\n").encode()


def _read(path: Path) -> dict:
    return _object(path.read_bytes())


def _object(raw: bytes) -> dict:
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object")
    return data


def _read_rows(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as source:
        for line in source:
            yield json.loads(line)


@contextmanager
def _lock(path: Path):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def _atomic_file(path: Path, body: bytes) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as staged:
        temp = Path(staged.name)
        try:
            staged.write(body)
            staged.flush()
            os.fsync(staged.fileno())
            os.replace(temp, path)
            _sync_directory(path.parent)
        finally:
            temp.unlink(missing_ok=True)


def _sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _publish(stage: Path, destination: Path) -> None:
    # Linux atomic no-replace semantics also protect a previously empty output.
    for folder, _, names in os.walk(stage, topdown=False):
        for name in names:
            with (Path(folder) / name).open("rb") as stream:
                os.fsync(stream.fileno())
        _sync_directory(Path(folder))
    libc = ctypes.CDLL(None, use_errno=True)
    rename = libc.renameat2
    rename.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(stage), -100, os.fsencode(destination), 1) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), destination)
    _sync_directory(destination.parent)


def _pin(value: object, root: Path, name: str) -> dict:
    model_keys = {"license"} if name in {"language_model", "quality_model"} else set()
    if name == "quality_model":
        model_keys.add("calibration")
    if (not isinstance(value, dict) or set(value) != {"path", "sha256"} | model_keys
            or not isinstance(value["path"], str) or not value["path"]
            or not _sha(value["sha256"])):
        raise ValueError(f"invalid {name} artifact pin")
    return {**value, "path": str((root / value["path"]).resolve())}


def load_recipe(path: Path) -> dict:
    recipe = _read(path)
    required = {"schema_version", "inputs", "policy", "tokenizer", "sequence_length",
                "shard_rows", "token_budgets", "curation", "release_use"}
    if (not required <= recipe.keys() or recipe.keys() - required - set(AUXILIARY_PINS)
            or type(recipe["schema_version"]) is not int or recipe["schema_version"] != 1):
        raise ValueError("unsupported or incomplete release recipe")
    if not isinstance(recipe["inputs"], list) or not recipe["inputs"]:
        raise ValueError("release requires at least one input shard")
    inputs = [_pin(x, path.resolve().parent, "input") for x in recipe["inputs"]]
    if len({x["sha256"] for x in inputs}) != len(inputs):
        raise ValueError("release input shard digests must be unique")
    if (type(recipe["sequence_length"]) is not int or not 2 <= recipe["sequence_length"] <= 65536
            or type(recipe["shard_rows"]) is not int or not 1 <= recipe["shard_rows"] <= 100000
            or not isinstance(recipe["token_budgets"], dict)
            or any(not isinstance(key, str) or type(value) is not int or value < 0
                   for key, value in recipe["token_budgets"].items())):
        raise ValueError("invalid release packing configuration")
    if recipe["release_use"] not in ("internal", "commercial", "redistribution"):
        raise ValueError("invalid release_use")
    if not isinstance(recipe["curation"], dict):
        raise ValueError("curation configuration must be an object")
    try:
        config = CurationConfig(**recipe["curation"])
        _json(asdict(config))
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid release curation configuration") from exc
    if config.as_of_date is None:
        raise ValueError("release requires a pinned curation.as_of_date")
    if (not isinstance(config.supported_languages, (tuple, list)) or not config.supported_languages
            or any(lang not in ("en", "ta", "hi") for lang in config.supported_languages)
            or len(config.supported_languages) != len(set(config.supported_languages))):
        raise ValueError("invalid supported languages")
    pinned = {key: _pin(recipe[key], path.resolve().parent, key) for key in AUXILIARY_PINS
              if key in recipe and recipe[key] is not None}
    if not {"policy", "tokenizer"} <= pinned.keys():
        raise ValueError("release requires policy and tokenizer")
    return {**recipe, **pinned, "inputs": inputs, "curation": json.loads(_json(asdict(config)))}


def _public_recipe(recipe: dict) -> dict:
    def strip(pin):
        return {key: value for key, value in pin.items() if key != "path"}
    fields = {k: v for k, v in recipe.items() if k not in (*AUXILIARY_PINS, "inputs")}
    fields["inputs"] = [strip(pin) for pin in recipe["inputs"]]
    return {**fields, **{key: strip(recipe[key]) for key in AUXILIARY_PINS if recipe.get(key)}}


def _check_pin(pin: dict, name: str) -> Path:
    path = Path(pin["path"])
    if not path.is_file() or file_digest(path) != pin["sha256"]:
        raise ValueError(f"{name} artifact digest mismatch")
    return path


def _tombstone_scope(pin: dict | None) -> dict:
    ids, shas = read_tombstones(_check_pin(pin, "denylist")) if pin else (set(), set())
    return {"doc_id_sha256": sorted(digest(doc_id.encode()) for doc_id in ids),
            "sha256": sorted(shas)}


def _resolved_tombstones(recipe: dict, work: Path, parent: Path) -> dict:
    base_pin = recipe.get("denylist")
    if not base_pin:
        raise ValueError("successor release requires a cumulative denylist")
    ids, shas = read_tombstones(_check_pin(base_pin, "denylist"))
    ids_hashed = {digest(doc_id.encode()) for doc_id in ids}
    old = _read(parent / "tombstone-scope.json")
    if not set(old["doc_id_sha256"]) <= ids_hashed:
        raise ValueError("successor denylist drops prior document tombstones")
    shas.update(old["sha256"])
    for row in _read_rows(parent / "aliases.jsonl"):
        if row["doc_id_sha256"] in ids_hashed:
            shas.add(row["sha256"])
    if ids_hashed == set(old["doc_id_sha256"]) and shas == set(old["sha256"]):
        raise ValueError("successor denylist introduces no new tombstones")
    rows = [{"doc_id": value} for value in sorted(ids)]
    rows.extend({"sha256": value} for value in sorted(shas))
    body = b"".join(_json(row) for row in rows)
    checksum = digest(body)
    dest = work / f"denylist-{checksum}.jsonl"
    _atomic_file(dest, body)
    return {"path": str(dest), "sha256": checksum}


def _setup(recipe_path: Path, work_dir: Path, parent: Path | None) -> dict:
    recipe = load_recipe(recipe_path)
    parent_hash = None
    effective_denylist = recipe.get("denylist")
    if parent is not None:
        parent_hash = verify_release(parent)["manifest_sha256"]
        effective_denylist = _resolved_tombstones(recipe, work_dir, parent)
    stable_recipe = _public_recipe(recipe)
    run_key = digest(_json({
        "curator_version": __version__, "recipe": stable_recipe, "parent": parent_hash,
        "effective_denylist": effective_denylist["sha256"] if effective_denylist else None,
    }))
    state = {
        "run_key": run_key, "recipe": recipe, "parent_manifest_sha256": parent_hash,
        "effective_denylist": effective_denylist, "curator_version": __version__,
    }
    descriptor = work_dir / "run.json"
    if descriptor.exists():
        if _read(descriptor) != state:
            raise ValueError("work directory already belongs to a different release recipe")
    else:
        _atomic_file(descriptor, _json(state))
    return state


def _task_path(work_dir: Path, task_id: int) -> Path:
    return work_dir / f"task-{task_id:07d}"


def _receipt(task_dir: Path, state: dict, task_id: int) -> dict:
    result = _read(task_dir / "receipt.json")
    artifacts = result.get("artifacts")
    if (result.get("run_key") != state["run_key"] or result.get("task_id") != task_id
            or not isinstance(artifacts, dict) or set(artifacts) != set(TASK_ARTIFACTS)
            or any(not _sha(checksum) or file_digest(task_dir / name) != checksum
                   for name, checksum in artifacts.items())):
        raise ValueError(f"task {task_id} has a corrupt receipt")
    return result


def run_worker(work_dir: Path, task_id: int) -> dict:
    from .release_intake import prepare_inputs

    work_dir = work_dir.resolve()
    state = _read(work_dir / "run.json")
    recipe = state["recipe"]
    if state["curator_version"] != __version__:
        raise ValueError("worker requires the same version as the coordinator")
    if type(task_id) is not int or not 0 <= task_id < len(recipe["inputs"]):
        raise ValueError("task identifier is outside the run")
    destination = _task_path(work_dir, task_id)
    with _lock(work_dir / f"task-{task_id:07d}.lock"):
        if destination.exists():
            return _receipt(destination, state, task_id)
        input_pin = recipe["inputs"][task_id]
        pins = {key: recipe[key] for key in AUXILIARY_PINS if recipe.get(key) and key != "denylist"}
        if state["effective_denylist"]:
            pins["denylist"] = state["effective_denylist"]
        checked = {key: _check_pin(pin, key) for key, pin in pins.items() if key != "tokenizer"}
        with tempfile.TemporaryDirectory(prefix=f".attempt-{task_id}-", dir=work_dir) as temp:
            stage = Path(temp) / "result"
            stats = prepare_inputs(
                Path(input_pin["path"]), stage, recipe=recipe, policy_path=checked["policy"],
                holdout_path=checked.get("holdout"), denylist_path=checked.get("denylist"),
                input_sha256=input_pin["sha256"],
            )
            for key in checked:
                _check_pin(pins[key], key)
            result = {"run_key": state["run_key"], "task_id": task_id, "stats": stats,
                      "artifacts": {name: file_digest(stage / name) for name in TASK_ARTIFACTS}}
            _atomic_file(stage / "receipt.json", _json(result))
            _publish(stage, destination)
        return result


def _verify_content_free(directory: Path, manifest: dict) -> None:
    keys = {"input_index", "line", "input_line_sha256", "source_id", "decision", "reason"}
    input_counts = manifest["input_line_counts"]
    if (not isinstance(input_counts, list) or len(input_counts) != len(manifest["recipe"]["inputs"])
            or any(type(count) is not int or count < 0 for count in input_counts)):
        raise ValueError("invalid input line counts")
    reasons = Counter()
    for row in _read_rows(directory / "audit.jsonl"):
        if (not isinstance(row, dict) or set(row) != keys or row["decision"] != "rejected"
                or type(row["input_index"]) is not int or not 0 <= row["input_index"] < len(input_counts)
                or type(row["line"]) is not int or not 1 <= row["line"] <= input_counts[row["input_index"]]
                or not _sha(row["input_line_sha256"])
                or (row["source_id"] is not None
                    and (not isinstance(row["source_id"], str) or not _SOURCE.fullmatch(row["source_id"])))
                or not isinstance(row["reason"], str)
                or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", row["reason"])):
            raise ValueError("audit contains an invalid or unsafe record")
        reasons[row["reason"]] += 1
    for row in _read_rows(directory / "aliases.jsonl"):
        if not isinstance(row, dict) or set(row) != {"doc_id_sha256", "sha256"} or not all(
                _sha(value) for value in row.values()):
            raise ValueError("invalid content-free alias record")
    scope = _read(directory / "tombstone-scope.json")
    if (set(scope) != {"doc_id_sha256", "sha256"}
            or any(not isinstance(values, list) or not all(_sha(value) for value in values)
                   or values != sorted(set(values)) for values in scope.values())):
        raise ValueError("invalid tombstone commitments")
    totals = manifest["totals"]
    accepted = sum(x["document_count"] for x in manifest["splits"].values())
    excluded = manifest["composition"]["budget_excluded_documents"]
    if totals != {"input_lines": sum(input_counts), "accepted": accepted,
                  "rejected": sum(reasons.values()), "budget_excluded": excluded,
                  "rejection_reasons": dict(reasons)} or sum(input_counts) != (
                      accepted + excluded + sum(reasons.values())):
        raise ValueError("release totals do not reconcile with its artifacts")


def verify_release(directory: Path, expected_manifest_sha256: str | None = None) -> dict:
    from .release_pack import verify_pack

    path = directory / "manifest.json"
    if directory.is_symlink() or path.is_symlink() or not path.is_file():
        raise ValueError("missing regular release manifest")
    raw = path.read_bytes()
    checksum = digest(raw)
    if expected_manifest_sha256 is not None and checksum != expected_manifest_sha256:
        raise ValueError("release manifest digest mismatch")
    manifest = _object(raw)
    if manifest.get("format") != FORMAT or manifest.get("schema_version") != 2:
        raise ValueError("unsupported release manifest")
    lineage = manifest.get("lineage")
    if (not isinstance(lineage, dict) or set(lineage) != {"parent_manifest_sha256"}
            or (lineage["parent_manifest_sha256"] is not None
                and not _sha(lineage["parent_manifest_sha256"]))):
        raise ValueError("invalid release lineage")
    artifacts = manifest["artifacts"]
    base = {"audit.jsonl", "aliases.jsonl", "tombstone-scope.json", "DATASET_CARD.md"}
    if lineage["parent_manifest_sha256"] is not None:
        base.add("revocation-impact.json")
    if not isinstance(artifacts, dict) or not base <= artifacts.keys():
        raise ValueError("release is missing required artifacts")
    for name, value in artifacts.items():
        relative = Path(name)
        target = directory / relative
        if (relative.is_absolute() or ".." in relative.parts or relative.as_posix() != name
                or name == "manifest.json"
                or target.resolve() != directory.resolve() / relative
                or not target.is_file() or not _sha(value) or file_digest(target) != value):
            raise ValueError("release artifact digest mismatch or invalid path")
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
    if actual != {"manifest.json", *artifacts}:
        raise ValueError("release contains undeclared artifacts")
    if "revocation-impact.json" in base:
        impact = _read(directory / "revocation-impact.json")
        if (set(impact) != {"matched_documents", "document_artifacts", "block_artifacts", "document_sha256"}
                or type(impact["matched_documents"]) is not int or impact["matched_documents"] < 0
                or any(not isinstance(impact[key], list) or not all(
                    isinstance(name, str) and re.fullmatch(
                        rf"{kind}/(?:train|validation|test)/part-[0-9]{{6}}\.parquet", name
                    ) for name in impact[key])
                    for key, kind in (("document_artifacts", "documents"), ("block_artifacts", "blocks")))
                or not isinstance(impact["document_sha256"], list)
                or not all(_sha(value) for value in impact["document_sha256"])):
            raise ValueError("invalid revocation artifact impact")
    _verify_content_free(directory, manifest)
    verify_pack(directory, manifest)
    if file_digest(path) != checksum:
        raise ValueError("release manifest changed during verification")
    return {"valid": True, "docs": manifest["totals"]["accepted"], "manifest_sha256": checksum,
            "curator_version": manifest["curator_version"]}


def _card(stage: Path, manifest: dict) -> None:
    # The private, typed documents retain per-source attribution; contract
    # details, source reviews and raw input are not included in this summary.
    lines = ["# ADA training release", "", f"Use scope {manifest['recipe']['release_use']}.",
             f"Rights reviewed as of {manifest['recipe']['curation']['as_of_date']}.",
             "", "Per-document Parquet provenance contains the attribution and license fields.",
             "Operator source review and independent evaluation govern approved uses.", "",
             "| Split | Documents | Active training targets |", "| --- | ---: | ---: |"]
    for split in SPLITS:
        part = manifest["splits"][split]
        lines.append(f"| {split} | {part['document_count']} | {part['active_targets']} |")
    lines.extend(("", "Candidate-based near-duplicate matching can miss pairs.",
                  "Privacy, language, and benchmark gates require evaluation on reviewed gold sets."))
    (stage / "DATASET_CARD.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _assemble(work: Path, stage: Path, state: dict, impact: dict | None) -> dict:
    from .release_pack import pack_release

    recipe = state["recipe"]
    inputs = recipe["inputs"]
    receipts = [_receipt(_task_path(work, i), state, i) for i in range(len(inputs))]
    directories = [_task_path(work, i) for i in range(len(inputs))]
    tokenizer_path = _check_pin(recipe["tokenizer"], "tokenizer")
    input_counts = [result["stats"]["lines"] for result in receipts]
    with tempfile.TemporaryDirectory(prefix=".merge-", dir=work) as scratch:
        with CorpusIndex(Path(scratch) / "merge.sqlite", CurationConfig(**recipe["curation"])) as index:
            index.ingest(directories, [pin["sha256"] for pin in inputs])
            index.assign()
            with (stage / "audit.jsonl").open("w", encoding="utf-8") as audit:
                index.write_audit(audit, directories, stage / "aliases.jsonl")
            pack = pack_release(
                stage, {split: index.rows(split) for split in SPLITS},
                tokenizer_path=tokenizer_path, tokenizer_sha256=recipe["tokenizer"]["sha256"],
                sequence_length=recipe["sequence_length"], shard_rows=recipe["shard_rows"],
                token_budgets=recipe["token_budgets"],
            )
    (stage / "tombstone-scope.json").write_bytes(_json(_tombstone_scope(state["effective_denylist"])))
    reasons = Counter(row["reason"] for row in _read_rows(stage / "audit.jsonl"))
    totals = {"input_lines": sum(input_counts),
              "accepted": sum(x["document_count"] for x in pack["splits"].values()),
              "rejected": sum(reasons.values()),
              "budget_excluded": pack["composition"]["budget_excluded_documents"],
              "rejection_reasons": dict(reasons)}
    stable = _public_recipe(recipe)
    manifest = {
        **pack, "schema_version": 2, "format": FORMAT, "curator_version": __version__,
        "run_key": state["run_key"], "recipe": stable, "totals": totals,
        "input_line_counts": input_counts,
        "effective_denylist_sha256": (state["effective_denylist"]["sha256"]
                                      if state["effective_denylist"] else None),
        "lineage": {"parent_manifest_sha256": state["parent_manifest_sha256"]},
    }
    _card(stage, manifest)
    for name in ("audit.jsonl", "aliases.jsonl", "tombstone-scope.json", "DATASET_CARD.md"):
        manifest["artifacts"][name] = file_digest(stage / name)
    if impact is not None:
        impact_file = stage / "revocation-impact.json"
        impact_file.write_bytes(_json(impact))
        manifest["artifacts"][impact_file.name] = file_digest(impact_file)
    (stage / "manifest.json").write_bytes(_json(manifest))
    return verify_release(stage)


def run_release(recipe_path: Path, work_dir: Path, output: Path, *, workers: int = 1,
                prepare_only: bool = False, parent: Path | None = None,
                status: Path | None = None) -> dict:
    from .release_governance import artifact_impact, quarantine_release, register_release

    if not 0 <= workers <= 64:
        raise ValueError("worker count must be within 0..64")
    if parent is not None and status is None:
        raise ValueError("successor requires the private release status registry")
    work = work_dir.resolve()
    work.mkdir(mode=0o700, parents=True, exist_ok=True)
    output = output.absolute()
    with _lock(work / "coordinator.lock"):
        state = _setup(recipe_path, work, parent)
        parent_hash = state["parent_manifest_sha256"]
        task_ids = range(len(state["recipe"]["inputs"]))
        if prepare_only:
            return {"prepared": True, "tasks": len(task_ids), "run_key": state["run_key"]}
        if output.exists():
            result = verify_release(output)
            if _read(output / "manifest.json")["run_key"] != state["run_key"]:
                raise FileExistsError("output already contains a different release")
        else:
            impact = None
            if parent_hash:
                quarantine_release(status, parent_hash, state["effective_denylist"]["sha256"])
                impact = artifact_impact(parent, _read(parent / "manifest.json"),
                                         Path(state["effective_denylist"]["path"]))
            if workers == 1:
                for task_id in task_ids:
                    run_worker(work, task_id)
            elif workers > 1:
                context = multiprocessing.get_context("spawn")
                with ProcessPoolExecutor(max_workers=min(workers, len(task_ids)), mp_context=context) as pool:
                    list(pool.map(run_worker, repeat(work), task_ids))
            output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".ada-release-", dir=output.parent) as temp:
                stage = Path(temp) / "release"
                stage.mkdir(mode=0o700)
                result = _assemble(work, stage, state, impact)
                _publish(stage, output)
        if status is not None:
            registered = register_release(status, result["manifest_sha256"], parent_sha256=parent_hash,
                                          replacement=parent_hash is not None)
            if not registered["active"]:
                raise ValueError("published release is quarantined")
        return result
