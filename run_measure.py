#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
요추 측면 X-ray 압박률 측정기 (Claude API 연결용)

- measure_prompt.txt(측정 프롬프트)를 시스템 프롬프트로 넣고 사진을 올린다.
- Claude가 코드 실행 도구로 추체 모서리와 전방·후방 높이를 잰다.
- 새 대화로 여러 번(기본 3회) 독립 실행하고, 높이를 평균한 뒤
  논문(손광현, 2010)의 방법 1~4를 이 프로그램이 직접 계산한다.

준비:  pip install -U anthropic pillow        (DICOM을 넣을 때만 pydicom 추가)
       환경변수 ANTHROPIC_API_KEY 설정

예:
  python run_measure.py I0000001.jpg I0000002.jpg I0000003.jpg --targets L2 L3
  python run_measure.py 사진.jpg --targets L2 L3 --runs 3 --model claude-opus-5-5
  python run_measure.py 사진.jpg --targets L2 L3 --dry-run        (API 호출 없이 준비만 확인)
  python run_measure.py --aggregate-only results/20260928_101500  (저장된 결과로 표만 다시 계산)
"""
import argparse
import csv
import datetime as dt
import json
import math
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

DEFAULT_MODEL = "claude-opus-5-5"
CODE_TOOL = {"type": "code_execution_20250825", "name": "code_execution"}
# 100만 토큰당 입력/출력 요금(USD). 2026-09 문서 기준. 바뀌면 여기만 고치면 됨.
PRICES = {
    "claude-fable-5-1": (10.0, 50.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
}
LEVELS = ["T10", "T11", "T12", "L1", "L2", "L3", "L4", "L5", "S1"]
MAX_STEPS = 15  # pause_turn 이어받기 포함 한 번 측정당 최대 API 호출 수


# ---------------------------------------------------------------- 레벨 도우미
def neighbors(level):
    i = LEVELS.index(level)
    up = LEVELS[i - 1] if i > 0 else None
    down = LEVELS[i + 1] if i + 1 < len(LEVELS) else None
    return up, down


def levels_to_measure(targets):
    s = set(targets)
    for t in targets:
        s.update(x for x in neighbors(t) if x)
    return [l for l in LEVELS if l in s]


# ---------------------------------------------------------------- 이미지 준비
def load_image(path):
    from PIL import Image
    path = Path(path)
    is_dcm = path.suffix.lower() in (".dcm", ".dicom")
    if not is_dcm:
        with open(path, "rb") as f:
            head = f.read(132)
        is_dcm = head[128:132] == b"DICM"
    if is_dcm:
        import numpy as np
        import pydicom
        try:
            from pydicom.pixels import apply_voi_lut
        except ImportError:
            from pydicom.pixel_data_handlers.util import apply_voi_lut
        ds = pydicom.dcmread(str(path))
        try:
            arr = apply_voi_lut(ds.pixel_array, ds).astype("float64")
        except Exception:
            arr = ds.pixel_array.astype("float64")
        if arr.ndim == 3:  # 다중 프레임이면 첫 장
            arr = arr[0]
        if str(getattr(ds, "PhotometricInterpretation", "")) == "MONOCHROME1":
            arr = arr.max() - arr
        lo, hi = np.percentile(arr, [0.5, 99.5])
        arr = np.clip((arr - lo) / max(hi - lo, 1e-9), 0, 1) * 255
        return Image.fromarray(arr.astype("uint8"), "L")
    return Image.open(path).convert("L")


def resized_size(width, height, max_edge, max_tokens):
    """Claude가 내부에서 줄이는 크기 (공식 문서 vision-coordinates의 계산식)."""
    def fits(w, h):
        return (math.ceil(w / 28) * 28 <= max_edge and math.ceil(h / 28) * 28 <= max_edge
                and math.ceil(w / 28) * math.ceil(h / 28) <= max_tokens)
    if fits(width, height):
        return width, height
    if height > width:
        h, w = resized_size(height, width, max_edge, max_tokens)
        return w, h
    ar = width / height
    lo, hi = 1, width
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if fits(mid, max(round(mid / ar), 1)):
            lo = mid
        else:
            hi = mid
    return lo, max(round(lo / ar), 1)


def vision_limits(model):
    # Claude 4.7 이후 모델은 고해상도(긴 변 2576px, 4784토큰), 그 외(Haiku 4.5 등)는 표준(1568px, 1568토큰)
    return (1568, 1568) if "haiku" in model else (2576, 4784)


def prepare_images(paths, model, out_dir):
    """사진을 모델이 실제로 보는 크기로 미리 줄여 PNG로 저장.
    이렇게 하면 모델이 눈으로 보는 좌표와 코드로 재는 좌표가 같은 픽셀 공간이 된다."""
    from PIL import Image
    max_edge, max_tokens = vision_limits(model)
    (out_dir / "inputs").mkdir(parents=True, exist_ok=True)
    films = []
    for i, p in enumerate(paths, 1):
        im = load_image(p)
        w, h = im.size
        nw, nh = resized_size(w, h, max_edge, max_tokens)
        if (nw, nh) != (w, h):
            im = im.resize((nw, nh), Image.LANCZOS)
        name = f"film{i}.png"
        dst = out_dir / "inputs" / name
        im.save(dst)
        films.append({"name": name, "source": str(Path(p).name), "orig_size": [w, h],
                      "sent_size": [nw, nh], "local_path": str(dst)})
    return films


# ---------------------------------------------------------------- 요청 구성
def submit_tool():
    point = {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2,
             "description": "[x, y] 업로드된 파일 기준 픽셀"}
    vert = {
        "type": "object",
        "properties": {
            "level": {"type": "string", "description": "예: L2"},
            "measurable": {"type": "boolean", "description": "윤곽이 불명확해 못 재면 false"},
            "AS": point, "AI": point, "PS": point, "PI": point,
            "AVH": {"type": "number", "description": "전방 높이(px)"},
            "PVH": {"type": "number", "description": "후방 높이(px)"},
            "note": {"type": "string"},
        },
        "required": ["level", "measurable"],
    }
    return {
        "name": "submit_measurement",
        "description": "측정이 모두 끝난 뒤 딱 한 번 호출해 결과를 제출한다. "
                       "좌표·높이는 업로드된 파일의 픽셀 기준이다. 압박률 계산은 프로그램이 한다.",
        "input_schema": {
            "type": "object",
            "properties": {
                "level_basis": {"type": "string", "description": "레벨 판단 근거"},
                "films": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "file": {"type": "string", "description": "예: film1.png"},
                            "vertebrae": {"type": "array", "items": vert},
                        },
                        "required": ["file", "vertebrae"],
                    },
                },
                "notes": {"type": "string",
                          "description": "측정 조건, 논문 절차와 달라진 점, 확인된 사실과 추정의 구분"},
            },
            "required": ["level_basis", "films", "notes"],
        },
    }


def build_user_content(films, targets):
    levels = levels_to_measure(targets)
    content = [{"type": "text", "text": (
        f"요추 측면 사진 {len(films)}장을 보낸다.\n"
        f"대상 골절 추체: {', '.join(targets)}\n"
        f"사진마다 잴 추체(대상 + 위·아래 인접): {', '.join(levels)}\n")}]
    for f in films:
        w, h = f["sent_size"]
        content.append({"type": "text", "text": f"[{f['name']}] 원본 {f['source']}, 크기 {w}x{h}px"})
        content.append({"type": "image", "source": {"type": "file", "file_id": f["file_id"]}})
        content.append({"type": "container_upload", "file_id": f["file_id"]})
    content.append({"type": "text", "text": (
        "[실행 규칙]\n"
        "- 위 사진들은 코드 실행 컨테이너에도 같은 파일 이름으로 올라가 있다(작업 폴더에서 못 찾으면 find로 찾는다). "
        "화면에 보이는 이미지와 파일 크기가 같으므로, 좌표와 높이는 이 파일 픽셀 기준으로 코드로 잰다.\n"
        "- 사진별로 측정점과 전방·후방 높이 선을 그린 이미지를 $OUTPUT_DIR/<파일이름>_marked.png 로 저장한다.\n"
        "- 윤곽이 겹쳐 불명확한 추체는 억지로 재지 말고 measurable=false 와 이유를 넣는다.\n"
        "- 끝나면 submit_measurement 도구를 한 번 호출해 결과를 제출한다. "
        "AVH·PVH는 보충 규칙 8의 반복 측정 평균값을 넣는다.")})
    return content


# ---------------------------------------------------------------- API 실행
def collect_file_ids(blocks):
    ids = []
    def walk(x):
        if isinstance(x, dict):
            if isinstance(x.get("file_id"), str):
                ids.append(x["file_id"])
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    for b in blocks:
        d = b.model_dump() if hasattr(b, "model_dump") else b
        if str(d.get("type", "")).endswith("tool_result"):
            walk(d.get("content"))
    return ids


def run_once(client, model, system_prompt, user_content, max_tokens, log):
    messages = [{"role": "user", "content": user_content}]
    tools = [CODE_TOOL, submit_tool()]
    container_id = None
    usage = {"input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 0,
             "cache_read_input_tokens": 0, "api_calls": 0}
    file_ids, texts, nudges = [], [], 0
    for step in range(MAX_STEPS):
        kwargs = dict(model=model, max_tokens=max_tokens, system=system_prompt,
                      tools=tools, messages=messages)
        if container_id:
            kwargs["container"] = container_id
        with client.messages.stream(**kwargs) as stream:
            resp = stream.get_final_message()
        usage["api_calls"] += 1
        for k in ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"):
            usage[k] += getattr(resp.usage, k, 0) or 0
        if getattr(resp, "container", None) is not None:
            container_id = resp.container.id
        file_ids += collect_file_ids(resp.content)
        texts += [b.text for b in resp.content if getattr(b, "type", "") == "text"]
        log(f"    호출 {step + 1}: stop_reason={resp.stop_reason}")

        submit = [b for b in resp.content
                  if getattr(b, "type", "") == "tool_use" and getattr(b, "name", "") == "submit_measurement"]
        if submit:
            return {"result": submit[0].input, "usage": usage, "file_ids": file_ids,
                    "texts": texts, "container_id": container_id}
        if resp.stop_reason == "pause_turn":
            messages.append({"role": "assistant", "content": resp.content})
            continue
        if resp.stop_reason == "max_tokens":
            log("    ※ 출력 한도(max_tokens)에 걸려 이번 측정을 멈춥니다. --max-tokens 값을 늘려 다시 실행하세요.")
            break
        if nudges >= 2:
            break
        nudges += 1
        messages.append({"role": "assistant", "content": resp.content})
        messages.append({"role": "user", "content":
                         "측정을 마무리하고 submit_measurement 도구로 결과를 제출해 주세요."})
    return {"result": None, "usage": usage, "file_ids": file_ids, "texts": texts,
            "container_id": container_id}


def est_cost(model, usage):
    if model not in PRICES:
        return None
    pin, pout = PRICES[model]
    tin = usage["input_tokens"] + usage["cache_creation_input_tokens"] + usage["cache_read_input_tokens"]
    return (tin * pin + usage["output_tokens"] * pout) / 1e6


# ---------------------------------------------------------------- 집계(논문 방식)
def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def heights(v):
    """보고된 AVH/PVH를 쓰고, 없으면 모서리 좌표로 계산."""
    avh, pvh = v.get("AVH"), v.get("PVH")
    try:
        if avh is None:
            avh = dist(v["AS"], v["AI"])
        if pvh is None:
            pvh = dist(v["PS"], v["PI"])
    except (KeyError, TypeError, IndexError):
        return None
    if not avh or not pvh:
        return None
    return float(avh), float(pvh)


def aggregate(runs, targets):
    # entry = (run번호, 사진) → {level: (AVH, PVH)}
    entries = []
    for r in runs:
        res = r.get("result") or {}
        for f in res.get("films", []):
            d = {}
            for v in f.get("vertebrae", []):
                if v.get("measurable") is False:
                    continue
                h = heights(v)
                if h:
                    d[str(v.get("level", "")).upper()] = h
            entries.append((r.get("run"), f.get("file"), d))

    def mean(xs):
        return sum(xs) / len(xs)

    formulas = {
        "M1": (lambda a, p, t, u, l: 1 - a[t] / p[t], lambda t, u, l: [t]),
        "M2": (lambda a, p, t, u, l: ((a[u] + a[l]) / 2 - a[t]) / ((a[u] + a[l]) / 2), lambda t, u, l: [t, u, l]),
        "M3": (lambda a, p, t, u, l: 1 - a[t] / a[u], lambda t, u, l: [t, u]),
        "M4": (lambda a, p, t, u, l: 1 - a[t] / a[l], lambda t, u, l: [t, l]),
    }
    table = {}
    for t in targets:
        u, l = neighbors(t)
        row = {}
        for m, (fn, need) in formulas.items():
            req = need(t, u, l)
            if None in req:
                row[m] = None
                continue
            use = [e for e in entries if all(x in e[2] for x in req)]
            if not use:
                row[m] = None
                continue
            a = {x: mean([e[2][x][0] for e in use]) for x in req}
            p = {x: mean([e[2][x][1] for e in use]) for x in req}
            val = fn(a, p, t, u, l)
            per = [fn({x: e[2][x][0] for x in req}, {x: e[2][x][1] for x in req}, t, u, l) for e in use]
            flag = [x for x in req[1:] if x in targets]
            row[m] = {"value": val, "min": min(per), "max": max(per), "n": len(use),
                      "reference_fractured": flag}
        table[t] = row
    return table, entries


def print_table(table):
    names = {"M1": "방법1 (1-AVH/PVH)", "M2": "방법2 (위아래 평균 기준)", "M3": "방법3 (위 추체 기준)",
             "M4": "방법4 (아래 추체 기준)"}
    print("\n압박률 (높이 평균 후 계산 / 괄호: 개별 측정 최소~최대 / n: 측정 건수)")
    for t, row in table.items():
        print(f"\n  [{t}]")
        for m, r in row.items():
            if r is None:
                print(f"    {names[m]} : 계산 불가(필요한 추체 측정 없음)")
                continue
            flag = f"  ※기준 추체 골절({','.join(r['reference_fractured'])})" if r["reference_fractured"] else ""
            print(f"    {names[m]} : {100 * r['value']:.1f}%  ({100 * r['min']:.1f}~{100 * r['max']:.1f}, n={r['n']}){flag}")


def save_summary(out_dir, table, entries):
    (out_dir / "summary.json").write_text(json.dumps(
        {"table": table, "entries": [{"run": e[0], "file": e[1], "heights": e[2]} for e in entries]},
        ensure_ascii=False, indent=1), encoding="utf-8")
    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["대상", "방법", "압박률(%)", "최소(%)", "최대(%)", "n", "기준 추체 골절"])
        for t, row in table.items():
            for m, r in row.items():
                if r:
                    w.writerow([t, m, round(100 * r["value"], 1), round(100 * r["min"], 1),
                                round(100 * r["max"], 1), r["n"], ",".join(r["reference_fractured"])])
        w.writerow([])
        w.writerow(["run", "사진", "추체", "AVH(px)", "PVH(px)"])
        for run, file, d in entries:
            for lv, (a, p) in sorted(d.items(), key=lambda kv: LEVELS.index(kv[0]) if kv[0] in LEVELS else 99):
                w.writerow([run, file, lv, round(a, 1), round(p, 1)])


# ---------------------------------------------------------------- 메인
def main():
    ap = argparse.ArgumentParser(description="Claude API로 요추 측면 사진 압박률 측정")
    ap.add_argument("images", nargs="*", help="측면 사진 파일(JPEG/PNG/DICOM)")
    ap.add_argument("--targets", nargs="+", default=["L2", "L3"], help="대상 골절 추체 (기본: L2 L3)")
    ap.add_argument("--runs", type=int, default=3, help="독립 측정 횟수 (기본 3)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--prompt", default=str(Path(__file__).with_name("measure_prompt.txt")))
    ap.add_argument("--out", default="results")
    ap.add_argument("--max-tokens", type=int, default=32000)
    ap.add_argument("--keep-files", action="store_true", help="API에 올린 파일을 끝나고 지우지 않음")
    ap.add_argument("--dry-run", action="store_true", help="API 호출 없이 이미지 변환과 요청 내용만 확인")
    ap.add_argument("--aggregate-only", metavar="결과폴더", help="저장된 run_*.json으로 표만 다시 계산")
    args = ap.parse_args()
    targets = [t.upper() for t in args.targets]
    for t in targets:
        if t not in LEVELS:
            ap.error(f"알 수 없는 추체: {t}")

    if args.aggregate_only:
        d = Path(args.aggregate_only)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8")) if (d / "meta.json").exists() else {}
        targets = meta.get("targets", targets)
        runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob("run_*.json"))]
        table, entries = aggregate(runs, targets)
        print_table(table)
        save_summary(d, table, entries)
        print(f"\n저장: {d / 'summary.csv'}")
        return

    if not args.images:
        ap.error("사진 파일을 하나 이상 넣으세요.")
    out_dir = Path(args.out) / dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    system_prompt = Path(args.prompt).read_text(encoding="utf-8")
    films = prepare_images(args.images, args.model, out_dir)
    for f in films:
        print(f"  {f['source']}: {f['orig_size'][0]}x{f['orig_size'][1]} → {f['sent_size'][0]}x{f['sent_size'][1]} ({f['name']})")

    if args.dry_run:
        for f in films:
            f["file_id"] = f"(업로드 후 채워짐:{f['name']})"
        preview = [c if c["type"] == "text" else {"type": c["type"], "file_id": c.get("file_id") or c["source"]["file_id"]}
                   for c in build_user_content(films, targets)]
        print("\n[dry-run] 시스템 프롬프트 글자 수:", len(system_prompt))
        print(json.dumps(preview, ensure_ascii=False, indent=1))
        print("\n[dry-run] API는 호출하지 않았습니다. 변환된 사진:", out_dir / "inputs")
        return

    import anthropic
    client = anthropic.Anthropic(timeout=900, max_retries=3)
    for f in films:
        with open(f["local_path"], "rb") as fh:
            f["file_id"] = client.files.upload(file=(f["name"], fh, "image/png")).id
    meta = {"model": args.model, "targets": targets, "runs": args.runs, "films": films,
            "prompt_file": str(args.prompt), "created": out_dir.name}
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    user_content = build_user_content(films, targets)
    runs, total_cost = [], 0.0
    try:
        for k in range(1, args.runs + 1):
            print(f"\n측정 {k}/{args.runs} (새 대화)")
            r = run_once(client, args.model, system_prompt, user_content, args.max_tokens, print)
            r["run"] = k
            run_dir = out_dir / f"run_{k}"
            run_dir.mkdir(exist_ok=True)
            for fid in dict.fromkeys(r["file_ids"]):
                try:
                    name = client.files.retrieve_metadata(fid).filename
                    client.files.download(fid).write_to_file(run_dir / name)
                    if not args.keep_files:
                        client.files.delete(fid)
                except Exception as e:
                    print(f"    출력 파일 {fid} 받기 실패: {e}")
            (out_dir / f"run_{k}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1, default=str),
                                                  encoding="utf-8")
            (run_dir / "claude_답변.txt").write_text("\n\n".join(r["texts"]), encoding="utf-8")
            c = est_cost(args.model, r["usage"])
            total_cost += c or 0
            u = r["usage"]
            print(f"    토큰 입력 {u['input_tokens'] + u['cache_creation_input_tokens'] + u['cache_read_input_tokens']:,}"
                  f" / 출력 {u['output_tokens']:,}" + (f" / 토큰 요금 약 ${c:.2f}" if c is not None else ""))
            if r["result"] is None:
                print("    ※ 이번 측정은 결과 제출 없이 끝났습니다. claude_답변.txt를 확인하세요.")
            runs.append(r)
    finally:
        if not args.keep_files:
            for f in films:
                try:
                    client.files.delete(f["file_id"])
                except Exception:
                    pass

    table, entries = aggregate(runs, targets)
    print_table(table)
    save_summary(out_dir, table, entries)
    print(f"\n토큰 요금 합계 약 ${total_cost:.2f} (캐시 할인 미반영, 코드 실행 시간 요금 제외)")
    print(f"결과 폴더: {out_dir.resolve()}  (summary.csv, run_*/ 측정점 이미지)")


if __name__ == "__main__":
    main()
