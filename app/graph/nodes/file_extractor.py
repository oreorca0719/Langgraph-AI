from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

from app.core import trace_buffer
from app.graph.states.state import GraphState


def _read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception as e:
        raise RuntimeError(f"PDF 추출을 위해 pypdf가 필요합니다: {e}")

    reader = PdfReader(str(path))
    out: list[str] = []
    for page in reader.pages:
        try:
            out.append(page.extract_text() or "")
        except Exception:
            out.append("")
    return "\n".join(out).strip()


def _read_docx(path: Path) -> str:
    try:
        import docx  # type: ignore  # python-docx
    except Exception as e:
        raise RuntimeError(f"DOCX 추출을 위해 python-docx가 필요합니다: {e}")

    d = docx.Document(str(path))
    parts: list[str] = []
    for p in d.paragraphs:
        if p.text:
            parts.append(p.text)
    for table in d.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                parts.append("\t".join(cells))
    return "\n".join(parts).strip()


def _read_xlsx(path: Path) -> str:
    try:
        import openpyxl  # type: ignore
    except Exception as e:
        raise RuntimeError(f"XLSX 추출을 위해 openpyxl이 필요합니다: {e}")

    wb = openpyxl.load_workbook(str(path), data_only=True)
    out: list[str] = []
    for ws in wb.worksheets:
        out.append(f"[SHEET] {ws.title}")
        for row in ws.iter_rows(values_only=True):
            vals = ["" if v is None else str(v) for v in row]
            if any(v.strip() for v in vals):
                out.append("\t".join(vals))
    return "\n".join(out).strip()


def _read_pptx(file_path: str) -> str:
    from pptx import Presentation

    prs = Presentation(file_path)
    out = []

    for slide in prs.slides:
        for shape in slide.shapes:
            # ✅ 문법 오류 수정 + 텍스트 존재 시만 append
            if hasattr(shape, "text") and shape.text:
                out.append(shape.text)

    return "\n".join(out).strip()


def extract_text_from_file(path: Path) -> Tuple[str, Dict[str, Any]]:
    suffix = path.suffix.lower()
    meta: Dict[str, Any] = {
        "path": str(path),
        "name": path.name,
        "suffix": suffix,
        "size_bytes": path.stat().st_size,
    }
 
    if suffix in {".txt", ".md", ".csv", ".log"}:
        return _read_txt(path), meta
    if suffix == ".pdf":
        return _read_pdf(path), meta
    if suffix == ".docx":
        return _read_docx(path), meta
    if suffix in {".xlsx", ".xlsm"}:
        return _read_xlsx(path), meta
    if suffix == ".pptx":
        return _read_pptx(path), meta

    raise RuntimeError(f"지원하지 않는 파일 형식: {suffix}")


def file_extractor_node(state: GraphState) -> GraphState:
    print("--- [NODE] File Extractor ---")

    trace_id = (state.get("trace_id") or "")
    args = state.get("task_args") or {}
    file_path = (args.get("file_path") or "").strip()

    trace_buffer.push(trace_id, node="file_extractor", event="enter", label="execute",
                      data={"file_path": file_path})

    if not file_path:
        trace_buffer.push(trace_id, node="file_extractor", event="exit", label="execute",
                          data={"ok": False, "error": "file_path 없음"})
        return {
            **state,
            "extracted_text": "",
            "extracted_meta": {"ok": False, "error": "file_path가 비어 있습니다.", "missing_fields": ["file_path"]},
        }

    p = Path(file_path)
    if not p.exists() or not p.is_file():
        trace_buffer.push(trace_id, node="file_extractor", event="exit", label="execute",
                          data={"ok": False, "error": f"파일 없음: {file_path}"})
        return {
            **state,
            "extracted_text": "",
            "extracted_meta": {"ok": False, "error": f"파일이 존재하지 않습니다: {file_path}"},
        }

    try:
        text, meta = extract_text_from_file(p)
        meta["ok"] = True
        trace_buffer.push(trace_id, node="file_extractor", event="exit", label="execute",
                          data={"ok": True, "suffix": meta.get("suffix", ""), "size_bytes": meta.get("size_bytes", 0)})
        return {**state, "extracted_text": text, "extracted_meta": meta, "clarification_count": 0}
    except Exception as e:
        trace_buffer.push(trace_id, node="file_extractor", event="exit", label="execute",
                          data={"ok": False, "error": str(e)})
        return {
            **state,
            "extracted_text": "",
            "extracted_meta": {"ok": False, "error": str(e), "path": file_path},
        }
