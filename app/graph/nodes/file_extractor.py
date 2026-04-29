from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from app.graph.states.state import GraphState


def _table_to_markdown(rows: List[List[str]]) -> str:
    """행 리스트를 마크다운 표로 변환. 첫 행을 헤더로 사용. 빈 입력은 ''."""
    if not rows:
        return ""
    col_count = max(len(r) for r in rows)
    if col_count == 0:
        return ""

    def _norm(cells: List[str]) -> List[str]:
        padded = (cells + [""] * col_count)[:col_count]
        return [c.replace("|", "\\|").replace("\n", " ").strip() for c in padded]

    header = _norm(rows[0])
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join(["---"] * col_count) + "|")
    for row in rows[1:]:
        lines.append("| " + " | ".join(_norm(row)) + " |")
    return "\n".join(lines)


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
    """DOCX: 본문 단락은 단락 단위로, 표는 마크다운 표로 변환.
    단락·표 사이는 \\n\\n으로 단락 분리하여 청킹 시 경계 보존.
    """
    try:
        import docx  # type: ignore  # python-docx
    except Exception as e:
        raise RuntimeError(f"DOCX 추출을 위해 python-docx가 필요합니다: {e}")

    d = docx.Document(str(path))
    parts: list[str] = []

    for p in d.paragraphs:
        text = (p.text or "").strip()
        if text:
            parts.append(text)

    for idx, table in enumerate(d.tables, start=1):
        rows: List[List[str]] = []
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                rows.append(cells)
        if rows:
            parts.append(f"## 표 {idx}")
            parts.append(_table_to_markdown(rows))

    return "\n\n".join(parts).strip()


def _read_xlsx(path: Path) -> str:
    """XLSX: 시트별 마크다운 표로 변환. 첫 행을 헤더로 사용.
    시트 간 \\n\\n으로 분리, 시트명을 헤딩으로 명시.
    """
    try:
        import openpyxl  # type: ignore
    except Exception as e:
        raise RuntimeError(f"XLSX 추출을 위해 openpyxl이 필요합니다: {e}")

    wb = openpyxl.load_workbook(str(path), data_only=True)
    parts: list[str] = []
    for ws in wb.worksheets:
        rows: List[List[str]] = []
        for row in ws.iter_rows(values_only=True):
            vals = ["" if v is None else str(v) for v in row]
            if any(v.strip() for v in vals):
                rows.append(vals)
        if not rows:
            continue
        parts.append(f"## 시트: {ws.title}")
        parts.append(_table_to_markdown(rows))
    return "\n\n".join(parts).strip()


def _read_pptx(file_path: str) -> str:
    """PPTX: 슬라이드 내 도형 텍스트 + 표(마크다운 변환) 추출.
    표는 _table_to_markdown으로 구조 보존. 슬라이드 단위 메타데이터·노트는 후속 PR에서 추가.
    """
    from pptx import Presentation

    prs = Presentation(file_path)
    out: list[str] = []

    for slide in prs.slides:
        for shape in slide.shapes:
            if getattr(shape, "has_table", False):
                rows: List[List[str]] = []
                for row in shape.table.rows:
                    cells = [(cell.text or "").strip() for cell in row.cells]
                    if any(cells):
                        rows.append(cells)
                if rows:
                    out.append(_table_to_markdown(rows))
            elif hasattr(shape, "text") and shape.text:
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

