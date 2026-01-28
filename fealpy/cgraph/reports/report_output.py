"""Graph nodes for assembling simulation report artifacts."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

from fealpy.cgraph.nodetype import CNodeType, PortConf, DataType

__all__ = ["ReportOutput"]


class ReportOutput(CNodeType):
    r"""Collect textual snippets from multiple sources and emit a report file."""

    TITLE: str = "报告输出"
    PATH: str = "后处理.报告生成"
    DESC: str = "汇总多个节点的报告内容并生成文本报告文件。"
    INPUT_SLOTS = [
        PortConf(
            "report_content",
            DataType.NONE,
            ttype=2,
            desc="报告内容，可连接多个上游节点",
            title="报告内容",
        ),
    ]
    OUTPUT_SLOTS = [
        PortConf("output_path", DataType.STRING, desc="生成的报告文件路径", title="输出路径"),
    ]

    @staticmethod
    def run(report_content: list[Any] | Any | None = None) -> str:
        # Normalize the collected payload into text blocks.
        payload: list[Any]
        if report_content is None:
            payload = []
        elif isinstance(report_content, list):
            payload = report_content
        else:
            payload = [report_content]

        rendered_blocks: list[str] = []
        for item in payload:
            if item is None:
                continue
            if isinstance(item, bytes):
                rendered_blocks.append(item.decode("utf-8", errors="replace"))
            elif isinstance(item, str):
                rendered_blocks.append(item)
            else:
                try:
                    rendered_blocks.append(json.dumps(item, ensure_ascii=False, indent=2))
                except TypeError:
                    rendered_blocks.append(str(item))

        if not rendered_blocks:
            rendered_blocks.append("(empty)")

        report_dir = Path.cwd() / "report"
        report_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"report_{timestamp}.txt"
        output_path = report_dir / filename

        lines: list[str] = []
        for index, block in enumerate(rendered_blocks, start=1):
            marker = f"--- Section {index} ---"
            lines.append(marker)
            lines.append(block)
            if not block.endswith("\n"):
                lines.append("")

        output_path.write_text("\n".join(lines), encoding="utf-8")
        return str(output_path.resolve())
