#!/usr/bin/env python3
"""export_budoux_models.py — 从本机 split_lang 依赖的 budoux 包导出分词模型。

src/textfront/data/budoux/{ja.json,zh-hans.json} 由 budoux(pip) 自带数据复制
而来(split_lang 用 load_default_japanese_parser / simplified_chinese_parser)。
不入库(.gitignore), 运行时由本脚本再生:

    python3 tools/export_budoux_models.py
"""
import shutil
from pathlib import Path

import budoux

SRC = Path(budoux.__file__).parent / "models"
DST = Path(__file__).resolve().parent.parent / "src/textfront/data/budoux"


def main():
    DST.mkdir(parents=True, exist_ok=True)
    for name in ("ja.json", "zh-hans.json"):
        src = SRC / name
        if not src.exists():
            raise SystemExit(f"budoux 模型缺失: {src} (pip install budoux)")
        shutil.copy2(src, DST / name)
        print(f"copied {src} -> {DST / name}")


if __name__ == "__main__":
    main()
