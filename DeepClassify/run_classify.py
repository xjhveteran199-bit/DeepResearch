# -*- coding: utf-8 -*-
"""
DeepClassify - 信号分类模块 入口
"""
import sys
import os
from pathlib import Path

# UTF-8
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Qt offscreen
os.environ["QT_QPA_PLATFORM"] = "offscreen"

APP_ROOT = Path(__file__).parent
sys.path.insert(0, str(APP_ROOT))
sys.path.insert(0, str(APP_ROOT / "src"))

if __name__ == "__main__":
    print("=" * 60)
    print("  DeepClassify - 信号分类模块")
    print("  启动 Gradio Web 界面...")
    print("=" * 60)

    from app import build_ui
    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        show_error=True
    )
