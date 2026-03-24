"""
DeepDetect 入口脚本
启动 Gradio Web 界面
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_demo

if __name__ == "__main__":
    print("=" * 60)
    print("  DeepDetect - 异常检测系统")
    print("  启动中... 访问 http://localhost:7861")
    print("=" * 60)

    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        show_error=True
    )
