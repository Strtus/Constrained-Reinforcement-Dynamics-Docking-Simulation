"""
启动Streamlit演示应用
===================

快速启动SafeRL的Web演示界面
"""

import subprocess
import sys
import os

def main():
    """启动Streamlit应用"""
    
    # 获取可视化脚本路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    viz_script = os.path.join(script_dir, "..", "src", "visualization.py")
    
    print("=" * 60)
    print("SafeRL - 航天器安全强化学习演示")
    print("=" * 60)
    print(f"启动Streamlit应用...")
    print(f"脚本路径: {viz_script}")
    print("\n访问地址: http://localhost:8501")
    print("按 Ctrl+C 停止服务")
    print("=" * 60)
    
    try:
        # 启动Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", viz_script,
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n\nStreamlit服务已停止")
    except Exception as e:
        print(f"启动失败: {e}")
        print("\n请确保已安装streamlit:")
        print("pip install streamlit")

if __name__ == "__main__":
    main()