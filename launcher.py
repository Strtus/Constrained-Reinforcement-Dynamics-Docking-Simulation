#!/usr/bin/env python3
"""
千帆卫星快速启动脚本
==================

Author: Strtus
"""

import os
import sys
from pathlib import Path

def main():
    """主函数"""
    print("=" * 60)
    print("千帆卫星自主对接训练系统")
    print("Qianfan Satellite Autonomous Docking Training System")
    print("=" * 60)
    print("Author: Strtus")
    print("Branch: qianfan-satellite")
    print()
    
    # 检查Python版本
    if sys.version_info < (3, 8):
        print("错误: 需要Python 3.8或更高版本")
        sys.exit(1)
    
    print("系统配置:")
    print(f"Python版本: {sys.version}")
    print(f"工作目录: {os.getcwd()}")
    print()
    
    # 显示千帆配置参数
    print("千帆卫星配置参数:")
    print("- 卫星质量: 150 kg")
    print("- 轨道高度: 500 km (太阳同步轨道)")
    print("- 推进系统: 8个冷气推进器")
    print("- 推力: 0.5 N/推进器")
    print("- 比冲: 70 s")
    print("- 对接精度: ±5 cm")
    print()
    
    # 显示可用命令
    print("可用训练命令:")
    print("1. 安装依赖:")
    print("   python install_framework.py")
    print()
    print("2. 开始千帆训练:")
    print("   python examples/train_qianfan.py --mode train")
    print()
    print("3. 评估千帆模型:")
    print("   python examples/train_qianfan.py --mode eval --model path/to/model.pth")
    print()
    print("4. 运行测试:")
    print("   python test_framework.py")
    print()
    print("5. 通用训练 (使用千帆配置):")
    print("   python main.py --mode train")
    print()
    
    # 交互选择
    while True:
        choice = input("请选择操作 (1-5, q退出): ").strip()
        
        if choice == 'q' or choice == 'Q':
            print("退出千帆训练系统")
            break
        elif choice == '1':
            print("正在安装框架依赖...")
            os.system("python install_framework.py")
        elif choice == '2':
            print("开始千帆卫星训练...")
            os.system("python examples/train_qianfan.py --mode train")
        elif choice == '3':
            model_path = input("请输入模型路径 (回车跳过): ").strip()
            if model_path:
                os.system(f"python examples/train_qianfan.py --mode eval --model {model_path}")
            else:
                os.system("python examples/train_qianfan.py --mode eval")
        elif choice == '4':
            print("运行测试...")
            os.system("python test_framework.py")
        elif choice == '5':
            print("运行通用训练...")
            os.system("python main.py --mode train")
        else:
            print("无效选择，请重试")
        
        print()


if __name__ == "__main__":
    main()