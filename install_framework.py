#!/usr/bin/env python3
"""
Installation and Setup Script for Spacecraft Safe RL Framework.
Handles dependency installation and environment configuration.

Aerospace Engineering Implementation
Author: Strtus
"""

import subprocess
import sys
import os
from pathlib import Path
import platform


class FrameworkInstaller:
    """Installation manager for the spacecraft framework."""
    
    def __init__(self):
        self.python_executable = sys.executable
        self.system_info = self.get_system_info()
        self.project_root = Path(__file__).parent
    
    def get_system_info(self):
        """Gather system information for optimal configuration."""
        return {
            'platform': platform.system(),
            'architecture': platform.machine(),
            'python_version': platform.python_version(),
            'cuda_available': self.check_cuda_availability()
        }
    
    def check_cuda_availability(self):
        """Check if CUDA is available for GPU acceleration."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def install_base_requirements(self):
        """Install core dependencies from requirements file."""
        print("Installing base requirements...")
        
        requirements_file = self.project_root / "requirements.txt"
        
        if not requirements_file.exists():
            print("Error: requirements.txt not found")
            return False
        
        try:
            subprocess.check_call([
                self.python_executable, "-m", "pip", "install", 
                "-r", str(requirements_file)
            ])
            print("Base requirements installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error installing requirements: {e}")
            return False
    
    def install_pytorch_with_cuda(self):
        """Install PyTorch with appropriate CUDA support."""
        print("Configuring PyTorch installation...")
        
        if self.system_info['cuda_available']:
            print("CUDA detected. Installing PyTorch with CUDA support...")
            torch_command = [
                self.python_executable, "-m", "pip", "install",
                "torch", "torchvision", "torchaudio",
                "--index-url", "https://download.pytorch.org/whl/cu118"
            ]
        else:
            print("CUDA not available. Installing CPU-only PyTorch...")
            torch_command = [
                self.python_executable, "-m", "pip", "install",
                "torch", "torchvision", "torchaudio",
                "--index-url", "https://download.pytorch.org/whl/cpu"
            ]
        
        try:
            subprocess.check_call(torch_command)
            print("PyTorch installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error installing PyTorch: {e}")
            return False
    
    def install_optional_aerospace_packages(self):
        """Install optional aerospace-specific packages if available."""
        print("Installing optional aerospace packages...")
        
        optional_packages = [
            "astropy",
            "poliastro", 
            "sgp4",
            "skyfield"
        ]
        
        for package in optional_packages:
            try:
                subprocess.check_call([
                    self.python_executable, "-m", "pip", "install", package
                ])
                print(f"Installed {package}")
            except subprocess.CalledProcessError:
                print(f"Could not install {package} (optional)")
    
    def verify_installation(self):
        """Verify that all critical components can be imported."""
        print("Verifying installation...")
        
        critical_imports = [
            "numpy",
            "scipy", 
            "torch",
            "gymnasium",
            "matplotlib"
        ]
        
        failed_imports = []
        
        for module in critical_imports:
            try:
                __import__(module)
                print(f"[OK] {module}")
            except ImportError:
                print(f"[FAIL] {module}")
                failed_imports.append(module)
        
        if failed_imports:
            print(f"\nFailed to import: {', '.join(failed_imports)}")
            return False
        
        print("\nAll critical components verified")
        return True
    
    def create_example_config(self):
        """Create example configuration files."""
        print("Creating example configuration files...")
        
        # Mission configuration
        mission_config = {
            'mission_parameters': {
                'max_episode_steps': 2000,
                'safety_threshold': 0.1,
                'target_success_rate': 0.95
            },
            'spacecraft_config': {
                'mass': 500.0,
                'inertia_matrix': [[100, 0, 0], [0, 100, 0], [0, 0, 100]],
                'thruster_configuration': 'redundant_quad'
            },
            'orbital_parameters': {
                'target_altitude': 400000,
                'inclination': 51.6,
                'eccentricity': 0.0001
            },
            'safety_constraints': {
                'max_relative_velocity': 0.5,
                'approach_corridor_angle': 15.0,
                'minimum_separation_distance': 1.0
            }
        }
        
        import json
        config_file = self.project_root / "example_mission_config.json"
        with open(config_file, 'w') as f:
            json.dump(mission_config, f, indent=2)
        
        print(f"Created example configuration: {config_file}")
    
    def run_installation(self):
        """Execute complete installation process."""
        print("Starting Spacecraft Safe RL Framework Installation")
        print("=" * 50)
        print(f"Python: {self.system_info['python_version']}")
        print(f"Platform: {self.system_info['platform']}")
        print(f"CUDA Available: {self.system_info['cuda_available']}")
        print("=" * 50)
        
        steps = [
            ("Installing PyTorch", self.install_pytorch_with_cuda),
            ("Installing base requirements", self.install_base_requirements),
            ("Installing optional packages", self.install_optional_aerospace_packages),
            ("Verifying installation", self.verify_installation),
            ("Creating example configs", self.create_example_config)
        ]
        
        for step_name, step_function in steps:
            print(f"\n{step_name}...")
            if not step_function():
                print(f"Installation failed at: {step_name}")
                return False
        
        print("\n" + "=" * 50)
        print("Installation completed successfully!")
        print("=" * 50)
        print("\nNext steps:")
        print("1. Run: python main.py --mode train")
        print("2. Or: python main.py --mode simulate")
        print("3. Check example_mission_config.json for configuration options")
        
        return True


def main():
    """Main installation function."""
    installer = FrameworkInstaller()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--verify-only":
        installer.verify_installation()
    else:
        installer.run_installation()


if __name__ == "__main__":
    main()