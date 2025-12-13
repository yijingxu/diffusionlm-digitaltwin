#!/usr/bin/env python3
"""
Dependency Checker for Dream 7B Integration

This script checks if all required dependencies and system requirements
are met before running Dream 7B simulations.

Usage:
    python check_dream_dependencies.py
"""

import sys
import importlib
from typing import Tuple, List


def check_python_version() -> Tuple[bool, str]:
    """Check if Python version is 3.11 or higher."""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 11:
        return True, f"✓ Python {version.major}.{version.minor}.{version.micro}"
    else:
        return False, f"✗ Python {version.major}.{version.minor}.{version.micro} (requires 3.11+)"


def check_package(package_name: str, required_version: str = None) -> Tuple[bool, str, str]:
    """
    Check if a package is installed and optionally check version.
    
    Returns:
        (is_installed, status_message, installed_version)
    """
    try:
        module = importlib.import_module(package_name)
        installed_version = getattr(module, '__version__', 'unknown')
        
        if required_version:
            if installed_version == required_version:
                return True, f"✓ {package_name} {installed_version} (required: {required_version})", installed_version
            else:
                return False, f"✗ {package_name} {installed_version} (required: {required_version})", installed_version
        else:
            return True, f"✓ {package_name} {installed_version}", installed_version
    except ImportError:
        return False, f"✗ {package_name} not installed", None


def check_torch_cuda() -> Tuple[bool, str, dict]:
    """Check if PyTorch is installed and CUDA is available."""
    try:
        import torch
        torch_version = torch.__version__
        
        cuda_available = torch.cuda.is_available()
        info = {
            'torch_version': torch_version,
            'cuda_available': cuda_available
        }
        
        if cuda_available:
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            
            info.update({
                'device_count': device_count,
                'device_name': device_name,
                'memory_total_gb': memory_total,
                'cuda_version': torch.version.cuda if hasattr(torch.version, 'cuda') else 'unknown'
            })
            
            if memory_total >= 20:
                return True, f"✓ PyTorch {torch_version} with CUDA - {device_name} ({memory_total:.1f} GB)", info
            else:
                return False, f"⚠ PyTorch {torch_version} with CUDA - {device_name} ({memory_total:.1f} GB) - WARNING: < 20GB recommended", info
        else:
            return False, f"⚠ PyTorch {torch_version} installed but CUDA not available (CPU mode will be very slow)", info
            
    except ImportError:
        return False, "✗ PyTorch not installed", {}


def check_transformers() -> Tuple[bool, str]:
    """Check if transformers is installed with correct version."""
    return check_package('transformers', '4.46.2')


def check_other_dependencies() -> List[Tuple[str, bool, str]]:
    """Check other required dependencies."""
    dependencies = [
        ('yaml', 'pyyaml'),
        ('tqdm', None),
        ('asyncio', None),  # Built-in
        ('dotenv', 'python-dotenv'),
    ]
    
    results = []
    for module_name, package_name in dependencies:
        package_to_check = package_name if package_name else module_name
        try:
            importlib.import_module(module_name)
            results.append((package_to_check, True, f"✓ {package_to_check} installed"))
        except ImportError:
            results.append((package_to_check, False, f"✗ {package_to_check} not installed"))
    
    return results


def check_dream_model_access() -> Tuple[bool, str]:
    """Check if Dream model can be accessed from HuggingFace."""
    try:
        from transformers import AutoTokenizer
        model_path = "Dream-org/Dream-v0-Instruct-7B"
        
        # Try to load tokenizer config (lightweight check)
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=False
            )
            return True, f"✓ Can access Dream model from HuggingFace"
        except Exception as e:
            return False, f"✗ Cannot access Dream model: {str(e)[:100]}"
    except Exception as e:
        return False, f"✗ Error checking model access: {str(e)[:100]}"


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    """Run all dependency checks."""
    print("Dream 7B Dependency Checker")
    print("=" * 60)
    
    all_checks_passed = True
    warnings = []
    
    # Python version
    print_section("Python Version")
    passed, msg = check_python_version()
    print(msg)
    if not passed:
        all_checks_passed = False
    
    # PyTorch and CUDA
    print_section("PyTorch & CUDA")
    passed, msg, info = check_torch_cuda()
    print(msg)
    if not passed:
        all_checks_passed = False
        warnings.append(msg)
    
    if info.get('cuda_available'):
        print(f"  Device: {info.get('device_name', 'Unknown')}")
        print(f"  Total Memory: {info.get('memory_total_gb', 0):.1f} GB")
        print(f"  CUDA Version: {info.get('cuda_version', 'Unknown')}")
    
    # Transformers
    print_section("Transformers")
    passed, msg, version = check_transformers()
    print(msg)
    if not passed:
        all_checks_passed = False
        print(f"  Install with: pip install transformers==4.46.2")
    
    # Other dependencies
    print_section("Other Dependencies")
    results = check_other_dependencies()
    for package, passed_check, msg in results:
        print(f"  {msg}")
        if not passed_check:
            all_checks_passed = False
            if package == 'pyyaml':
                print(f"    Install with: pip install pyyaml")
            elif package == 'python-dotenv':
                print(f"    Install with: pip install python-dotenv")
    
    # Model access
    print_section("Dream Model Access")
    print("  Checking HuggingFace model access (this may take a moment)...")
    passed, msg = check_dream_model_access()
    print(f"  {msg}")
    if not passed:
        warnings.append("Model access check failed - you may need internet connection or HuggingFace login")
    
    # Summary
    print_section("Summary")
    if all_checks_passed:
        print("✓ All critical dependencies are installed!")
        if warnings:
            print("\n⚠ Warnings:")
            for warning in warnings:
                print(f"  - {warning}")
        print("\n✓ Ready to run Dream 7B simulations!")
        print("\nNext steps:")
        print("  1. Ensure you have at least 20GB GPU memory available")
        print("  2. Run: python text_simulation/run_LLM_simulations_dream.py \\")
        print("         --config text_simulation/configs/dream_config.yaml \\")
        print("         --max_personas 5")
        return 0
    else:
        print("✗ Some dependencies are missing or incorrect!")
        print("\nInstallation commands:")
        print("  pip install transformers==4.46.2")
        print("  pip install torch==2.5.1")
        print("  pip install pyyaml python-dotenv tqdm")
        print("\nFor CUDA support, install PyTorch with CUDA:")
        print("  Visit: https://pytorch.org/get-started/locally/")
        if warnings:
            print("\n⚠ Warnings:")
            for warning in warnings:
                print(f"  - {warning}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

