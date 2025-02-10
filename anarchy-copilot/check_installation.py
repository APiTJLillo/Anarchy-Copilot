#!/usr/bin/env python3
import sys
import os
import subprocess
import importlib.util
import pkg_resources

GREEN = "\033[0;32m"
RED = "\033[0;31m"
YELLOW = "\033[1;33m"
NC = "\033[0m"

def print_header(text):
    print("\n" + "-" * 50)
    print(text)
    print("-" * 50)

def check_mark(success):
    return f"{GREEN}✓{NC}" if success else f"{RED}✗{NC}"

def check_command(cmd):
    try:
        result = subprocess.run([cmd, '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            return True
        result = subprocess.run([cmd, '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            return True
        result = subprocess.run([cmd, '-h'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0
    except:
        return False

def get_package_version(package_name):
    try:
        return pkg_resources.get_distribution(package_name).version
    except:
        return None

def check_python_package(package_spec):
    try:
        # Split package name and version
        if '>=' in package_spec:
            package_name, required_version = package_spec.split('>=')
        else:
            package_name = package_spec
            required_version = None

        # Handle special cases
        if '[' in package_name:
            base_package = package_name.split('[')[0]
            extras = package_name[package_name.index('['):]
            package_name = base_package

        # Try to get the installed version
        installed_version = get_package_version(package_name)
        
        if installed_version:
            if required_version:
                try:
                    pkg_resources.parse_version(installed_version) >= pkg_resources.parse_version(required_version)
                    return True
                except:
                    return False
            return True
        return False
    except:
        return False

def check_virtual_env():
    return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

def check_go_installation():
    try:
        result = subprocess.run(['go', 'version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            # Check if GOPATH is set and in PATH
            gopath = os.environ.get('GOPATH')
            if gopath and os.path.exists(gopath):
                gobin = os.path.join(gopath, 'bin')
                path = os.environ.get('PATH', '')
                if gobin in path:
                    return 'working'
            return 'installed but not properly configured'
        return 'not installed'
    except:
        return 'not installed'

def check_go_tools():
    results = {}
    tools = {
        'amass': 'OWASP Amass',
        'subfinder': 'Subfinder',
        'assetfinder': 'Assetfinder',
        'dnsx': 'DNSx'
    }
    
    for tool, description in tools.items():
        # First try direct command
        if check_command(tool):
            results[tool] = 'working'
            continue
            
        # Then try with full path
        go_path = os.path.expanduser('~/go/bin')
        tool_path = os.path.join(go_path, tool)
        if os.path.exists(tool_path) and os.access(tool_path, os.X_OK):
            if check_command(tool_path):
                results[tool] = 'working'
                continue
        
        if os.path.exists(tool_path):
            results[tool] = 'installed but not working properly'
        else:
            results[tool] = 'not installed'
    
    return results

def main():
    print("Checking Anarchy-Copilot Installation\n")

    # Check virtual environment
    print_header("Virtual Environment:")
    in_venv = check_virtual_env()
    print(f"{check_mark(in_venv)} Running in virtual environment")

    # Check system requirements
    print_header("System Requirements:")
    go_status = check_go_installation()
    go_working = go_status == 'working'
    node_working = check_command('node')
    npm_working = check_command('npm')
    
    print(f"{check_mark(go_working)} go is {go_status}")
    print(f"{check_mark(node_working)} {'node is installed and working' if node_working else 'node is not installed'}")
    print(f"{check_mark(npm_working)} {'npm is installed and working' if npm_working else 'npm is not installed'}")

    # Check Go tools
    print_header("Go Tools:")
    go_tools = check_go_tools()
    for tool, status in go_tools.items():
        mark = check_mark(status == 'working')
        print(f"{mark} {tool} is {status}")

    # Check Python packages from requirements.txt
    print_header("Python Packages:")
    with open('requirements.txt', 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    success_count = 1 if in_venv else 0  # Start with 1 for virtual env
    total_count = 1  # Start with 1 for virtual env
    
    for req in requirements:
        is_installed = check_python_package(req)
        print(f"{check_mark(is_installed)} {req} is {'installed' if is_installed else 'not installed'}")
        if is_installed:
            success_count += 1
        total_count += 1

    # Check frontend
    print_header("Frontend:")
    has_node_modules = os.path.exists('frontend/node_modules')
    print(f"{check_mark(has_node_modules)} Frontend dependencies {'installed' if has_node_modules else 'not installed'}")
    if has_node_modules:
        success_count += 1
    total_count += 1

    # Print summary
    print_header("Installation Status:")
    print(f"Total components checked: {total_count}")
    print(f"Successfully installed: {success_count}")
    print(f"Missing or failed: {total_count - success_count}\n")

    if success_count < total_count:
        print(f"{RED}✗ Some components are missing or not working properly.")
        print("Please run './install_dependencies.sh' to fix missing components.\n")
        print(f"{YELLOW}Note: Some components require specific environment activation:")
        print("1. Activate Python virtual environment: source venv/bin/activate")
        print(f"2. Ensure Go tools are in PATH: source $HOME/.bashrc{NC}")
        sys.exit(1)
    else:
        print(f"{GREEN}✓ All components are properly installed!{NC}")
        sys.exit(0)

if __name__ == "__main__":
    main()
