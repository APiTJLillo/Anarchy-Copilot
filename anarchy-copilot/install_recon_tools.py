#!/usr/bin/env python3
import os
import subprocess
import sys
import platform

def run_command(command, shell=False):
    """Run a command and return its output."""
    try:
        if shell:
            process = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            process = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command {command}: {e}")
        print(f"Error output: {e.stderr.decode()}")
        return False

def check_go():
    """Check if Go is installed."""
    try:
        subprocess.run(["go", "version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except subprocess.CalledProcessError:
        return False
    except FileNotFoundError:
        return False

def install_go_tool(tool_repo):
    """Install a Go-based tool."""
    print(f"Installing {tool_repo}...")
    return run_command(["go", "install", "-v", tool_repo+"@latest"])

def install_python_tool(package):
    """Install a Python-based tool."""
    print(f"Installing {package}...")
    return run_command([sys.executable, "-m", "pip", "install", package])

def install_apt_package(package):
    """Install a package using apt."""
    print(f"Installing {package}...")
    return run_command(["sudo", "apt-get", "install", "-y", package])

def main():
    system = platform.system().lower()
    if system != "linux":
        print("This script is designed for Linux systems.")
        sys.exit(1)

    # Check if running as root or with sudo
    if os.geteuid() != 0:
        print("This script needs to be run with sudo privileges.")
        sys.exit(1)

    # Install system dependencies
    print("Installing system dependencies...")
    run_command("apt-get update", shell=True)
    packages = ["libpcap-dev", "chromium-browser", "nmap", "masscan"]
    for package in packages:
        install_apt_package(package)

    # Check and install Go if needed
    if not check_go():
        print("Installing Go...")
        run_command("apt-get install -y golang", shell=True)

    # Install Go-based tools
    go_tools = [
        "github.com/projectdiscovery/nuclei/v3/cmd/nuclei",
        "github.com/projectdiscovery/httpx/cmd/httpx",
        "github.com/projectdiscovery/subfinder/v2/cmd/subfinder",
        "github.com/projectdiscovery/dnsx/cmd/dnsx",
        "github.com/tomnomnom/assetfinder",
        "github.com/tomnomnom/httprobe",
        "github.com/OWASP/Amass/v3/...@master"
    ]

    print("\nInstalling Go-based tools...")
    for tool in go_tools:
        install_go_tool(tool)

    # Install Python-based tools
    python_tools = [
        "webtech",
        "pyppeteer"  # For taking screenshots
    ]

    print("\nInstalling Python-based tools...")
    for tool in python_tools:
        install_python_tool(tool)

    print("\nInstallation complete! The following tools were installed:")
    print("- Amass (Subdomain enumeration)")
    print("- Subfinder (Subdomain discovery)")
    print("- DNSx (DNS toolkit)")
    print("- Assetfinder (Asset discovery)")
    print("- HTTProbe (Probe for active HTTP servers)")
    print("- HTTPx (HTTP toolkit)")
    print("- Nuclei (Pattern scanning)")
    print("- Masscan (Port scanning)")
    print("- Nmap (Network scanning)")
    print("- Webtech (Web technology detection)")
    print("- Chromium (For screenshots)")

if __name__ == "__main__":
    main()
