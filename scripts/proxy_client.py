import os
import time
import requests

PROXY_API_URL = f"http://{os.getenv('PROXY_HOST', 'localhost')}:{os.getenv('PROXY_PORT', '8000')}/api/proxy/status"

def fetch_proxy_settings():
    try:
        response = requests.get(PROXY_API_URL)
        response.raise_for_status()
        data = response.json()
        return data["settings"] if data["isRunning"] else None
    except requests.RequestException as e:
        print(f"Error fetching proxy settings: {e}")
        return None

def apply_proxy_settings(settings):
    # Always use port 8080 for the actual proxy traffic
    proxy_port = 8080
    
    if os.name == 'nt':  # Windows
        # Use netsh to set proxy settings on Windows
        os.system(f'netsh winhttp set proxy {settings["host"]}:{proxy_port} bypass-list="*.local"')
    else:  # Linux
        # Set proxy mode to manual
        os.system(f'gsettings set org.gnome.system.proxy mode "manual"')
        
        # Configure HTTP proxy
        os.system(f'gsettings set org.gnome.system.proxy.http host "{settings["host"]}"')
        os.system(f'gsettings set org.gnome.system.proxy.http port {proxy_port}')
        
        # Configure HTTPS proxy
        os.system(f'gsettings set org.gnome.system.proxy.https host "{settings["host"]}"')
        os.system(f'gsettings set org.gnome.system.proxy.https port {proxy_port}')
        
        # Set bypass list similar to Windows
        os.system('gsettings set org.gnome.system.proxy ignore-hosts "[\'.local\']"')
        
        # Enable HTTPS proxy explicitly
        os.system('gsettings set org.gnome.system.proxy.https enabled true')
        
        # Set SOCKS proxy for better compatibility
        os.system(f'gsettings set org.gnome.system.proxy.socks host "{settings["host"]}"')
        os.system(f'gsettings set org.gnome.system.proxy.socks port {proxy_port}')

def remove_proxy_settings():
    if os.name == 'nt':  # Windows
        # Reset proxy settings on Windows
        os.system('netsh winhttp reset proxy')
    else:  # Linux
        # Reset proxy settings on Linux
        os.system('gsettings set org.gnome.system.proxy mode "none"')

def main():
    while True:
        old_settings = None
        new_settings = fetch_proxy_settings()
        if new_settings and new_settings != old_settings:
            apply_proxy_settings(new_settings)
            print(f"Applied proxy settings: {new_settings}")
        else:
            remove_proxy_settings()
            print("Proxy is not running")
        time.sleep(5)  # Poll every 5 seconds

if __name__ == "__main__":
    main()
