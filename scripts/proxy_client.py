import os
import time
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API settings
PROXY_API_HOST = os.getenv('ANARCHY_API_HOST', 'localhost')
PROXY_API_PORT = os.getenv('ANARCHY_API_PORT', '8000')
PROXY_API_URL = f"http://{PROXY_API_HOST}:{PROXY_API_PORT}/api/proxy/status"

# Proxy settings
PROXY_HOST = "localhost"
PROXY_PORT = 8083
PROXY_URL = f"http://{PROXY_HOST}:{PROXY_PORT}"

def fetch_proxy_settings():
    try:
        # Create session that doesn't use proxy for API calls
        session = requests.Session()
        session.trust_env = False
        
        # Add no_proxy for API host to ensure direct connection
        os.environ['no_proxy'] = f"{PROXY_API_HOST}"
        
        response = session.get(PROXY_API_URL, timeout=5)
        response.raise_for_status()
        data = response.json()
        return data["settings"] if data["isRunning"] else None
    except requests.RequestException as e:
        logger.error(f"Error fetching proxy settings: {e}")
        return None

def apply_proxy_settings(settings):
    try:
        # Set environment variables for proxy
        os.environ['HTTP_PROXY'] = PROXY_URL
        os.environ['HTTPS_PROXY'] = PROXY_URL
        
        if os.name == 'nt':  # Windows
            os.system(f'netsh winhttp set proxy {PROXY_HOST}:{PROXY_PORT} bypass-list="localhost;127.0.0.1;{PROXY_API_HOST}"')
        else:  # Linux
            try:
                # Set proxy mode to manual
                os.system('gsettings set org.gnome.system.proxy mode "manual"')
                
                # Configure HTTP proxy
                os.system(f'gsettings set org.gnome.system.proxy.http host "{PROXY_HOST}"')
                os.system(f'gsettings set org.gnome.system.proxy.http port {PROXY_PORT}')
                
                # Configure HTTPS proxy
                os.system(f'gsettings set org.gnome.system.proxy.https host "{PROXY_HOST}"')
                os.system(f'gsettings set org.gnome.system.proxy.https port {PROXY_PORT}')
                
                # Set bypass list
                os.system(f'gsettings set org.gnome.system.proxy ignore-hosts "[\'localhost\', \'127.0.0.1\', \'{PROXY_API_HOST}\']"')
                
                # Enable HTTPS proxy explicitly
                os.system('gsettings set org.gnome.system.proxy.https enabled true')
                
                logger.info(f"Applied proxy settings: host={PROXY_HOST}, port={PROXY_PORT}")
            except Exception as e:
                logger.error(f"Error applying gsettings proxy settings: {e}")
                
            # Set environment variables as fallback
            os.environ['HTTP_PROXY'] = PROXY_URL
            os.environ['HTTPS_PROXY'] = PROXY_URL
            os.environ['no_proxy'] = f"localhost,127.0.0.1,{PROXY_API_HOST}"
            
    except Exception as e:
        logger.error(f"Error applying proxy settings: {e}")

def remove_proxy_settings():
    try:
        # Clear environment variables
        for var in ['HTTP_PROXY', 'HTTPS_PROXY', 'no_proxy']:
            os.environ.pop(var, None)
            
        if os.name == 'nt':  # Windows
            os.system('netsh winhttp reset proxy')
        else:  # Linux
            try:
                os.system('gsettings set org.gnome.system.proxy mode "none"')
                logger.info("Removed proxy settings")
            except Exception as e:
                logger.error(f"Error removing gsettings proxy settings: {e}")
    except Exception as e:
        logger.error(f"Error removing proxy settings: {e}")

def test_proxy_connection():
    try:
        # Test connection through proxy with proper headers
        test_url = "http://example.com"
        proxies = {
            'http': PROXY_URL,
            'https': PROXY_URL
        }
        headers = {
            'Accept-Encoding': 'identity',  # Prevent gzip compression
            'User-Agent': 'AnarchyCopilot-ProxyClient/1.0'
        }
        response = requests.get(test_url, proxies=proxies, headers=headers, timeout=5)
        response.raise_for_status()
        logger.info("Proxy connection test successful")
        return True
    except requests.exceptions.ContentDecodingError as e:
        logger.error(f"Content decoding error: {e}")
        return False
    except Exception as e:
        logger.error(f"Proxy connection test failed: {e}")
        return False

def main():
    old_settings = None
    logger.info("Starting proxy client...")
    
    while True:
        try:
            new_settings = fetch_proxy_settings()
            
            if new_settings != old_settings:
                if new_settings is not None:  # Change condition to handle None case
                    apply_proxy_settings(new_settings)
                    # Test proxy connection after applying settings
                    if test_proxy_connection():
                        old_settings = new_settings
                else:
                    remove_proxy_settings()
                    old_settings = None
                    
            time.sleep(5)  # Poll every 5 seconds
            
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            remove_proxy_settings()
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            time.sleep(5)  # Wait before retrying

if __name__ == "__main__":
    main()
