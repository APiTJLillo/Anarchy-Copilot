# Anarchy Copilot

AI-powered bug bounty suite that integrates modern technologies to streamline the bug bounty process. The suite includes a custom web proxy with advanced interception capabilities, along with comprehensive reconnaissance and vulnerability scanning features.

## Features

- **Proxy Client**: A client application that polls for proxy settings and applies them as needed.
- **Local Proxy Checkbox**: A UI checkbox to enable the local proxy, which only activates when the Docker proxy is running.
- **Proxy Settings Endpoint**: A new `/settings` endpoint to provide current proxy settings to the client.

### Reconnaissance Module ✅
- Automated subdomain discovery
- Port scanning and service detection
- Web endpoint analysis
- Continuous monitoring with change detection
- Scheduling system for recurring scans
- AI-assisted pattern recognition for interesting findings
- Real-time progress tracking

### Vulnerability Discovery Module 🔄
- **Advanced Proxy Platform**:
  - Custom web attack proxy with request/response interception
  - Detailed request/response inspection with modal view
  - Navigation between captured requests with prev/next controls
  - History view available even when proxy is stopped
  - Real-time traffic analysis and pattern matching
  - Session management and replay capabilities
  - HTTP/HTTPS/HTTP2 and WebSocket support
  - Plugin system for custom interceptors
  - Automated security testing workflows

- **Integrated Scanning**:
  - Nuclei integration as primary scanning engine
  - Auxiliary tool integration (SQLMap, XSSHunter, etc.)
  - AI-driven parameter mutation and fuzzing
  - Automated vulnerability verification
  - Advanced false positive reduction with ML
  - JavaScript deobfuscation and analysis

- **AI & Language Features** ✨:
  - Natural language generation for clear explanations
  - Language model integration for enhanced analysis
  - Semantic analysis of responses and contexts
  - Cross-lingual support with neural translation
  - Cultural context awareness and adaptation
  - Advanced validation and quality checks
  - Learning capabilities for continuous improvement

- **Performance & Scale**:
  - Efficient traffic handling and connection pooling
  - Distributed scanning capabilities
  - Advanced resource management
  - Real-time monitoring and alerting

## Installation

```bash
# Development installation
pip install -e ".[dev]"

# Regular installation
pip install .
```

## Docker Development

After making changes to Python dependencies (requirements.txt or setup.py):
```bash
# Rebuild containers with new dependencies
docker compose build dev proxy

# Restart services
docker compose up -d
```

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/anarchycopilot/anarchycopilot.git
cd anarchycopilot
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

3. Install development dependencies:
```bash
pip install -e ".[dev]"
```

4. Configure environment variables:
```bash
# Copy example environment file
cp .env.example .env

# Edit .env file with your settings
nano .env  # or use your preferred editor
```

5. Run environment validation:
```bash
python -m tests.tools.validate_environment
```

6. Run tests:
```bash
make test
```

## Environment Configuration

The application uses environment variables for configuration. Default values are provided in `.env.example`. Key configuration areas include:

### API Settings
- `ANARCHY_API_TITLE` - API name in OpenAPI docs
- `ANARCHY_API_VERSION` - API version
- `ANARCHY_DEBUG` - Enable debug mode
- `ANARCHY_CORS_ORIGINS` - Allowed CORS origins

### Proxy Settings
- `ANARCHY_PROXY_HOST` - Proxy server host
- `ANARCHY_PROXY_PORT` - Proxy server port
- `ANARCHY_PROXY_INTERCEPT_REQUESTS` - Enable request interception
- `ANARCHY_PROXY_INTERCEPT_RESPONSES` - Enable response interception
- `ANARCHY_PROXY_MAX_CONNECTIONS` - Maximum concurrent connections

### AI & Language Settings
- `ANARCHY_LLM_MODEL` - Language model selection
- `ANARCHY_LLM_API_KEY` - API key for language model
- `ANARCHY_TRANSLATION_MODEL` - Translation model selection
- `ANARCHY_SEMANTIC_CACHE` - Enable semantic caching
- `ANARCHY_CULTURAL_DB` - Path to cultural context database

### Certificate Settings
- `CA_CERT_PATH` - Path to CA certificate
- `CA_KEY_PATH` - Path to CA private key

For detailed configuration options, see `.env.example`.

## Key Components

```
anarchy-copilot/
├── models/              # Database models
│   ├── base.py         # Core models (User, Project)
│   ├── recon.py        # Reconnaissance models
│   └── vulnerability.py # Vulnerability and reporting models
├── api/                # API endpoints and handlers
│   ├── projects/       # Project management
│   └── proxy/         # Web Proxy Module
│       ├── core.py    # Proxy server implementation
│       ├── interceptor.py # Request/response interception
│       ├── session.py # Session management
│       └── config.py  # Proxy configuration
├── recon_module/         # Reconnaissance module
│   ├── models.py        # Data structures
│   ├── recon_manager.py # Main interface
│   ├── scheduler.py     # Scheduling system
│   └── tools/          # Scanner implementations
├── vuln_module/         # Vulnerability module
├── ai_module/           # AI and language processing
│   ├── nlg.py          # Natural language generation
│   ├── llm.py          # Language model integration
│   ├── semantic.py     # Semantic analysis
│   ├── multilingual.py # Cross-lingual support
│   └── cultural.py    # Cultural context handling
│   ├── models.py       # Vulnerability models
│   ├── vuln_manager.py # Vulnerability interface
│   └── scanner/       # Scanner implementations
├── tests/              # Test suite ✨
│   ├── tools/         # Test utilities
│   ├── examples/      # Example tests
│   └── data/         # Test data & assets
├── examples/          # Usage examples
└── docker-compose.yml # Container orchestration
```

## Components

### Web Proxy Module
- Custom HTTP/HTTPS proxy with advanced features
- Request/response interception and modification
- Session management and history tracking
- Plugin system for custom security checks
- Certificate authority for HTTPS inspection
- Real-time traffic analysis

## Testing

### Quick Start
```bash
# Validate environment
python -m tests.tools.validate_environment

# Run all tests
make test

# Run specific test types
make test-unit         # Unit tests
make test-integration  # Integration tests
make test-ai          # AI/Language tests
```

### Test Categories
- Unit tests (fast, no external dependencies)
- Integration tests (external tool interactions)
- API tests (endpoint testing)
- AI/Language tests (NLG, LLM, semantic analysis)
- Security tests (scanning & validation)

For detailed testing information, see `tests/README.md`.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Set up development environment
4. Make your changes
5. Run tests and checks
6. Submit a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
