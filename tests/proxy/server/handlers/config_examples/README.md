# Performance Test Configurations

This directory contains example performance test configurations for different environments and use cases. Each configuration is optimized for its specific purpose while maintaining consistent measurement and analysis capabilities.

## Available Configurations

### CI Configuration (`ci.yml`)
- **Purpose**: Fast feedback in continuous integration pipelines
- **Characteristics**:
  - Reduced connection and request counts
  - Smaller data sizes
  - Relaxed performance thresholds
  - Minimal data storage
- **Use when**: Running tests in CI/CD pipelines
- **Key settings**:
  ```yaml
  concurrent_connections: 50
  request_count: 500
  min_throughput: 500.0
  ```

### Development Configuration (`development.yml`)
- **Purpose**: Local development and testing
- **Characteristics**:
  - Moderate load testing
  - Full range of data sizes
  - Relaxed thresholds
  - Detailed debugging data
- **Use when**: Developing new features or debugging
- **Key settings**:
  ```yaml
  concurrent_connections: 100
  request_count: 1000
  min_throughput: 800.0
  ```

### Production Configuration (`production.yml`)
- **Purpose**: Production performance baseline and regression testing
- **Characteristics**:
  - High load testing
  - Real-world data size distribution
  - Strict performance thresholds
  - Comprehensive metrics collection
- **Use when**: Verifying production readiness
- **Key settings**:
  ```yaml
  concurrent_connections: 500
  request_count: 10000
  min_throughput: 5000.0
  ```

### Stress Testing Configuration (`stress.yml`)
- **Purpose**: System limits and breaking point identification
- **Characteristics**:
  - Maximum load testing
  - Large data sizes
  - Extended test duration
  - Aggressive resource usage
- **Use when**: Testing system limits and stability
- **Key settings**:
  ```yaml
  concurrent_connections: 1000
  request_count: 100000
  min_throughput: 10000.0
  ```

## Usage

1. Select appropriate configuration:
   ```bash
   # For CI pipeline
   pytest --perf-config=config_examples/ci.yml

   # For development
   pytest --perf-config=config_examples/development.yml

   # For production testing
   pytest --perf-config=config_examples/production.yml

   # For stress testing
   pytest --perf-config=config_examples/stress.yml
   ```

2. Customize configuration:
   ```bash
   # Copy and modify
   cp config_examples/development.yml custom_config.yml
   # Edit custom_config.yml
   pytest --perf-config=custom_config.yml
   ```

## Configuration Parameters

### Common Settings
- `concurrent_connections`: Number of simultaneous connections
- `request_count`: Total number of requests to send
- `warmup_requests`: Requests to send before measuring
- `cooldown_time`: Time to wait between tests

### Performance Thresholds
- `min_throughput`: Minimum acceptable requests per second
- `max_latency_p95`: Maximum acceptable P95 latency
- `max_memory_mb`: Maximum acceptable memory usage
- `max_cpu_percent`: Maximum acceptable CPU usage

### Data Settings
- `data_sizes`: List of payload sizes to test
- `data_distributions`: Distribution of request sizes
- `save_raw_data`: Whether to save raw test data

### System Settings
- `disable_gc`: Whether to disable garbage collection
- `process_priority`: Process nice value
- `thread_count`: Number of threads to use

## Best Practices

1. **Environment Selection**
   - Use `ci.yml` for quick feedback
   - Use `development.yml` for local testing
   - Use `production.yml` for release validation
   - Use `stress.yml` for load testing

2. **Configuration Customization**
   - Start with the closest example
   - Adjust thresholds to your needs
   - Consider your hardware capabilities
   - Monitor system resources

3. **Test Execution**
   - Ensure clean test environment
   - Run baseline tests first
   - Monitor system during tests
   - Save results for comparison

4. **Results Analysis**
   - Compare against baselines
   - Check all metrics
   - Look for patterns
   - Document findings

## Troubleshooting

### Common Issues

1. **Resource Limits**
   - Check system ulimits
   - Monitor available memory
   - Watch for file descriptor limits
   - Check CPU usage

2. **Performance Issues**
   - Verify system load
   - Check for background processes
   - Monitor network conditions
   - Review resource usage

3. **Configuration Problems**
   - Validate YAML syntax
   - Check parameter types
   - Verify file permissions
   - Review logs for errors

## Contributing

When adding new configurations:
1. Follow the existing format
2. Document all parameters
3. Include example values
4. Add usage instructions
5. Update this README
