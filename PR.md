# Advanced Filtering System with Bypass Mode

This PR implements an advanced filtering system with bypass mode and post-analysis filter addition capabilities as requested.

## Features

- **Bypass Mode**: Run without filters and observe traffic without blocking
- **Post-Analysis Filter Addition**: Add specific traffic to filters after analysis
- **Flexible Condition Matching**: Create complex filter rules with various operators
- **Traffic History Recording**: Record all traffic for later analysis
- **Filter Suggestions**: Get suggested filter rules based on traffic patterns

## Implementation Details

### Core Backend Components
- `FilterMode` enum (ACTIVE/BYPASS)
- `FilterRule` class with flexible condition matching
- `FilterStorage` class for persisting rules (database and file implementations)
- `FilterManager` class to centrally manage filtering rules
- `FilterInterceptor` class that integrates with the proxy system

### Post-Analysis Filter Addition
- Traffic history recording in the `FilterManager`
- Methods to create filter rules from traffic history
- Condition suggestion algorithms based on traffic patterns

### Frontend Components
- `FilteringModeToggle` component for switching between Active/Bypass modes
- `TrafficHistoryView` component to display recorded traffic
- "Add to Filter" button for traffic history items
- `FilterRuleEditor` with preview capability
- `FilterSuggestionPanel` to show suggested conditions

### API Integration
- API endpoints for managing filter rules (CRUD operations)
- API endpoints for switching filtering modes
- API endpoints for accessing traffic history
- API endpoints for creating rules from traffic
- API endpoints for getting filter suggestions

### Proxy Integration
- Registration of the `FilterInterceptor` with the proxy system
- Proper handling of bypass mode
- Traffic recording for all requests/responses

### Database Schema
- Added tables for filter rules and settings

## Testing
- Unit tests for all components
- Integration tests for proxy and API integration
- End-to-end tests for complete workflows

## How to Use

1. Use the UI to switch between Active and Bypass modes
2. In Bypass mode, observe traffic without filtering
3. Use the "Add to Filter" button to create filter rules from specific traffic
4. Use the filter suggestions to create rules based on traffic patterns
5. Edit and preview filter rules before applying them
6. Switch to Active mode to apply the filters

## Screenshots

(Screenshots would be added here in a real PR)
