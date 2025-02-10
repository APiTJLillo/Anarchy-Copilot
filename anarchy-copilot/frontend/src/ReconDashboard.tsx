import React, { useState } from 'react';
import { API_URL } from './config';

type ResultType = string[] | PortScanResult | HttpxResult | WebTechResult | NucleiResult;

interface ReconResult {
  id: number;
  tool: string;
  domain: string;
  results: ResultType;
  status: string;
  error_message: string | null;
  start_time: string | null;
  end_time: string | null;
  project_id: number | null;
  total_found?: number;
  response_time?: number;
  categories?: {
    [key: string]: string[];
  };
}

const isStringArray = (results: ResultType): results is string[] => {
  return Array.isArray(results);
};

const isPortScanResult = (results: ResultType): results is PortScanResult => {
  return !Array.isArray(results) && 'total_ports_scanned' in results;
};

const isHttpxResult = (results: ResultType): results is HttpxResult => {
  return !Array.isArray(results) && 'endpoints' in results;
};

const isWebTechResult = (results: ResultType): results is WebTechResult => {
  return !Array.isArray(results) && 'technologies' in results && !('endpoints' in results);
};

const isNucleiResult = (results: ResultType): results is NucleiResult => {
  return !Array.isArray(results) && 'findings' in results;
};

interface DomainCategory {
  name: string;
  description: string;
  color: string;
}

const domainCategories: DomainCategory[] = [
  { name: 'api', description: 'API Endpoints', color: 'bg-purple-100' },
  { name: 'dev', description: 'Development/Staging', color: 'bg-yellow-100' },
  { name: 'admin', description: 'Admin Interfaces', color: 'bg-red-100' },
  { name: 'cdn', description: 'Content Delivery', color: 'bg-blue-100' },
  { name: 'static', description: 'Static Content', color: 'bg-green-100' }
];

interface ReconFilter {
  tool?: string;
  domain?: string;
  project_id?: number;
  category?: string;
  minResults?: number;
  maxResults?: number;
  dateRange?: string;
}

const tools = [
  { value: 'amass', label: 'Amass - Subdomain Discovery', description: 'Passive subdomain enumeration' },
  { value: 'subfinder', label: 'Subfinder - Subdomain Discovery', description: 'Fast subdomain discovery tool' },
  { value: 'assetfinder', label: 'Assetfinder - Asset Discovery', description: 'Find domains and subdomains related to a target' },
  { value: 'dnsx', label: 'DNSx - DNS Toolkit', description: 'DNS resolution and analysis' },
  { value: 'portscan', label: 'Port Scanner', description: 'Fast port scanning and service detection' },
  { value: 'httprobe', label: 'HTTProbe', description: 'Probe for working HTTP/HTTPS servers' },
  { value: 'httpx', label: 'HTTPx - Web Analysis', description: 'HTTP endpoint analysis and tech detection' },
  { value: 'webtech', label: 'WebTech', description: 'Website technology fingerprinting' },
  { value: 'nuclei', label: 'Nuclei - Pattern Scan', description: 'Smart pattern-based scanning engine' }
];

interface Service {
  state: string;
  service: string;
}

interface Port {
  port: string;
  details: Service;
}

interface PortScanResult {
  total_ports_scanned: number;
  open_ports: number;
  services: {
    [key: string]: Service;
  };
}

interface HttpEndpoint {
  url: string;
  status_code: number;
  title: string;
  technologies: string[];
  screenshot?: string;
}

interface HttpxResult {
  total_endpoints: number;
  endpoints: HttpEndpoint[];
}

interface WebTechResult {
  technologies: string[];
  headers: { [key: string]: string };
  cookies: string[];
}

interface NucleiResult {
  total_findings: number;
  findings: {
    template: string;
    severity: string;
    name: string;
    matched: string;
  }[];
}

const getSeverityColor = (severity: string): string => {
  switch (severity.toLowerCase()) {
    case 'critical':
      return 'bg-red-600 text-white';
    case 'high':
      return 'bg-red-500 text-white';
    case 'medium':
      return 'bg-orange-500 text-white';
    case 'low':
      return 'bg-yellow-500';
    case 'info':
      return 'bg-blue-500 text-white';
    default:
      return 'bg-gray-500 text-white';
  }
};

const downloadJson = (data: any, filename: string) => {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
};

export default function ReconDashboard() {
  const [reconResults, setReconResults] = useState<ReconResult[]>([]);
  const [domain, setDomain] = useState('');
  const [tool, setTool] = useState('amass');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'scan' | 'history'>('scan');
  const [historyFilter, setHistoryFilter] = useState<ReconFilter>({});
  const [scanStatus, setScanStatus] = useState<string | null>(null);
  const [selectedCategories, setSelectedCategories] = useState<string[]>([]);
  const [showAdvancedFilters, setShowAdvancedFilters] = useState(false);
  const [progressPercentage, setProgressPercentage] = useState(0);
  const [selectedTechnology, setSelectedTechnology] = useState<string>('');
  const [selectedStatusCode, setSelectedStatusCode] = useState<string>('');
  const [selectedSeverity, setSelectedSeverity] = useState<string>('');
  const [sortField, setSortField] = useState<string>('');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc');

  const validateDomain = (domain: string): boolean => {
    const forbiddenDomains = ['.gov', '.mil', '.edu', 'nhs.uk', 'gc.ca', 'gov.uk', 'mil.ru'];
    return !forbiddenDomains.some(forbidden => domain.toLowerCase().includes(forbidden));
  };

  const fetchHistory = async (filter: ReconFilter = {}) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const params = new URLSearchParams();
      if (filter.tool) params.append('tool', filter.tool);
      if (filter.domain) params.append('domain', filter.domain);
      if (filter.project_id) params.append('project_id', filter.project_id.toString());
      
      const response = await fetch(`${API_URL}/recon_results/history/?${params}`);
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to fetch history');
      }

      const data = await response.json();
      setReconResults(data.map((result: any) => ({
        ...result,
        results: result.results?.domains || []
      })));
    } catch (error) {
      console.error('Error fetching history:', error);
      setError(error instanceof Error ? error.message : 'An error occurred while fetching history');
    } finally {
      setIsLoading(false);
    }
  };

  const runReconScan = async () => {
    if (!domain) {
      setError('Please enter a domain.');
      return;
    }

    if (!validateDomain(domain)) {
      setError('This domain is out of permitted scope.');
      return;
    }

    setIsLoading(true);
    setError(null);
    setScanStatus('Starting scan...');

    try {
      setScanStatus('Initializing scan...');
      setProgressPercentage(10);

      const response = await fetch(`${API_URL}/recon_results/?tool=${tool}&domain=${domain}`);
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to fetch results');
      }

      setScanStatus('Processing results...');
      setProgressPercentage(50);

      const result = await response.json();
      
      // Categorize domains
      const categories: { [key: string]: string[] } = {};
      result.results?.domains?.forEach((domain: string) => {
        if (domain.includes('api.') || domain.includes('-api.')) {
          categories.api = [...(categories.api || []), domain];
        } else if (domain.includes('dev.') || domain.includes('staging.')) {
          categories.dev = [...(categories.dev || []), domain];
        } else if (domain.includes('admin.') || domain.includes('manage.')) {
          categories.admin = [...(categories.admin || []), domain];
        } else if (domain.includes('cdn.') || domain.includes('assets.')) {
          categories.cdn = [...(categories.cdn || []), domain];
        } else if (domain.includes('static.') || domain.includes('media.')) {
          categories.static = [...(categories.static || []), domain];
        }
      });

      // Calculate response time
      const startTime = result.start_time ? new Date(result.start_time).getTime() : Date.now();
      const endTime = result.end_time ? new Date(result.end_time).getTime() : Date.now();
      const responseTime = (endTime - startTime) / 1000; // Convert to seconds

      setScanStatus('Scan completed successfully');
      setProgressPercentage(100);
      setReconResults([{
        ...result,
        results: result.tool === 'amass' || result.tool === 'subfinder' || result.tool === 'assetfinder' || result.tool === 'dnsx' || result.tool === 'httprobe'
          ? result.results?.domains || []
          : result.results
      }]);
    } catch (error) {
      console.error('Error fetching reconnaissance results:', error);
      setError(error instanceof Error ? error.message : 'An error occurred while fetching results');
      setScanStatus('Scan failed');
    } finally {
      setIsLoading(false);
    }
  };

  const renderScanTab = () => (
    <div className="mb-4 p-4 bg-gray-100 rounded">
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <label className="block">
          <span className="text-gray-700">Domain:</span>
          <input
            type="text"
            className="mt-1 block w-full rounded border-gray-300 shadow-sm"
            value={domain}
            onChange={(e) => setDomain(e.target.value)}
            placeholder="example.com"
            disabled={isLoading}
          />
        </label>
        
        <label className="block">
          <span className="text-gray-700">Tool:</span>
          <select 
            className="mt-1 block w-full rounded border-gray-300 shadow-sm"
            value={tool}
            onChange={(e) => setTool(e.target.value)}
            disabled={isLoading}
          >
            {tools.map(({ value, label }) => (
              <option key={value} value={value}>{label}</option>
            ))}
          </select>
        </label>
        
        <div className="flex items-end space-x-4 col-span-1">
          <button
            className={`px-4 py-2 rounded ${
              isLoading
                ? 'bg-gray-400'
                : 'bg-blue-500 hover:bg-blue-600'
            } text-white`}
            onClick={runReconScan}
            disabled={isLoading}
          >
            {isLoading ? 'Running Scan...' : 'Run Recon'}
          </button>
        </div>
      </div>

      {error && (
        <div className="mt-4 p-2 bg-red-100 border border-red-400 text-red-700 rounded">
          {error}
        </div>
      )}

      {scanStatus && !error && (
        <div className="mt-4 p-2 bg-blue-100 border border-blue-400 text-blue-700 rounded">
          {scanStatus}
        </div>
      )}
    </div>
  );

  const renderHistoryTab = () => (
    <div className="mb-4 p-4 bg-gray-100 rounded">
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <input
          type="text"
          className="rounded border-gray-300 shadow-sm"
          placeholder="Filter by domain"
          value={historyFilter.domain || ''}
          onChange={(e) => {
            const newFilter = { ...historyFilter, domain: e.target.value };
            setHistoryFilter(newFilter);
            fetchHistory(newFilter);
          }}
        />
        <select
          className="rounded border-gray-300 shadow-sm"
          value={historyFilter.tool || ''}
          onChange={(e) => {
            const newFilter = { ...historyFilter, tool: e.target.value || undefined };
            setHistoryFilter(newFilter);
            fetchHistory(newFilter);
          }}
        >
          <option value="">All Tools</option>
          {tools.map(({ value, label }) => (
            <option key={value} value={value}>{label}</option>
          ))}
        </select>
      </div>
    </div>
  );

  return (
    <div className="p-4">
      <h1 className="text-2xl font-bold mb-4">Reconnaissance Dashboard</h1>
      
      <div className="mb-4">
        <div className="border-b border-gray-200">
          <nav className="-mb-px flex">
            <button
              className={`px-4 py-2 border-b-2 ${
                activeTab === 'scan'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
              onClick={() => setActiveTab('scan')}
            >
              Run Scan
            </button>
            <button
              className={`ml-8 px-4 py-2 border-b-2 ${
                activeTab === 'history'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
              onClick={() => {
                setActiveTab('history');
                fetchHistory();
              }}
            >
              History
            </button>
          </nav>
        </div>
      </div>

      {activeTab === 'scan' ? renderScanTab() : renderHistoryTab()}

      <div className="space-y-4">
        {reconResults.map((result, index) => (
          <div key={index} className="p-4 bg-white shadow rounded">
            <div className="flex justify-between items-start border-b pb-2 mb-4">
              <div>
                <h2 className="text-xl font-semibold">
                  {result.tool} results for {result.domain}
                </h2>
                <p className="text-sm text-gray-500">
                  Status: {result.status}
                  {result.project_id && ` | Project ID: ${result.project_id}`}
                </p>
              </div>
              <div className="text-right">
                <div className="text-sm text-gray-500">
                  {result.start_time && `Started: ${new Date(result.start_time).toLocaleString()}`}
                </div>
                <div className="text-sm text-gray-500">
                  {result.end_time && `Completed: ${new Date(result.end_time).toLocaleString()}`}
                </div>
              </div>
            </div>

            {result.error_message && (
              <div className="mb-4 p-2 bg-red-100 border border-red-400 text-red-700 rounded">
                Error: {result.error_message}
              </div>
            )}

            {progressPercentage > 0 && progressPercentage < 100 && (
        <div className="w-full h-2 bg-gray-200 rounded mt-4">
          <div 
            className="h-full bg-blue-500 rounded transition-all duration-500"
            style={{ width: `${progressPercentage}%` }}
          />
        </div>
      )}

      <div className="mt-4">
        <div className="flex justify-between items-center mb-3">
          <h3 className="font-semibold">Subdomains:</h3>
          {result.total_found && (
            <span className="text-sm text-gray-500">
              Total found: {result.total_found} | Response time: {result.response_time?.toFixed(2)}s
            </span>
          )}
        </div>

        <div className="mb-4">
          <div className="flex flex-wrap gap-2">
            {domainCategories.map(category => (
              <button
                key={category.name}
                onClick={() => {
                  setSelectedCategories(prev => 
                    prev.includes(category.name)
                      ? prev.filter(c => c !== category.name)
                      : [...prev, category.name]
                  );
                }}
                className={`px-3 py-1 rounded text-sm ${
                  selectedCategories.includes(category.name)
                    ? `${category.color} border-2 border-gray-400`
                    : 'bg-gray-100'
                }`}
              >
                {category.description}
              </button>
            ))}
          </div>
        </div>

        <div className="flex justify-end mb-4 space-x-2">
          <button
            onClick={() => downloadJson(result.results, `recon-${result.domain}-${result.tool}.json`)}
            className="px-3 py-1 text-sm bg-green-500 text-white rounded hover:bg-green-600"
          >
            Export Results
          </button>

          <select
            value={sortField}
            onChange={(e) => setSortField(e.target.value)}
            className="px-3 py-1 text-sm border rounded"
          >
            <option value="">Sort by...</option>
            <option value="domain">Domain Name</option>
            <option value="date">Discovery Date</option>
            <option value="status">Status Code</option>
            <option value="tech">Technology</option>
          </select>

          <button
            onClick={() => setSortDirection(prev => prev === 'asc' ? 'desc' : 'asc')}
            className="px-3 py-1 text-sm border rounded"
          >
            {sortDirection === 'asc' ? '↑' : '↓'}
          </button>

          {result.tool === 'httpx' && (
            <select
              value={selectedStatusCode}
              onChange={(e) => setSelectedStatusCode(e.target.value)}
              className="px-3 py-1 text-sm border rounded"
            >
              <option value="">Filter by Status</option>
              <option value="200">200 OK</option>
              <option value="301">301 Redirect</option>
              <option value="302">302 Redirect</option>
              <option value="401">401 Unauthorized</option>
              <option value="403">403 Forbidden</option>
              <option value="404">404 Not Found</option>
              <option value="500">500 Server Error</option>
            </select>
          )}

          {(result.tool === 'httpx' || result.tool === 'webtech') && (
            <select
              value={selectedTechnology}
              onChange={(e) => setSelectedTechnology(e.target.value)}
              className="px-3 py-1 text-sm border rounded"
            >
              <option value="">Filter by Tech</option>
              <option value="nginx">Nginx</option>
              <option value="apache">Apache</option>
              <option value="php">PHP</option>
              <option value="wordpress">WordPress</option>
              <option value="react">React</option>
              <option value="vue">Vue</option>
              <option value="angular">Angular</option>
            </select>
          )}

          {result.tool === 'nuclei' && (
            <select
              value={selectedSeverity}
              onChange={(e) => setSelectedSeverity(e.target.value)}
              className="px-3 py-1 text-sm border rounded"
            >
              <option value="">Filter by Severity</option>
              <option value="critical">Critical</option>
              <option value="high">High</option>
              <option value="medium">Medium</option>
              <option value="low">Low</option>
              <option value="info">Info</option>
            </select>
          )}
        </div>

        {/* Port Scan Results */}
        {result.tool === 'portscan' && (
          <div className="mb-6">
            <h3 className="text-lg font-semibold mb-2">Port Scan Results</h3>
            <div className="bg-gray-50 p-4 rounded">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <p className="text-sm text-gray-600">Total Ports Scanned: {(result.results as PortScanResult).total_ports_scanned}</p>
                  <p className="text-sm text-gray-600">Open Ports: {(result.results as PortScanResult).open_ports}</p>
                </div>
                <div className="overflow-auto max-h-96">
                  <table className="min-w-full">
                    <thead>
                      <tr>
                        <th className="px-4 py-2 bg-gray-100">Port</th>
                        <th className="px-4 py-2 bg-gray-100">State</th>
                        <th className="px-4 py-2 bg-gray-100">Service</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries((result.results as PortScanResult).services).map(([port, service]) => (
                        <tr key={port} className="hover:bg-gray-50">
                          <td className="px-4 py-2 border">{port}</td>
                          <td className="px-4 py-2 border">{service.state}</td>
                          <td className="px-4 py-2 border">{service.service}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* HTTPx Results */}
        {result.tool === 'httpx' && (
          <div className="mb-6">
            <h3 className="text-lg font-semibold mb-2">HTTP Analysis Results</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {(result.results as HttpxResult).endpoints
                .filter(endpoint => !selectedStatusCode || endpoint.status_code.toString() === selectedStatusCode)
                .filter(endpoint => !selectedTechnology || endpoint.technologies.some(tech => tech.toLowerCase().includes(selectedTechnology.toLowerCase())))
                .map((endpoint, idx) => (
                  <div key={idx} className="bg-white shadow rounded-lg p-4">
                    <div className="mb-2">
                      <a href={endpoint.url} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">
                        {endpoint.url}
                      </a>
                    </div>
                    <div className="text-sm">
                      <p>Status: <span className={`px-2 py-1 rounded ${endpoint.status_code === 200 ? 'bg-green-100' : 'bg-yellow-100'}`}>
                        {endpoint.status_code}
                      </span></p>
                      <p>Title: {endpoint.title}</p>
                      {endpoint.technologies.length > 0 && (
                        <div className="mt-2">
                          <p>Technologies:</p>
                          <div className="flex flex-wrap gap-1 mt-1">
                            {endpoint.technologies.map((tech, techIdx) => (
                              <span key={techIdx} className="px-2 py-1 bg-blue-100 rounded text-xs">
                                {tech}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                      {endpoint.screenshot && (
                        <div className="mt-2">
                          <img src={endpoint.screenshot} alt="Screenshot" className="w-full rounded shadow-sm" />
                        </div>
                      )}
                    </div>
                  </div>
                ))}
            </div>
          </div>
        )}

        {/* WebTech Results */}
        {result.tool === 'webtech' && (
          <div className="mb-6">
            <h3 className="text-lg font-semibold mb-2">Technology Detection Results</h3>
            <div className="bg-white shadow rounded-lg p-4">
              <div className="mb-4">
                <h4 className="font-medium mb-2">Detected Technologies:</h4>
                <div className="flex flex-wrap gap-2">
                  {(result.results as WebTechResult).technologies
                    .filter(tech => !selectedTechnology || tech.toLowerCase().includes(selectedTechnology.toLowerCase()))
                    .map((tech, idx) => (
                      <span key={idx} className="px-3 py-1 bg-blue-100 rounded">
                        {tech}
                      </span>
                    ))}
                </div>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <h4 className="font-medium mb-2">Headers:</h4>
                  <div className="bg-gray-50 p-2 rounded">
                    {Object.entries((result.results as WebTechResult).headers).map(([key, value]) => (
                      <div key={key} className="text-sm">
                        <span className="font-mono text-gray-600">{key}:</span> {value}
                      </div>
                    ))}
                  </div>
                </div>
                <div>
                  <h4 className="font-medium mb-2">Cookies:</h4>
                  <div className="bg-gray-50 p-2 rounded">
                    {(result.results as WebTechResult).cookies.map((cookie, idx) => (
                      <div key={idx} className="text-sm font-mono text-gray-600">
                        {cookie}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Nuclei Results */}
        {result.tool === 'nuclei' && (
          <div className="mb-6">
            <h3 className="text-lg font-semibold mb-2">Pattern Scan Results</h3>
            <div className="bg-white shadow rounded-lg p-4">
              <p className="mb-4">Total Findings: {(result.results as NucleiResult).total_findings}</p>
              <div className="space-y-4">
                {(result.results as NucleiResult).findings
                  .filter(finding => !selectedSeverity || finding.severity.toLowerCase() === selectedSeverity.toLowerCase())
                  .map((finding, idx) => (
                    <div key={idx} className="border-l-4 border-l-gray-300 pl-4">
                      <div className="flex items-center gap-2 mb-1">
                        <span className={`px-2 py-1 rounded text-sm ${getSeverityColor(finding.severity)}`}>
                          {finding.severity}
                        </span>
                        <h4 className="font-medium">{finding.name}</h4>
                      </div>
                      <p className="text-sm text-gray-600 mb-1">Template: {finding.template}</p>
                      <p className="text-sm">
                        <a href={finding.matched} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">
                          {finding.matched}
                        </a>
                      </p>
                    </div>
                  ))}
              </div>
            </div>
          </div>
        )}

        {isStringArray(result.results) && result.results.length > 0 ? (
          <div className="mt-2 max-h-64 overflow-y-auto">
            <div className="grid grid-cols-1 gap-2">
              {Object.entries(result.categories || {}).map(([category, domains]) => (
                selectedCategories.length === 0 || selectedCategories.includes(category) ? (
                  <div key={category} className={`p-2 rounded ${
                    domainCategories.find(c => c.name === category)?.color || 'bg-gray-100'
                  }`}>
                    <h4 className="font-medium mb-2">{
                      domainCategories.find(c => c.name === category)?.description || category
                    }</h4>
                    <ul className="list-disc pl-5 space-y-1">
                      {domains.map((subdomain, idx) => (
                        <li key={idx} className="text-gray-700">
                          <a 
                            href={`https://${subdomain}`} 
                            target="_blank" 
                            rel="noopener noreferrer"
                            className="hover:text-blue-600"
                          >
                            {subdomain}
                          </a>
                        </li>
                      ))}
                    </ul>
                  </div>
                ) : null
              ))}
              
              {/* Uncategorized domains */}
              <div className="p-2 rounded bg-gray-100">
                <h4 className="font-medium mb-2">Other Domains</h4>
                <ul className="list-disc pl-5 space-y-1">
                  {(result.results as string[]).filter((domain: string) => 
                    !Object.values(result.categories || {}).flat().includes(domain)
                  ).map((subdomain: string, idx: number) => (
                    <li key={idx} className="text-gray-700">
                      <a 
                        href={`https://${subdomain}`} 
                        target="_blank" 
                        rel="noopener noreferrer"
                        className="hover:text-blue-600"
                      >
                        {subdomain}
                      </a>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        ) : (
          <p className="mt-2 text-gray-500">No subdomains found.</p>
        )}
      </div>
          </div>
        ))}
      </div>
    </div>
  );
}
