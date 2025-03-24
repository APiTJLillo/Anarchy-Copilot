"""Domain aggregation and categorization functionality."""

from typing import List, Dict, Set

class DomainAggregator:
    def __init__(self):
        self.domains: Set[str] = set()
        
    def add(self, domains: List[str]):
        """Add domains to the set, removing duplicates."""
        self.domains.update(domains)
        
    def get_domains(self) -> List[str]:
        """Return sorted list of unique domains."""
        return sorted(list(self.domains))

    def categorize_domains(self, domains: List[str]) -> Dict[str, List[str]]:
        """Categorize domains based on common patterns."""
        categories: Dict[str, List[str]] = {
            "api": [],
            "dev": [],
            "admin": [],
            "cdn": [],
            "static": []
        }
        
        for domain in domains:
            if "api." in domain or "-api." in domain:
                categories["api"].append(domain)
            elif "dev." in domain or "staging." in domain:
                categories["dev"].append(domain)
            elif "admin." in domain or "manage." in domain:
                categories["admin"].append(domain)
            elif "cdn." in domain or "assets." in domain:
                categories["cdn"].append(domain)
            elif "static." in domain or "media." in domain:
                categories["static"].append(domain)
        
        # Remove empty categories
        return {k: v for k, v in categories.items() if v}
