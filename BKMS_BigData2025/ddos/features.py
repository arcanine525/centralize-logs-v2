"""Feature extraction from Apache logs."""
import re
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional
from collections import Counter

LOG_PATTERN = re.compile(
    r'(\d+\.\d+\.\d+\.\d+)\s+-\s+-\s+\[([^\]]+)\]\s+"(\w+)\s+([^"]+)[^"]*"\s+(\d+)\s+(\d+|-)'
)


def parse_log(line: str) -> Optional[Dict]:
    """Parse a single Apache log line."""
    m = LOG_PATTERN.match(line.strip())
    if not m:
        return None
    
    ip, ts, method, url, status, bytes_ = m.groups()
    try:
        timestamp = datetime.strptime(ts.split()[0], '%d/%b/%Y:%H:%M:%S')
    except:
        timestamp = datetime.now()
    
    return {
        "ip": ip,
        "timestamp": timestamp,
        "method": method.upper(),
        "url": url.split('?')[0],  # Remove query params
        "status": int(status),
        "bytes": int(bytes_) if bytes_ != '-' else 0
    }


def entropy(items: List) -> float:
    """Calculate Shannon entropy."""
    if not items:
        return 0.0
    counts = Counter(items)
    total = len(items)
    return -sum((c/total) * np.log2(c/total) for c in counts.values() if c > 0)


def extract_features(logs: List[Dict], time_window: int = 60) -> np.ndarray:
    """
    Extract 16 features from a list of parsed logs.
    
    Features:
    1. request_count - Total requests
    2. unique_ips - Distinct IPs
    3. requests_per_ip - Avg requests per IP
    4. unique_methods - Distinct HTTP methods
    5. get_ratio - GET requests ratio
    6. post_ratio - POST requests ratio
    7. avg_bytes - Average response bytes
    8. total_bytes - Total response bytes
    9. status_2xx_ratio - Success ratio
    10. status_4xx_ratio - Client error ratio
    11. status_5xx_ratio - Server error ratio
    12. unique_urls - Distinct URLs
    13. avg_url_length - Average URL length
    14. request_rate - Requests per second
    15. ip_entropy - IP distribution entropy
    16. url_entropy - URL distribution entropy
    """
    if not logs:
        return np.zeros(16, dtype=np.float32)
    
    n = len(logs)
    ips = [l['ip'] for l in logs]
    methods = [l['method'] for l in logs]
    statuses = [l['status'] for l in logs]
    urls = [l['url'] for l in logs]
    bytes_list = [l['bytes'] for l in logs]
    
    unique_ips = len(set(ips))
    unique_urls = len(set(urls))
    
    return np.array([
        n,                                          # 1. request_count
        unique_ips,                                 # 2. unique_ips
        n / max(unique_ips, 1),                     # 3. requests_per_ip
        len(set(methods)),                          # 4. unique_methods
        methods.count('GET') / n,                   # 5. get_ratio
        methods.count('POST') / n,                  # 6. post_ratio
        np.mean(bytes_list) if bytes_list else 0,   # 7. avg_bytes
        sum(bytes_list),                            # 8. total_bytes
        sum(200 <= s < 300 for s in statuses) / n,  # 9. status_2xx_ratio
        sum(400 <= s < 500 for s in statuses) / n,  # 10. status_4xx_ratio
        sum(s >= 500 for s in statuses) / n,        # 11. status_5xx_ratio
        unique_urls,                                # 12. unique_urls
        np.mean([len(u) for u in urls]),            # 13. avg_url_length
        n / time_window,                            # 14. request_rate
        entropy(ips),                               # 15. ip_entropy
        entropy(urls),                              # 16. url_entropy
    ], dtype=np.float32)

