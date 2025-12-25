"""
Simulate: Import logs to ES and trigger prediction

Usage:
    python simulate.py                    # Sample logs
    python simulate.py --attack           # DDoS attack simulation
    python simulate.py --file access.log  # Import from file
    python simulate.py --trigger          # Just trigger prediction (no import)
"""

import requests
import argparse
import re
import os
import json
from datetime import datetime
import time

ES_URL = os.getenv("ES_URL", "http://localhost:9200")
API_URL = os.getenv("API_URL", "http://localhost:8000")
INDEX = "logs"

LOG_PATTERN = re.compile(
    r'(\d+\.\d+\.\d+\.\d+)\s+-\s+-\s+\[([^\]]+)\]'
)


def clear_index():
    """Delete all documents in index."""
    try:
        requests.post(f"{ES_URL}/{INDEX}/_delete_by_query", json={"query": {"match_all": {}}})
        requests.post(f"{ES_URL}/{INDEX}/_refresh")
        print("Cleared index")
    except:
        pass


def parse_timestamp(log: str) -> str:
    """Extract timestamp from log line."""
    match = LOG_PATTERN.search(log)
    if match:
        ts_str = match.group(2).split()[0]
        try:
            ts = datetime.strptime(ts_str, '%d/%b/%Y:%H:%M:%S')
            return ts.isoformat()
        except:
            pass
    return datetime.now().isoformat()


def import_from_file(filepath: str, limit: int = None):
    """Import logs from file to Elasticsearch using bulk API."""
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return 0
    
    print(f"\nImporting logs from: {filepath}")
    
    # Use session for connection reuse
    session = requests.Session()
    
    count = 0
    batch = []
    batch_size = 500  # Bulk insert batch size
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or not LOG_PATTERN.search(line):
                continue
            
            # Bulk API format: action + document
            batch.append('{"index":{}}')
            batch.append(json.dumps({"log": line, "timestamp": parse_timestamp(line)}))
            count += 1
            
            # Send batch
            if len(batch) >= batch_size * 2:
                bulk_data = '\n'.join(batch) + '\n'
                session.post(f"{ES_URL}/{INDEX}/_bulk", 
                           data=bulk_data, 
                           headers={"Content-Type": "application/x-ndjson"})
                batch = []
                print(f"  Imported {count} logs...")
            
            if limit and count >= limit:
                break
    
    # Send remaining
    if batch:
        bulk_data = '\n'.join(batch) + '\n'
        session.post(f"{ES_URL}/{INDEX}/_bulk", 
                   data=bulk_data, 
                   headers={"Content-Type": "application/x-ndjson"})
    
    session.post(f"{ES_URL}/{INDEX}/_refresh")
    session.close()
    print(f"Imported {count} logs")
    return count


def insert_sample_logs(attack=False):
    """Insert sample logs."""
    if attack:
        # DDoS: many requests from few IPs
        logs = [
            f'10.0.0.{i%3} - - [15/Jan/2024:10:01:{i%60:02d} +0000] "GET /api/data HTTP/1.1" 200 100'
            for i in range(200)
        ]
        print("\nInserting DDoS attack simulation (200 requests from 3 IPs)...")
    else:
        # Normal: few requests from many IPs
        logs = [
            f'192.168.1.{i} - - [15/Jan/2024:10:00:{i%60:02d} +0000] "GET /page{i} HTTP/1.1" 200 1024'
            for i in range(20)
        ]
        print("\nInserting normal traffic (20 requests from 20 IPs)...")
    
    for log in logs:
        requests.post(f"{ES_URL}/{INDEX}/_doc", json={
            "log": log,
            "timestamp": parse_timestamp(log)
        })
    
    requests.post(f"{ES_URL}/{INDEX}/_refresh")
    print(f"Inserted {len(logs)} logs")


def trigger_predict():
    """Trigger /predict endpoint to process all unpredicted logs."""
    print("\nTriggering prediction...")
    
    try:
        resp = requests.post(f"{API_URL}/predict", timeout=60)
        result = resp.json()
        
        print(f"  Processed: {result.get('processed', 0)} logs")
        
        if 'windows' in result:
            ddos_count = sum(1 for w in result['windows'] if w.get('status') == 'DDOS')
            normal_count = len(result['windows']) - ddos_count
            print(f"  Windows: {len(result['windows'])} total, {ddos_count} DDOS, {normal_count} NORMAL")
            
            # Show DDOS windows
            for w in result['windows']:
                if w.get('status') == 'DDOS':
                    print(f"    [DDOS] {w['window']} ({w['logs']} logs)")
        
        return result
    except Exception as e:
        print(f"  Error: {e}")
        return None


def show_stats():
    """Show prediction statistics from ES."""
    print("\nResults in Elasticsearch:")
    
    try:
        # Count by status
        resp = requests.get(f"{ES_URL}/{INDEX}/_search", json={
            "size": 0,
            "aggs": {
                "by_status": {"terms": {"field": "status", "missing": "PENDING"}}
            }
        })
        
        buckets = resp.json().get('aggregations', {}).get('by_status', {}).get('buckets', [])
        for b in buckets:
            print(f"  {b['key']}: {b['doc_count']} logs")
        
        # Show sample DDOS logs
        resp = requests.get(f"{ES_URL}/{INDEX}/_search", json={
            "query": {"term": {"status": "DDOS"}},
            "size": 5,
            "sort": [{"probability": "desc"}]
        })
        
        hits = resp.json()['hits']['hits']
        if hits:
            print("\n  Sample DDOS detections:")
            for h in hits:
                src = h['_source']
                log_short = src['log'][:50] + '...' if len(src['log']) > 50 else src['log']
                print(f"    p={src.get('probability', 0):.2f} | {log_short}")
    except Exception as e:
        print(f"  Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="DDoS Detection Demo")
    parser.add_argument('--file', '-f', type=str, help='Import logs from file (e.g., ../../dataset/access.log)')
    parser.add_argument('--limit', '-n', type=int, help='Limit number of logs to import')
    parser.add_argument('--attack', action='store_true', help='Insert DDoS attack simulation')
    parser.add_argument('--trigger', action='store_true', help='Only trigger prediction (no import)')
    parser.add_argument('--clear', action='store_true', help='Clear index first')
    parser.add_argument('--stats', action='store_true', help='Show statistics only')
    args = parser.parse_args()
    
    print("=" * 60)
    print("DDoS Detection - Simulate & Trigger")
    print("=" * 60)
    
    # Check services
    try:
        requests.get(f"{ES_URL}", timeout=5)
    except:
        print(f"ERROR: Elasticsearch not running at {ES_URL}")
        print("Run 'docker-compose up -d' first")
        return
    
    try:
        requests.get(f"{API_URL}/health", timeout=5)
    except:
        print(f"ERROR: API not running at {API_URL}")
        print("Run 'docker-compose up -d' first")
        return
    
    # Just show stats
    if args.stats:
        show_stats()
        return
    
    # Clear index
    if args.clear:
        clear_index()
    
    # Import or trigger only
    if not args.trigger:
        if args.file:
            import_from_file(args.file, args.limit)
        else:
            insert_sample_logs(attack=args.attack)
        time.sleep(1)  # Wait for ES indexing
    
    # Trigger prediction
    trigger_predict()
    
    # Show stats
    show_stats()
    
    print("\n" + "=" * 60)
    print("Commands:")
    print(f"  Import file:  python simulate.py --file ../../dataset/access.log")
    print(f"  Trigger only: python simulate.py --trigger")
    print(f"  View DDOS:    curl '{ES_URL}/{INDEX}/_search?q=status:DDOS&pretty'")
    print("=" * 60)


if __name__ == "__main__":
    main()

