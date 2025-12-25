#!/usr/bin/env python3
"""
DoS Attack Simulation Script
Sends many 404 requests to trigger the "High 404 Errors from Single IP" alert.

Usage:
    python attack_simulation.py [--target URL] [--duration SECONDS] [--rate RPS]
"""

import asyncio
import aiohttp
import time
import random
import argparse
from datetime import datetime


def generate_random_ip():
    """Generate a random IP address"""
    return f"{random.randint(1, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"


async def send_request(session: aiohttp.ClientSession, url: str, num: int, random_ip: bool = True):
    """Send request to endpoint with optional random IP"""
    try:
        headers = {}
        if random_ip:
            headers["X-Forwarded-For"] = generate_random_ip()
        async with session.get(url, headers=headers) as response:
            return response.status
    except Exception as e:
        return None


async def dos_attack(target_url: str, duration: int, rate: int, random_ip: bool = True):
    """Execute DoS attack simulation"""
    print("=" * 60)
    print("🎯 DoS Attack Simulation")
    print("=" * 60)
    print(f"   Target: {target_url}")
    print(f"   Duration: {duration}s")
    print(f"   Rate: {rate} req/s")
    print(f"   Random IP: {'Yes' if random_ip else 'No (single IP)'}")
    print(f"   Start Time: {datetime.now().isoformat()}")
    print("-" * 60)

    start_time = time.time()
    request_count = 0
    success_count = 0
    error_count = 0

    async with aiohttp.ClientSession() as session:
        while time.time() - start_time < duration:
            tasks = []
            for i in range(rate):
                request_count += 1
                # Request non-existent paths to generate 404s
                url = f"{target_url}/non-existent-path-{request_count}"
                task = send_request(session, url, request_count, random_ip)
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for r in results:
                if r == 404:
                    success_count += 1
                elif r is None or isinstance(r, Exception):
                    error_count += 1

            elapsed = time.time() - start_time
            print(f"[{elapsed:6.1f}s] Requests: {request_count:6d} | "
                  f"404s: {success_count:6d} | Errors: {error_count:3d}")

            await asyncio.sleep(1)

    print("-" * 60)
    print("✅ Attack simulation completed")
    print(f"   Total requests: {request_count}")
    print(f"   404 responses: {success_count}")
    print(f"   Errors: {error_count}")
    print(f"   Duration: {time.time() - start_time:.1f}s")
    print(f"   End Time: {datetime.now().isoformat()}")
    print("=" * 60)
    print("\n⚠️  Check Kibana for alerts!")
    print("   Dashboard: http://localhost:5601")


async def normal_traffic(target_url: str, duration: int, rate: int):
    """Simulate normal traffic pattern"""
    print("=" * 60)
    print("📊 Normal Traffic Simulation")
    print("=" * 60)
    print(f"   Target: {target_url}")
    print(f"   Duration: {duration}s")
    print(f"   Rate: {rate} req/s")
    print("-" * 60)

    endpoints = [
        "/health",
        "/api/users",
        "/api/users?page=1",
        "/api/users?page=2",
        "/api/products",
        "/api/products?category=electronics",
        "/api/orders",
    ]

    start_time = time.time()
    request_count = 0
    success_count = 0

    async with aiohttp.ClientSession() as session:
        while time.time() - start_time < duration:
            tasks = []
            for i in range(rate):
                request_count += 1
                endpoint = endpoints[request_count % len(endpoints)]
                url = f"{target_url}{endpoint}"
                task = send_request(session, url, request_count)
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)
            success_count += sum(1 for r in results if r == 200)

            elapsed = time.time() - start_time
            print(f"[{elapsed:6.1f}s] Requests: {request_count:6d} | "
                  f"Success: {success_count:6d}")

            await asyncio.sleep(1)

    print("-" * 60)
    print("✅ Normal traffic simulation completed")
    print(f"   Total requests: {request_count}")
    print(f"   Successful: {success_count}")


def main():
    parser = argparse.ArgumentParser(description="Traffic simulation for log analysis")
    parser.add_argument("--target", default="http://localhost:8000",
                        help="Target URL (default: http://localhost:8000)")
    parser.add_argument("--duration", type=int, default=60,
                        help="Duration in seconds (default: 60)")
    parser.add_argument("--rate", type=int, default=500,
                        help="Requests per second (default: 500)")
    parser.add_argument("--mode", choices=["dos", "normal"], default="dos",
                        help="Simulation mode (default: dos)")
    parser.add_argument("--random-ip", action="store_true",
                        help="Use random IP for each request (X-Forwarded-For header)")

    args = parser.parse_args()

    if args.mode == "dos":
        asyncio.run(dos_attack(args.target, args.duration, args.rate, args.random_ip))
    else:
        asyncio.run(normal_traffic(args.target, args.duration, args.rate))


if __name__ == "__main__":
    main()
