#!/usr/bin/env python3
"""Check the status of all TripMate AI agents and wait for them to become active."""

import os
import sys
import time

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

load_dotenv()

AGENTS = ["travel-concierge", "trip-scout", "booking-manager", "tripmate"]
POLL_INTERVAL = 10  # seconds
MAX_WAIT = 300  # 5 minutes


def main() -> None:
    endpoint = os.getenv("AZURE_AI_PROJECT_ENDPOINT")
    if not endpoint:
        print("ERROR: AZURE_AI_PROJECT_ENDPOINT not set. Run 'azd up' first.", file=sys.stderr)
        sys.exit(1)

    client = AIProjectClient(endpoint=endpoint, credential=DefaultAzureCredential())

    wait = "--wait" in sys.argv
    start = time.time()

    while True:
        all_active = True
        print(f"\n{'Agent':<25} {'Version':<10} {'Status':<15} {'Type'}")
        print("-" * 65)

        for name in AGENTS:
            try:
                versions = list(client.agents.list_versions(name))
                if not versions:
                    print(f"{name:<25} {'—':<10} {'not found':<15}")
                    all_active = False
                    continue
                latest = versions[0]
                status = getattr(latest, "status", "unknown")
                kind = getattr(latest.definition, "kind", "unknown") if hasattr(latest, "definition") else "unknown"
                print(f"{name:<25} {latest.version:<10} {status:<15} {kind}")
                if status != "active":
                    all_active = False
            except Exception as e:
                print(f"{name:<25} {'—':<10} {'error':<15} {e}")
                all_active = False

        if all_active:
            print("\nAll agents are active and ready.")
            break

        if not wait:
            print("\nSome agents are not active yet. Use --wait to poll until ready.")
            sys.exit(1)

        elapsed = time.time() - start
        if elapsed > MAX_WAIT:
            print(f"\nTimed out after {MAX_WAIT}s. Some agents are still not active.")
            sys.exit(1)

        print(f"\nWaiting {POLL_INTERVAL}s for agents to become active... ({int(elapsed)}s elapsed)")
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
