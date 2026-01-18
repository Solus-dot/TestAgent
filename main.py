#!/usr/bin/env python3
"""
Entry point for the TestAgent application.
Run this script to start the AI agent.
"""

import asyncio
from agent.main import run_agent

if __name__ == "__main__":
    asyncio.run(run_agent())
