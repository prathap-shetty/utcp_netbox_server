# LLM Generated Code

import asyncio, utcp_http
from utcp.utcp_client import UtcpClient

async def main():
    client = await UtcpClient.create(config="providers.json")

    # Look for anything around "dcim" or "devices"
    for q in ["dcim", "device", "devices", "dcim devices"]:
        tools = await client.search_tools(q, limit=50)
        print(f"\n== Query: {q} ==")
        for t in tools:
            print("-", t.name)

asyncio.run(main())
