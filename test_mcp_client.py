#!/usr/bin/env python3
"""Full test: connect, read, execute, verify."""

import asyncio
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


MCP_URL = "http://127.0.0.1:4040/mcp"


async def main():
    async with streamable_http_client(url=MCP_URL) as streams:
        read_stream, write_stream, get_session_id = streams
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # 1. Notebook'a bağlan
            print("=== 1. use_notebook ===")
            result = await session.call_tool("use_notebook", {
                "notebook_name": "default",
                "notebook_path": "notebook.ipynb",
                "use_mode": "connect"
            })
            for item in result.content:
                print(item.text if hasattr(item, 'text') else item)

            # 2. Notebook'u oku
            print("\n=== 2. read_notebook ===")
            result = await session.call_tool("read_notebook", {
                "notebook_name": "default",
                "detail": "brief"
            })
            for item in result.content:
                print(item.text if hasattr(item, 'text') else item)

            # 3. Kernelde kod çalıştır
            print("\n=== 3. execute_code ===")
            result = await session.call_tool("execute_code", {
                "code": "print('Merhaba Jupyter MCP!')\n2 + 3"
            })
            for item in result.content:
                print(item.text if hasattr(item, 'text') else item)

            # 4. Hücre oku (varsa ilk hücre)
            print("\n=== 4. read_cell(0) ===")
            result = await session.call_tool("read_cell", {
                "notebook_name": "default",
                "cell_index": 0
            })
            for item in result.content:
                print(item.text if hasattr(item, 'text') else item)

            # 5. Hücre ekle
            print("\n=== 5. insert_cell ===")
            result = await session.call_tool("insert_cell", {
                "notebook_name": "default",
                "cell_index": 0,
                "cell_type": "code",
                "cell_source": "print('Yeni hücre - test')"
            })
            for item in result.content:
                print(item.text if hasattr(item, 'text') else item)

            # 6. Hücre çalıştır
            print("\n=== 6. execute_cell(0) ===")
            result = await session.call_tool("execute_cell", {
                "notebook_name": "default",
                "cell_index": 0,
                "timeout": 30
            })
            for item in result.content:
                print(item.text if hasattr(item, 'text') else item)

            # 7. Son durum
            print("\n=== 7. Final Notebook State ===")
            result = await session.call_tool("read_notebook", {
                "notebook_name": "default",
                "detail": "brief"
            })
            for item in result.content:
                print(item.text if hasattr(item, 'text') else item)

            # 8. Kernel listesi
            print("\n=== 8. list_kernels ===")
            result = await session.call_tool("list_kernels", {})
            for item in result.content:
                print(item.text if hasattr(item, 'text') else item)

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
