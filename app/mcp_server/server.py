"""
MCP (Model Context Protocol) Server.
Exposes the chatbot's tools via the MCP standard so external LLM clients
can discover and use them.

Note: Requires Python 3.10+. If running on Python 3.9, this module
will gracefully skip MCP server functionality.
"""

import asyncio
import logging

logger = logging.getLogger(__name__)

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import TextContent

    from app.mcp_server.tools import MCP_TOOLS, TOOL_HANDLERS

    MCP_AVAILABLE = True

    # Create MCP server instance
    mcp_server = Server("scalable-ai-chatbot")

    @mcp_server.list_tools()
    async def list_tools():
        """Return the list of available MCP tools."""
        return MCP_TOOLS

    @mcp_server.call_tool()
    async def call_tool(name: str, arguments: dict):
        """Execute an MCP tool by name with the given arguments."""
        handler = TOOL_HANDLERS.get(name)

        if not handler:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

        try:
            result = await handler(arguments)
            return [TextContent(type="text", text=result)]
        except Exception as e:
            logger.error(f"MCP tool error ({name}): {e}")
            return [TextContent(type="text", text=f"Tool execution error: {str(e)}")]

    async def run_mcp_server():
        """Run the MCP server via stdio transport."""
        async with stdio_server() as (read_stream, write_stream):
            await mcp_server.run(
                read_stream,
                write_stream,
                mcp_server.create_initialization_options(),
            )

except ImportError:
    MCP_AVAILABLE = False
    logger.info("MCP package not available (requires Python 3.10+). MCP server disabled.")

    async def run_mcp_server():
        logger.warning("MCP server cannot run: mcp package not installed.")


if __name__ == "__main__":
    asyncio.run(run_mcp_server())
