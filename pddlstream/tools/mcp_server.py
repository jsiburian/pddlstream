"""Serves the agent tools over the Model Context Protocol (stdio), for agents that speak MCP.

    workspace = Workspace()
    workspace.add_problem('kitchen', problem, stream_info)
    serve(workspace)  # Blocks; an MCP client (e.g. a coding agent) launches this script

Needs the optional dependency `mcp` (pip install 'pddlstream[agent]'); the core never imports it.
"""

from pddlstream.tools.toolbox import TOOLS, Toolbox


def serve(workspace, name='pddlstream'):
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as error:
        raise ImportError("The MCP server needs the mcp package: pip install 'pddlstream[agent]'") from error
    toolbox = Toolbox(workspace)
    server = FastMCP(name)
    for tool in TOOLS:

        def handler(_name=tool.name, **arguments):
            return toolbox.call(_name, arguments)

        server.add_tool(handler, name=tool.name, description=tool.description)
    server.run()
