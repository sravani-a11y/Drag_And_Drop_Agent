"""FastAPI integration layer exposing the existing LangGraph agent to the InnoIDE React frontend.

See api/server.py for the app and endpoints, api/models.py for the request/
response shapes. This package only adds an HTTP boundary around the
existing agent (graph.run_workflow()) - it does not change or duplicate
the Planner, Tool Executor, Verifier, Context Engine, or MCP Bridge.
"""
