import os
from pathlib import Path
from contextlib import asynccontextmanager

from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Route

from src.brain.engine import WednesdayEngine

CURRENT_DIR = Path(__file__).resolve().parent
TEMPLATE_PATH = CURRENT_DIR / "templates" / "index.html"

engine = WednesdayEngine()


@asynccontextmanager
async def lifespan(app):
    """Lifecycle manager: connects MCP client and loads models on startup."""
    print("\n[Wednesday Web] Initializing Wednesday Engine & MCP tools...")
    await engine.initialize()
    print(f"[Wednesday Web] Ready. Connected {len(engine.tools)} MCP tools.\n")
    yield
    print("\n[Wednesday Web] Shutting down Wednesday Engine...")
    await engine.close()


async def homepage(request):
    """Serve the Wednesday Addams styled Single Page Application."""
    if not TEMPLATE_PATH.exists():
        return HTMLResponse("<h1>Template not found</h1>", status_code=404)
    content = TEMPLATE_PATH.read_text(encoding="utf-8")
    return HTMLResponse(content)


async def api_status(request):
    """Get system and connection status."""
    return JSONResponse(
        {
            "status": "online",
            "assistant": "Wednesday Addams",
            "model": "openai/gpt-oss-120b",
            "tools_count": len(engine.tools),
            "pending_draft": bool(engine.pending_draft),
            "draft_filename": (
                engine.pending_draft["filename"] if engine.pending_draft else None
            ),
        }
    )


async def api_history(request):
    """Retrieve persistent conversation history from SQLite."""
    history = engine.get_history_list(limit=50)
    return JSONResponse({"history": history})


async def api_tools(request):
    """List available MCP tools."""
    tools = engine.get_tools_list()
    return JSONResponse({"tools": tools})


async def api_chat(request):
    """Handle chat message from the user."""
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON payload."}, status_code=400)

    message = data.get("message", "").strip()
    if not message:
        return JSONResponse({"error": "Message cannot be empty."}, status_code=400)

    response = await engine.process_message(message)
    return JSONResponse(response)


async def api_draft_action(request):
    """Handle draft decisions: approve, cancel, or revise."""
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON payload."}, status_code=400)

    action = data.get("action", "").lower().strip()

    if action == "approve":
        result = await engine.approve_pending_draft()
        return JSONResponse(result)
    elif action == "cancel" or action == "discard":
        result = await engine.cancel_pending_draft()
        return JSONResponse(result)
    elif action == "revise":
        revision = data.get("revision", "")
        result = await engine.revise_pending_draft(revision)
        return JSONResponse(result)
    else:
        return JSONResponse(
            {"status": "error", "message": f"Unknown action: {action}"},
            status_code=400,
        )


app = Starlette(
    debug=True,
    lifespan=lifespan,
    routes=[
        Route("/", homepage, methods=["GET"]),
        Route("/api/status", api_status, methods=["GET"]),
        Route("/api/history", api_history, methods=["GET"]),
        Route("/api/tools", api_tools, methods=["GET"]),
        Route("/api/chat", api_chat, methods=["POST"]),
        Route("/api/draft/action", api_draft_action, methods=["POST"]),
    ],
)
