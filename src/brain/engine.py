import asyncio
import json
import re
import sys
from contextlib import AsyncExitStack
from pathlib import PurePosixPath, PureWindowsPath

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from src.brain.llm import model, chat_history as original_history
from src.brain.memory import initialize_memory, load_history, save_exchange, DB_PATH
import sqlite3

SYSTEM_PROMPT = original_history[0].content

SYSTEM_PROMPT += """

BROWSER INTENT RULES:
- Always prioritise the user's latest explicit instruction.
- Previous conversation history must not override a new command.
- "Open Chrome" means open the browser, not YouTube.
- "Open YouTube" means navigate to YouTube.
- "Search YouTube for X" means use browser_youtube_search.
- "Play X on YouTube" means use browser_youtube_play.
- If Wednesday has just asked the user for a song or video title after opening YouTube, interpret the next title/query as a PLAY request.
- In that situation, call browser_youtube_play, not browser_youtube_search.
- Pass the complete song title and artist as the query when known.
- Never claim a video is playing unless playback is verified.
- If YouTube requires verification, ask the user to complete it manually.
- Do not repeatedly retry blocked pages.
- Wait for the playback tool result before responding.
"""

SYSTEM_PROMPT += """

MAC FILE AND FINDER RULES:

1. FINDER
- If the user says "open Finder" without specifying a folder,
  call open_finder with {"folder": "."}.
- Never pass null for the folder argument.
- If the user specifies a folder, pass its path as a string.
- If the user specifies a folder, pass that folder to open_finder.
- If the user asks to choose or select a file, call finder_select_file.
- If the user asks to choose or select a folder, call finder_select_folder.

2. FILE DISCOVERY
- If the user asks to find or list files in a folder, call list_user_files.
- Use the folder path supplied by the user.
- Relative paths are relative to the user's home folder.

3. READING FILES
- If the user asks to read a file, call read_user_file.
- Use the exact path returned by Finder or file discovery.
- Never invent a file path.

4. OPENING FILES
- If the user asks to open a file, call open_user_file.
- If the file was selected using Finder, use the returned path.
- Do not claim a file was opened unless the tool succeeds.

5. SECURITY
- Never attempt to bypass protected-file restrictions.
- Never read hidden files, credentials, API keys or protected folders.
- Never modify or delete user files without explicit approval.
- Use workspace creation and editing workflows for workspace files.
- Do not claim access to a file if macOS denies permission.

6. TOOL SELECTION
- Use Finder tools for opening Finder and selecting files or folders.
- Use list_user_files for browsing file names.
- Use read_user_file for reading file contents.
- Use open_user_file for opening files in their default applications.
"""

FILE_DRAFT_PROMPT = """
You prepare files for Wednesday, a personal assistant.

The user wants a file prepared for review before it is saved.

Return ONLY a valid JSON object with these two string fields:
{
"filename": "example.py",
"content": "print('Hello World')"
}

Rules:
- filename must be a relative path inside the workspace.
- Never use an absolute path or .. path traversal.
- Generate the complete requested file content.
- Preserve code indentation and newlines.
- Do not use Markdown fences around the JSON.
- Do not claim the file has been saved.
- If revising a draft, return the entire revised file.
"""


def extract_text(content):
    """Convert supported LangChain content formats into plain text."""
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text", "")
                if text:
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts).strip()

    return str(content).strip()


def is_file_request(text):
    """Recognize common explicit requests to create a new file."""
    patterns = [
        r"\bcreate\b.{0,50}\bfile\b",
        r"\bmake\b.{0,50}\bfile\b",
        r"\bwrite\b.{0,50}\bfile\b",
        r"\bgenerate\b.{0,50}\bfile\b",
        r"\bcreate\b.{0,50}\bscript\b",
        r"\bmake\b.{0,50}\bscript\b",
        r"\bwrite\b.{0,50}\bscript\b",
        r"\bsave\b.{0,40}\bas\b.{0,40}\bfile\b",
    ]
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def parse_draft(raw):
    """Parse the JSON draft returned by the LLM."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    start = raw.find("{")
    end = raw.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("The model did not return a valid JSON draft.")

    draft = json.loads(raw[start : end + 1])

    filename = draft.get("filename")
    content = draft.get("content")

    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("The draft has no valid filename.")

    if not isinstance(content, str):
        raise ValueError("The draft has no valid file content.")

    return {"filename": filename.strip(), "content": content}


def is_approval(text):
    return text.lower().strip().rstrip(".!?") in {
        "yes", "yes save", "save", "save it", "approve",
        "approved", "confirm", "confirmed", "go ahead",
        "do it", "proceed", "save the file",
    }


def is_cancellation(text):
    return text.lower().strip().rstrip(".!?") in {
        "no", "cancel", "cancel it", "discard", "discard it",
        "don't save", "do not save", "never mind", "nevermind",
    }


def is_revision(text):
    return text.lower().strip().rstrip(".!?") in {
        "change it", "revise it", "edit it", "modify it",
        "make changes", "try again",
    }


def safe_filename(filename):
    """Reject paths that are clearly unsafe before calling MCP."""
    path = PurePosixPath(filename)
    windows_path = PureWindowsPath(filename)

    if (
        not filename.strip()
        or path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or ".." in path.parts
        or "\\" in filename
        or path.name in {"", ".", ".."}
    ):
        raise ValueError(
            "Use a relative filename inside the workspace, "
            "without parent-directory traversal."
        )

    return filename


async def prepare_draft(request, previous_draft=None, revision=None):
    """Generate a draft without writing anything to disk."""
    previous_draft_text = (
        json.dumps(previous_draft, ensure_ascii=False)
        if previous_draft is not None
        else "None"
    )

    messages = [
        SystemMessage(content=FILE_DRAFT_PROMPT),
        HumanMessage(
            content=(
                f"Original request:\n{request}\n\n"
                f"Previous draft:\n{previous_draft_text}\n\n"
                f"Requested changes:\n{revision or 'None'}\n\n"
                "Prepare the complete file and return the required JSON."
            )
        ),
    ]

    response = await model.ainvoke(messages)
    draft = parse_draft(extract_text(response.content))
    draft["filename"] = safe_filename(draft["filename"])

    return draft


class WednesdayEngine:
    """Core Wednesday AI Assistant Engine that coordinates memory, agent, and MCP tools."""

    def __init__(self):
        self.stack = None
        self.client = None
        self.agent = None
        self.tools = []
        self.create_tool = None
        self.conversation_history = []
        self.pending_draft = None
        self.original_request = None
        self.lock = asyncio.Lock()
        self.initialized = False

    async def initialize(self):
        """Start MCP client, connect to tools, and build the LangChain agent."""
        if self.initialized:
            return

        initialize_memory()
        self.conversation_history = load_history()

        self.client = MultiServerMCPClient(
            {
                "local_files": {
                    "transport": "stdio",
                    "command": sys.executable,
                    "args": ["-m", "src.tools.local_files_server"],
                },
                "android": {
                    "transport": "stdio",
                    "command": sys.executable,
                    "args": ["-m", "src.tools.android_communication_server"],
                },
                "browser": {
                    "transport": "stdio",
                    "command": sys.executable,
                    "args": ["-m", "src.tools.browser_automation_server"],
                },
            }
        )

        self.stack = AsyncExitStack()
        await self.stack.__aenter__()

        browser_session = await self.stack.enter_async_context(
            self.client.session("browser")
        )
        browser_tools = await load_mcp_tools(browser_session)

        file_tools = await self.client.get_tools(server_name="local_files")
        android_tools = await self.client.get_tools(server_name="android")

        self.tools = file_tools + android_tools + browser_tools

        self.create_tool = next(
            (tool for tool in self.tools if tool.name.endswith("create_workspace_file")),
            None,
        )

        agent_tools = [
            tool
            for tool in self.tools
            if not (
                tool.name.endswith("create_workspace_file")
                or tool.name.endswith("edit_workspace_file")
            )
        ]

        self.agent = create_agent(
            model=model,
            tools=agent_tools,
            system_prompt=SYSTEM_PROMPT,
        )

        self.initialized = True

    def get_tools_list(self):
        """Return information about available MCP tools."""
        return [
            {
                "name": tool.name,
                "description": tool.description or "",
            }
            for tool in self.tools
        ]

    def get_history_list(self, limit=50):
        """Load conversation records from SQLite."""
        try:
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT id, role, content, timestamp FROM messages ORDER BY id DESC LIMIT ?",
                    (limit,),
                )
                rows = cursor.fetchall()
                # Return in chronological order
                return [
                    {
                        "id": row[0],
                        "role": row[1],
                        "content": row[2],
                        "timestamp": row[3],
                    }
                    for row in reversed(rows)
                ]
        except Exception:
            return []

    async def approve_pending_draft(self):
        """Approve and save currently pending draft."""
        async with self.lock:
            if not self.pending_draft or not self.create_tool:
                return {
                    "status": "error",
                    "message": "No pending file draft to save.",
                }

            draft = self.pending_draft
            try:
                result = await self.create_tool.ainvoke(
                    {
                        "filename": draft["filename"],
                        "content": draft["content"],
                    }
                )
                result_text = extract_text(result)
                reply = f"Approved. {result_text}"
                save_exchange(self.original_request or "Create file", reply)
                self.conversation_history.extend(
                    [
                        HumanMessage(content=f"Approved file {draft['filename']}"),
                        AIMessage(content=reply),
                    ]
                )
                self.conversation_history = self.conversation_history[-20:]
                self.pending_draft = None
                self.original_request = None
                return {
                    "status": "saved",
                    "filename": draft["filename"],
                    "message": reply,
                }
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Failed to save file: {e}",
                }

    async def cancel_pending_draft(self):
        """Discard currently pending draft."""
        async with self.lock:
            if not self.pending_draft:
                return {
                    "status": "error",
                    "message": "No pending draft to cancel.",
                }
            filename = self.pending_draft["filename"]
            self.pending_draft = None
            self.original_request = None
            reply = f"Discarded draft for {filename}."
            return {
                "status": "cancelled",
                "message": reply,
            }

    async def revise_pending_draft(self, revision_notes: str):
        """Revise the current pending draft based on user feedback."""
        async with self.lock:
            if not self.pending_draft:
                return {
                    "status": "error",
                    "message": "No draft currently pending revision.",
                }
            try:
                new_draft = await prepare_draft(
                    self.original_request,
                    previous_draft=self.pending_draft,
                    revision=revision_notes,
                )
                self.pending_draft = new_draft
                return {
                    "status": "review",
                    "draft": new_draft,
                    "message": f"Updated draft for {new_draft['filename']}. Review proposed changes.",
                }
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Failed to revise draft: {e}",
                }

    async def process_message(self, query: str):
        """Process incoming query through confirmation check or agent execution."""
        async with self.lock:
            query = query.strip()
            if not query:
                return {"type": "chat", "text": "I am waiting."}

            # 1. Handle existing pending draft
            if self.pending_draft is not None:
                if is_approval(query):
                    result = await self.create_tool.ainvoke(
                        {
                            "filename": self.pending_draft["filename"],
                            "content": self.pending_draft["content"],
                        }
                    )
                    reply = extract_text(result)
                    saved_filename = self.pending_draft["filename"]
                    self.pending_draft = None
                    self.original_request = None
                    save_exchange(query, reply)
                    return {
                        "type": "draft_saved",
                        "text": reply,
                        "filename": saved_filename,
                    }

                if is_cancellation(query):
                    cancelled_filename = self.pending_draft["filename"]
                    self.pending_draft = None
                    self.original_request = None
                    reply = f"File creation for {cancelled_filename} was cancelled."
                    save_exchange(query, reply)
                    return {
                        "type": "draft_cancelled",
                        "text": reply,
                        "filename": cancelled_filename,
                    }

                # Otherwise revise draft
                try:
                    self.pending_draft = await prepare_draft(
                        self.original_request,
                        previous_draft=self.pending_draft,
                        revision=query,
                    )
                    reply = (
                        f"I have revised {self.pending_draft['filename']}.\n"
                        "Review the updated content below before approving."
                    )
                    return {
                        "type": "draft_review",
                        "text": reply,
                        "draft": self.pending_draft,
                    }
                except Exception as e:
                    return {
                        "type": "chat",
                        "text": f"Could not revise draft: {e}",
                    }

            # 2. Check if this is a new file creation request
            if is_file_request(query):
                try:
                    draft = await prepare_draft(query)
                    self.pending_draft = draft
                    self.original_request = query
                    reply = (
                        f"I have prepared a draft for {draft['filename']}.\n"
                        "It has not been saved yet. Please review the code below."
                    )
                    return {
                        "type": "draft_review",
                        "text": reply,
                        "draft": draft,
                    }
                except Exception as error:
                    return {
                        "type": "chat",
                        "text": f"I couldn't prepare a valid draft. {error}",
                    }

            # 3. Standard conversational query + MCP tool call
            try:
                result = await self.agent.ainvoke(
                    {
                        "messages": (
                            self.conversation_history
                            + [HumanMessage(content=query)]
                        )
                    }
                )

                final_message = next(
                    (
                        message
                        for message in reversed(result["messages"])
                        if isinstance(message, AIMessage)
                        and not getattr(message, "tool_calls", None)
                        and extract_text(message.content)
                    ),
                    None,
                )

                if final_message is None:
                    return {"type": "chat", "text": "No response was generated."}

                reply = extract_text(final_message.content)
                save_exchange(query, reply)

                self.conversation_history.extend(
                    [
                        HumanMessage(content=query),
                        AIMessage(content=reply),
                    ]
                )
                self.conversation_history = self.conversation_history[-20:]

                return {"type": "chat", "text": reply}

            except Exception as e:
                return {
                    "type": "error",
                    "text": f"Error processing request: {type(e).__name__}: {e}",
                }

    async def close(self):
        """Cleanly close all connected MCP sessions."""
        if self.stack:
            await self.stack.aclose()
            self.initialized = False
