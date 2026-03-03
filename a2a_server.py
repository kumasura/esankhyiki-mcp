"""A2A server powered by Google ADK for the MoSPI MCP workflow.

This service exposes a lightweight HTTP API that accepts natural-language prompts,
runs a Google ADK agent, and returns structured responses based on the existing
MoSPI MCP tool chain.
"""

from __future__ import annotations

import os
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from mospi_server import know_about_mospi_api, get_indicators, get_metadata, get_data


class A2ARequest(BaseModel):
    message: str = Field(..., description="User message to process")
    session_id: str = Field(default="default-session", description="Conversation/session identifier")


class A2AResponse(BaseModel):
    session_id: str
    output: Any


def _build_adk_runner():
    """Create a Google ADK runner with tools mapped to existing MCP functions."""
    try:
        from google.adk.agents import Agent
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService
    except ImportError as exc:  # pragma: no cover - runtime dependency guard
        raise RuntimeError(
            "google-adk is not installed. Install dependencies with: pip install -r requirements.txt"
        ) from exc

    instructions = (
        "You are the MoSPI A2A assistant. Always follow the tool workflow exactly: "
        "1_know_about_mospi_api -> 2_get_indicators -> 3_get_metadata -> 4_get_data. "
        "Use returned metadata codes and never guess filters."
    )

    agent = Agent(
        name="mospi_a2a_agent",
        model=os.getenv("ADK_MODEL", "gemini-2.0-flash"),
        instruction=instructions,
        tools=[know_about_mospi_api, get_indicators, get_metadata, get_data],
    )

    app_name = os.getenv("ADK_APP_NAME", "mospi_a2a_server")
    session_service = InMemorySessionService()

    # google-adk Runner constructor has changed across versions:
    # - older: Runner(agent=..., session_service=...)
    # - newer: Runner(app_name=..., agent=..., session_service=...)
    # Prefer the newer signature, then gracefully fall back.
    try:
        return Runner(app_name=app_name, agent=agent, session_service=session_service)
    except (TypeError, ValueError):
        return Runner(agent=agent, session_service=session_service)


app = FastAPI(title="MoSPI A2A Server (Google ADK)", version="0.1.0")
_runner = _build_adk_runner()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/a2a/message", response_model=A2AResponse)
async def process_message(payload: A2ARequest) -> A2AResponse:
    """Process an A2A message via ADK and return the model/tool output."""
    try:
        result = await _runner.run_async(user_input=payload.message, session_id=payload.session_id)
    except Exception as exc:  # pragma: no cover - defensive runtime handling
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    output = getattr(result, "output", None)
    if output is None:
        output = str(result)

    return A2AResponse(session_id=payload.session_id, output=output)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
