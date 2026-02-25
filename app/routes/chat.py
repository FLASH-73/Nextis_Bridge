import json
import os

from fastapi import APIRouter, BackgroundTasks, Request
from pydantic import BaseModel

from app.dependencies import get_state

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    message: str


@router.post("/execute")
async def execute_endpoint(request: Request, background_tasks: BackgroundTasks):
    system = get_state()
    if not system.orchestrator:
        return {"status": "error", "message": "Orchestrator not initialized"}
    data = await request.json()
    plan = data.get("plan")
    if not plan:
        return {"status": "error", "message": "No plan provided"}

    background_tasks.add_task(system.orchestrator.execute_plan, plan)
    return {"status": "success", "message": "Execution started"}


@router.post("/chat")
async def chat_endpoint(request: Request):
    system = get_state()
    data = await request.json()
    user_msg = data.get("message", "").lower()
    messages = data.get("messages", []) # Full conversation history

    # Lazy-load planner on first use
    if system.planner is None:
        try:
            from app.core.planner import GeminiPlanner, LocalPlanner
            gemini_key = os.getenv("GEMINI_API_KEY")
            if gemini_key and gemini_key.strip():
                print("Lazy-loading GeminiPlanner...")
                system.planner = GeminiPlanner(api_key=gemini_key)
            else:
                print("Lazy-loading LocalPlanner (Qwen)...")
                system.planner = LocalPlanner("Qwen/Qwen2.5-7B-Instruct", device="cuda")
        except Exception as e:
            print(f"Warning: Planner load failed: {e}")

    # Simple keyword matching for MVP
    actions = []
    response = ""
    plan = []

    if "stop" in user_msg:
        actions = []
        response = "Stopping all tasks."
        if system.orchestrator:
            system.orchestrator.stop()
            system.orchestrator.task_chain = []
    else:
        if system.planner:
            print(f"Planning for: {user_msg}")

            # Use history if available, otherwise just the message
            input_data = messages if messages else user_msg

            plan = system.planner.plan(input_data)
            print(f"Task chain loaded: {json.dumps(plan, indent=2)}")

            # Orchestrator expects a list of strings for now, or we need to upgrade Orchestrator
            # For now, let's extract the task names or format them
            actions = [p.get("task") for p in plan]

            # CRITICAL: The response content MUST be the JSON string so it gets added to the chat history
            # this allows the planner to 'see' its previous plans in future turns.
            response = json.dumps(plan)
        else:
            # Fallback
            actions = ["move_to_bin", "pick_object", "place_in_box"]
            response = "Planner not ready. Using dummy plan."

    # Send tasks to orchestrator
    if actions and system.orchestrator:
        system.orchestrator.load_task_chain(actions)

    return {
        "reply": response, # Frontend expects 'reply'
        "response": response, # Legacy
        "actions": actions,
        "plan": plan if system.planner else [] # For visualizer
    }
