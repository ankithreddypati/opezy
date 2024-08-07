"""
API entrypoint for backend API.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api_models.ai_request import AIRequest
from opezy.opezy_ai_agent import OpezyAIAgent

app = FastAPI()

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Agent pool keyed by session_id to retain memories/history in-memory.
# Note: the context is lost every time the service is restarted.
agent_pool = {}

@app.get("/")
def root():
    """
    Health probe endpoint.
    """
    return {"status": "ready"}

@app.post("/ai")
def run_opezy_works_ai_agent(request: AIRequest):
    """
    Run the Opezy Works AI agent.
    """
    if request.session_id not in agent_pool:
        agent_pool[request.session_id] = OpezyAIAgent(request.session_id)
    return { "message": agent_pool[request.session_id].run(request.prompt) }
