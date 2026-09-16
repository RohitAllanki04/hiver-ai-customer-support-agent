from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.support_agent import SupportAgent


app = FastAPI(
    title="Amazon Support AI Agent",
    description="AI-powered customer support agent for AmazonHelp",
    version="1.0.0"
)


agent = SupportAgent()


class SupportRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=5,
        max_length=2000,
        description="Customer support message"
    )


class SupportResponse(BaseModel):
    message: str
    intent: str
    reply: str
    valid: bool
    validation_issues: list[str]
    decision: str
    decision_reason: str

@app.get("/health")
def health_check():
    return {
        "status": "ok"
    }


@app.post("/api/v1/support/reply", response_model=SupportResponse)
def generate_support_reply(request: SupportRequest):

    result = agent.generate_reply(request.message)

    return {
        "message": result["customer_message"],
        "intent": result["intent"],
        "reply": result["reply"],
        "valid": result["valid"],
        "validation_issues": result["validation_issues"],
        "decision": result["decision"],
        "decision_reason": result["decision_reason"]
    }