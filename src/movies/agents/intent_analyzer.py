
from typing import Any, Dict

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, SystemMessage

from movies.prompts import INTENT_ANALYZER_PROMPT
from movies.states.state import IntentResult


class IntentAnalyzerAgent:
    def __init__(self) -> None:
        raw_model = init_chat_model(
            model_provider="openai",
            model="gpt-4o-mini",
        )
        self.model = raw_model.with_structured_output(IntentResult)

    def invoke(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        incoming_messages = payload.get("messages", [])
        messages = [SystemMessage(content=INTENT_ANALYZER_PROMPT)] + incoming_messages
        response = self.model.invoke(messages)

        return {"intent_data": response}


intent_analyzer = IntentAnalyzerAgent()
