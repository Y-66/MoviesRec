from typing import Any, Dict, Iterator

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, SystemMessage


class SummarizerAgent:
    def __init__(self) -> None:
        self.model = init_chat_model(
            model_provider="openai",
            model="gpt-4o",
        )

    def invoke(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        incoming_messages = payload.get("messages", [])
        response = self.model.invoke(incoming_messages)

        if isinstance(response, AIMessage):
            return {"messages": [response]}
        return {"messages": [AIMessage(content=str(response))]}

    def stream(self, payload: Dict[str, Any]) -> Iterator[str]:
        incoming_messages = payload.get("messages", [])
        for chunk in self.model.stream(incoming_messages):
            piece = getattr(chunk, "content", "")
            if isinstance(piece, str):
                if piece:
                    yield piece
            elif isinstance(piece, list):
                text = " ".join(str(item) for item in piece).strip()
                if text:
                    yield text
            else:
                text = str(piece).strip()
                if text:
                    yield text


summarizer = SummarizerAgent()
