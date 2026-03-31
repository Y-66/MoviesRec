

from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model

from movies.prompts import INTENT_ANALYZER_PROMPT


model = init_chat_model(
    model_provider="openai",
    model="gpt-4o-mini",
)

intent_analyzer = create_deep_agent(
    model=model,
    system_prompt=INTENT_ANALYZER_PROMPT
)
