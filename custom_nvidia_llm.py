# custom_nvidia_llm.py
from llama_index.llms.nvidia import NVIDIA
from langchain.chat_models.base import BaseChatModel
from typing import Optional, List, Mapping, Any
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage,
    ChatResult,
    ChatGeneration,
)

class CustomNvidiaLLM(BaseChatModel):
    model_name: str
    llm: Any = None  # Declare llm with a default value

    class Config:
        arbitrary_types_allowed = True  # Allow arbitrary types for llm

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = kwargs.get("model_name")
        self.llm = NVIDIA(model=self.model_name)

    @property
    def _llm_type(self) -> str:
        return "custom_nvidia_llm"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any
    ) -> ChatResult:
        # Convert messages to the format expected by self.llm.chat()
        nvidia_messages = []
        for message in messages:
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            elif isinstance(message, SystemMessage):
                role = "system"
            else:
                role = message.role
            nvidia_messages.append({"role": role, "content": message.content})

        # Call the NVIDIA LLM
        response = self.llm.chat(messages=nvidia_messages)

        # Construct the ChatResult
        ai_message = AIMessage(content=response.message.content)
        generations = [ChatGeneration(message=ai_message)]
        llm_output = {"token_usage": {}}
        return ChatResult(generations=generations, llm_output=llm_output)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_name": self.model_name}
