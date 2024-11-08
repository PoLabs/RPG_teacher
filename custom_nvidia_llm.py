from typing import Any, Optional, List, Mapping, Dict
from langchain.chat_models.base import BaseChatModel
from langchain.schema import AIMessage, ChatResult, ChatGeneration
from langchain.callbacks.manager import CallbackManagerForLLMRun
from llama_index.llms.nvidia import NVIDIA

class CustomNvidiaLLM(BaseChatModel):
    model_name: str
    llm: Any = None  # Declare llm with a default value

    class Config:
        arbitrary_types_allowed = True  # Allows arbitrary types like custom classes

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = kwargs.get("model_name")
        self.llm = NVIDIA(model=self.model_name)

    @property
    def _llm_type(self) -> str:
        return "custom_nvidia_llm"

    def _generate(
        self,
        messages: List[Dict[str, str]],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any
    ) -> ChatResult:
        # Call the NVIDIA LLM
        response = self.llm.chat(messages=messages)

        # Extract the assistant's reply
        assistant_content = response.message.content  # Adjust based on your LLM's response
        ai_message = AIMessage(content=assistant_content)
        generations = [ChatGeneration(message=ai_message)]
        llm_output = {"token_usage": {}}  # Include any additional info if available

        return ChatResult(generations=generations, llm_output=llm_output)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_name": self.model_name}
