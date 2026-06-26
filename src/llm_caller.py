from dataclasses import dataclass, field
from typing import List, Optional
from openai import OpenAI
import os
from dotenv import load_dotenv; load_dotenv(); api_key = os.getenv("DEEPS_KEY")

@dataclass
class DeepSeekMessage:
    role: str
    content: str

@dataclass
class DeepSeekResponse:
    content: str

@dataclass
class DeepSeekConfig:
    model: str = "deepseek-chat"
    temperature: float = 1.0
    max_tokens: int = 200

@dataclass
class DeepSeekCaller:
    api_key: str = api_key
    config: DeepSeekConfig = field(default_factory=DeepSeekConfig)
    client: OpenAI = field(init=False)
    
    def __post_init__(self):
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")
    
    def _call(self, messages: List[DeepSeekMessage], system_prompt: Optional[str] = None) -> DeepSeekResponse:

        api_messages = []

        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})

        for msg in messages:
            api_messages.append({"role": msg.role, "content": msg.content})

        call_params = {
            "model": self.config.model,
            "messages": api_messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        response = self.client.chat.completions.create(**call_params)

        content = response.choices[0].message.content
        
        return DeepSeekResponse(content=content)
    
    def call(self, prompt: str, system_prompt: Optional[str] = None) -> DeepSeekResponse:

        messages = [DeepSeekMessage(role="user", content=prompt)]
        return self._call(messages, system_prompt)