from openai import AsyncOpenAI
import asyncio
# from transformers import AutoTokenizer

from agentlite.llm.LLMConfig import LLMConfig

OPENAI_CHAT_MODELS = [
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k-0613",
    "gpt-3.5-turbo-16k",
    "gpt-4",
    "gpt-4-0613",
    "gpt-4-turbo",
    "gpt-4-32k",
    "gpt-4-32k-0613",
    "gpt-4-1106-preview",
    "gpt-4o"
]

from abc import abstractmethod
import abc
import asyncio

class BaseLLM(abc.ABC):
    def __init__(self, llm_config: LLMConfig):
        self.is_open = llm_config.is_open
        self.llm_name = llm_config.llm_name
        self.llm_dir = llm_config.llm_dir
        self.context_len: int = llm_config.context_len
        self.stop: list = llm_config.stop
        self.max_tokens: int = llm_config.max_tokens
        self.temperature: float = llm_config.temperature
        self.end_of_prompt: str = llm_config.end_of_prompt
    
    def get_open(self):
        return self.is_open 

    def get_modelname(self, model_name):
        return self.model_name
    
    @classmethod
    @abstractmethod
    async def run_with_stream(self, query:str, k=10, stop_judge = None):
        pass

    @classmethod
    @abstractmethod
    async def run(self, query:str): #get full output tokens
        return NotImplementedError

    def append_user_input(self, origin, new):
        origin.append({"role": "user", "content": new})
        return origin

    def append_system_input(self, origin, new):
        origin = [{"role": "system", "content": new}]
        return origin
    
    def append_assistant_input(self, origin, new):
        origin.append({"role": "assistant", "content": new})
        return origin


class OpenAIChatLLM(BaseLLM):
    def __init__(self, llm_config:LLMConfig):
        super().__init__(llm_config)
        self.client:AsyncOpenAI = AsyncOpenAI(api_key=llm_config.api_key, base_url=llm_config.base_url)

    async def run_with_stream(self, query, k=10, stop_judge = None):
        return NotImplementedError

    async def run(self, query):
        response = await self.client.chat.completions.create(
            model=self.llm_name,
            messages=query,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message['content']

class OpenLLM(BaseLLM):
    def __init__(self, llm_config:LLMConfig):
        super().__init__(llm_config)
        # self.tokenizer = AutoTokenizer.from_pretrained(llm_config.llm_dir)
        openai_api_key = "EMPTY"
        openai_api_base = llm_config.base_url
        self.client = AsyncOpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        
    async def get_first_tokens(self, query, k=10, stop_judge = None): #query can be in the form of tokens
        response = self.client.chat.completions.create(
            model=self.llm_name, #gpt-4-turbo , gpt-4
            messages=query,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True
        ) 
        response = ""
        judge_criteria = k

        async for chunk in response:
            response += chunk.choices[0].delta.content
            if (len(response) > judge_criteria):
                # response_str = self.tokenizer.decode(response, skip_special_tokens=True)
                if stop_judge is not None and stop_judge(response):
                    response.close()
                    return None, None
                judge_criteria = 0x7fffffff # No longer should be involved

        # response_str = self.tokenizer.decode(response, skip_special_tokens=True)
        return response

    async def get_output_tokens(self, query):
        completion = await self.client.chat.completions.create(
            model=self.llm_name,
            messages=query,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        response = completion.choices[0].message['content']
        # response_str = self.tokenizer.decode(response, skip_special_tokens=True)
        return response


def get_llm_backend(llm_config: LLMConfig):
    llm_name = llm_config.llm_name
    llm_provider = llm_config.provider
    if llm_name in OPENAI_CHAT_MODELS:
        return OpenAIChatLLM(llm_config)
    else:
        return OpenLLM(llm_config)
