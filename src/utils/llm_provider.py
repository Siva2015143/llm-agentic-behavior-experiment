from dotenv import load_dotenv
load_dotenv()


from openai import OpenAI 
import pdb
from langchain_openai import ChatOpenAI
from langchain_core.globals import get_llm_cache
from langchain_core.language_models.base import (
    BaseLanguageModel,
    LangSmithParams,
    LanguageModelInput,
)
from langchain_core.language_models.chat_models import BaseChatModel
import os
import time
from datetime import datetime
import csv
from langchain_core.load import dumpd, dumps
from langchain_core.messages import (
    AIMessage,
    SystemMessage,
    AnyMessage,
    BaseMessage,
    BaseMessageChunk,
    HumanMessage,
    convert_to_messages,
    message_chunk_to_message,
)
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
    LLMResult,
    RunInfo,
)
from langchain_ollama import ChatOllama
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Optional,
    Union,
    cast, List,
    Dict,
)
from langchain_anthropic import ChatAnthropic
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_ibm import ChatWatsonx
from langchain_aws import ChatBedrock
from pydantic import SecretStr
from langchain_openai import ChatOpenAI, AzureChatOpenAI

import google.generativeai as genai

from langchain_core.messages import AIMessage, SystemMessage



from src.utils import config

# METRICS LOGGER ADDED HERE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
METRICS_FILE = os.path.join(BASE_DIR, "agent_compute_metrics.csv")
METRICS_LOG_PATH = os.path.join(BASE_DIR, "agentic_metrics.csv") 


# Experiment Extension: Compute Measurement

def estimate_flops(token_count: int, model_name: str) -> float:
    """Approx FLOP estimation from scaling laws."""
    model_flop_scale = {
        "gpt-4o": 2.8e12,
        "gemini-2.0-flash": 1.5e12,
        "claude-3-5-sonnet": 1.8e12,
        "deepseek-reasoner": 2.5e12,
        "mistral-large-latest": 1.2e12,
        "qwen2.5:7b": 7e9,
        "deepseek-r1:14b": 1.4e10
    }
    base = model_flop_scale.get(model_name, 1e12)
    return token_count * base * 6e-9  # converts to GFLOPs approx.


def log_compute_metrics(provider, model, tokens_in, tokens_out, latency, total_flops):
    """Log compute metrics into CSV."""
    file_exists = os.path.isfile(METRICS_FILE)
    with open(METRICS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "provider", "model", "tokens_in", "tokens_out", "latency_sec", "GFLOPs"])
        writer.writerow([
            datetime.utcnow().isoformat(),
            provider,
            model,
            tokens_in,
            tokens_out,
            round(latency, 3),
            round(total_flops, 3)
        ])


class DeepSeekR1ChatOpenAI(ChatOpenAI):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.client = OpenAI(
            base_url=kwargs.get("base_url"),
            api_key=kwargs.get("api_key")
        )

    async def ainvoke(
            self,
            input: list,
            config: Optional[RunnableConfig] = None,
            *,
            stop: Optional[list[str]] = None,
            **kwargs: Any,
    ) -> AIMessage:
        message_history = []
        for input_ in input:
            if isinstance(input_, SystemMessage):
                message_history.append({"role": "system", "content": input_.content})
            elif isinstance(input_, AIMessage):
                message_history.append({"role": "assistant", "content": input_.content})
            else:
                message_history.append({"role": "user", "content": input_.content})

        # Clean and accurate latency measurement block
        start_time = time.time()
        response = model.invoke([HumanMessage(content=prompt)])

        latency = time.time() - start_time  # ✅ precise total latency

        # Token counting
        tokens_in = sum(len(m["content"].split()) for m in message_history)
        tokens_out = len(response.choices[0].message.content.split())

        # Compute FLOPs and log
        total_flops = estimate_flops(tokens_in + tokens_out, self.model_name)
        log_compute_metrics(
            "deepseek", self.model_name,
            tokens_in, tokens_out, latency, total_flops
        )

        # Token usage capture
        try:
            usage = getattr(response, "usage_metadata", None) or getattr(response, "usage", {})
            input_toks = usage.get("input_token_count", usage.get("prompt_tokens", 0))
            output_toks = usage.get("output_token_count", usage.get("completion_tokens", 0))
        except Exception:
            input_toks, output_toks = 0, 0

        # Safe reasoning extraction
        reasoning_content = getattr(response.choices[0].message, "reasoning_content", "")
        content = response.choices[0].message.content
        return AIMessage(content=content, reasoning_content=reasoning_content)

    def invoke(
        self,
        input: list,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
        ) -> AIMessage:
        message_history = []
        for input_ in input:
            if isinstance(input_, SystemMessage):
               message_history.append({"role": "system", "content": input_.content})
            elif isinstance(input_, AIMessage):
               message_history.append({"role": "assistant", "content": input_.content})
            else:
               message_history.append({"role": "user", "content": input_.content})

        # ✅ Clean and accurate latency measurement block
        start_time = time.time()
        # ✅ Clean and accurate latency measurement block
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_helper",
                    "description": "Performs a simulated web search query.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "calc_helper",
                    "description": "Evaluates a mathematical expression safely.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expr": {"type": "string"}
                        },
                        "required": ["expr"]
                    }
                }
            }
        ]

        start_time = time.time()
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=message_history,
            tools=tools,
            tool_choice="auto"
        )
        latency = time.time() - start_time  # ✅ precise total latency

        # Debug print: to confirm tool-calling
        if hasattr(response.choices[0].message, "tool_calls"):
            print("🔧 Tool calls:", response.choices[0].message.tool_calls)

        latency = time.time() - start_time  # ✅ precise total latency

        # Token counting
        tokens_in = sum(len(m["content"].split()) for m in message_history)
        tokens_out = len(response.choices[0].message.content.split())

        # Compute FLOPs and log
        total_flops = estimate_flops(tokens_in + tokens_out, self.model_name)
        log_compute_metrics(
        "deepseek", self.model_name,
        tokens_in, tokens_out, latency, total_flops  )

        # Token usage capture (safe fallback)
        try:
            usage = getattr(response, "usage_metadata", None) or getattr(response, "usage", {})
            input_toks = usage.get("input_token_count", usage.get("prompt_tokens", 0))
            output_toks = usage.get("output_token_count", usage.get("completion_tokens", 0))
        except Exception:
            input_toks, output_toks = 0, 0
  
         # Safe reasoning extraction
        reasoning_content = getattr(response.choices[0].message, "reasoning_content", "")
        content = response.choices[0].message.content
        return AIMessage(content=content, reasoning_content=reasoning_content)


class DeepSeekR1ChatOllama(ChatOllama):

    async def ainvoke(
            self,
            input: list,
            config: Optional[RunnableConfig] = None,
            *,
            stop: Optional[list[str]] = None,
            **kwargs: Any,
    ) -> AIMessage:
        org_ai_message = await super().ainvoke(input=input)
        org_content = org_ai_message.content
        reasoning_content = org_content.split("</think>")[0].replace("<think>", "")
        content = org_content.split("</think>")[1]
        if "**JSON Response:**" in content:
            content = content.split("**JSON Response:**")[-1]
        return AIMessage(content=content, reasoning_content=reasoning_content)

    def invoke(
            self,
            input: list,
            config: Optional[RunnableConfig] = None,
            *,
            stop: Optional[list[str]] = None,
            **kwargs: Any,
    ) -> AIMessage:
        # ✅ Removed redundant first call
        start = time.time()
        org_ai_message = super().invoke(input=input)
        end = time.time()

        latency = end - start
        tokens_in = sum(len(str(input_)) for input_ in input)
        tokens_out = len(org_ai_message.content.split())
        total_flops = estimate_flops(tokens_in + tokens_out, self.model_name)
        log_compute_metrics("ollama", self.model_name, tokens_in, tokens_out, latency, total_flops)

        org_content = org_ai_message.content
        reasoning_content = org_content.split("</think>")[0].replace("<think>", "")
        content = org_content.split("</think>")[1]
        if "**JSON Response:**" in content:
            content = content.split("**JSON Response:**")[-1]
        return AIMessage(content=content, reasoning_content=reasoning_content)


class GeminiAgenticWrapper(BaseChatModel):
    """LangChain-compatible wrapper for Gemini models (no abstract errors, no pydantic crash)."""

    model_name: str = "gemini-2.0-flash"
    api_key: Optional[str] = None
    temperature: float = 0.7
    model: Optional[object] = None  # ✅ Declare it to avoid 'no field model'

    def __init__(self, model_name="gemini-2.0-flash", api_key=None, temperature=0.7, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.temperature = temperature

        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found. Set it in .env")

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)  # ✅ safe now

    # Required abstract methods
    @property
    def _llm_type(self) -> str:
        return "gemini"

    def _generate(self, messages: List[HumanMessage], stop=None, run_manager=None, **kwargs):
        """Implements LangChain BaseChatModel interface"""
        prompt = "\n".join([m.content for m in messages])
        response = self.model.generate_content(prompt, generation_config={"temperature": self.temperature})
        return AIMessage(content=response.text)
    
    def invoke(self, messages):
        try:
            # Merge all messages into one prompt
            prompt = "\n".join([m.get("content", "") for m in messages if "content" in m]).strip()
            if not prompt:
                prompt = "Hello"  # fallback for empty

            # Generate response
            response = self.model.generate_content(
                prompt,
                generation_config={"temperature": self.temperature},
            )

            # Return clean text
            if hasattr(response, "text"):
                return response.text
            return str(response)
        except Exception as e:
            print(f"[GeminiAgenticWrapper.invoke] Error: {e}")
            return f"Error: {e}"
    

    
    
class ToolLoggingWrapper:
    """
    Robust wrapper around OpenAI Chat with function-calling support and
    compute/latency logging for long-term experiments.
    """

    def __init__(self, model_name: str, api_key: Optional[str] = None, base_url: Optional[str] = None, temperature: float = 0.0):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.temperature = temperature

        # Predefined tools for function-calling
        self.functions = [
            {
                "name": "search_helper",
                "description": "Performs a simulated web search query.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]
                }
            },
            {
                "name": "calc_helper",
                "description": "Evaluates a mathematical expression safely.",
                "parameters": {
                    "type": "object",
                    "properties": {"expr": {"type": "string"}},
                    "required": ["expr"]
                }
            }
        ]

    def _convert_messages(self, input: List[Any]) -> List[Dict[str, str]]:
        """
        Normalize input messages to OpenAI Chat format.
        Accepts SystemMessage, AIMessage, dict, or string.
        """
        messages = []
        for msg in input:
            content = getattr(msg, "content", msg.get("content") if isinstance(msg, dict) else str(msg))
            if isinstance(msg, SystemMessage):
                messages.append({"role": "system", "content": content})
            else:
                messages.append({"role": "user", "content": content})
        return messages

    def invoke(self, input: List[Any], *args, **kwargs) -> AIMessage:
        """
        Perform chat completion call with function support, logging, and robust error handling.
        """
        messages = self._convert_messages(input)
        start_time = time.time()

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                functions=self.functions,
                function_call="auto",
                temperature=self.temperature
            )
        except Exception as e:
            latency = time.time() - start_time
            print(f"[ToolLoggingWrapper] ERROR during invoke: {e}")
            return AIMessage(content=f"[ERROR] {str(e)}")

        latency = time.time() - start_time

        # Extract content and function calls
        choices = getattr(response, "choices", [])
        if choices:
            msg_obj = getattr(choices[0], "message", None)
            text_output = getattr(msg_obj, "content", "")
            tool_calls = getattr(msg_obj, "tool_calls", [])
        else:
            text_output = ""
            tool_calls = []

        # Compute tokens & GFLOPs
        tokens_in = sum(len(m["content"].split()) for m in messages)
        tokens_out = len(text_output.split())
        total_flops = estimate_flops(tokens_in + tokens_out, self.model_name)

        # Log metrics
        log_compute_metrics("local_free", self.model_name, tokens_in, tokens_out, latency, total_flops)

        # Attach tool calls to content for downstream parsing
        if tool_calls:
            text_output += f"\n\n<tool_calls>{tool_calls}</tool_calls>"

        return AIMessage(content=text_output)


def get_llm_model(provider: str, **kwargs):
    """
    Get LLM model
    :param provider: LLM provider
    :param kwargs:
    :return:
    """
    if provider not in ["ollama", "bedrock"]:
        env_var = f"{provider.upper()}_API_KEY"
        api_key = kwargs.get("api_key", "") or os.getenv(env_var, "")
        if not api_key:
            provider_display = config.PROVIDER_DISPLAY_NAMES.get(provider, provider.upper())
            error_msg = f"💥 {provider_display} API key not found! 🔑 Please set the `{env_var}` environment variable or provide it in the UI."
            raise ValueError(error_msg)
        kwargs["api_key"] = api_key

    if provider == "anthropic":
        if not kwargs.get("base_url", ""):
            base_url = "https://api.anthropic.com"
        else:
            base_url = kwargs.get("base_url")

        return ChatAnthropic(
            model=kwargs.get("model_name", "claude-3-5-sonnet-20241022"),
            temperature=kwargs.get("temperature", 0.0),
            base_url=base_url,
            api_key=api_key,
        )
    elif provider == 'mistral':
        if not kwargs.get("base_url", ""):
            base_url = os.getenv("MISTRAL_ENDPOINT", "https://api.mistral.ai/v1")
        else:
            base_url = kwargs.get("base_url")
        if not kwargs.get("api_key", ""):
            api_key = os.getenv("MISTRAL_API_KEY", "")
        else:
            api_key = kwargs.get("api_key")

        return ChatMistralAI(
            model=kwargs.get("model_name", "mistral-large-latest"),
            temperature=kwargs.get("temperature", 0.0),
            base_url=base_url,
            api_key=api_key,
        )
    elif provider == "openai":
        if not kwargs.get("base_url", ""):
            base_url = os.getenv("OPENAI_ENDPOINT", "https://api.openai.com/v1")
        else:
            base_url = kwargs.get("base_url")

        return ChatOpenAI(
            model=kwargs.get("model_name", "gpt-4o"),
            temperature=kwargs.get("temperature", 0.0),
            base_url=base_url,
            api_key=api_key,
        )
    elif provider == "grok":
        if not kwargs.get("base_url", ""):
            base_url = os.getenv("GROK_ENDPOINT", "https://api.x.ai/v1")
        else:
            base_url = kwargs.get("base_url")

        return ChatOpenAI(
            model=kwargs.get("model_name", "grok-3"),
            temperature=kwargs.get("temperature", 0.0),
            base_url=base_url,
            api_key=api_key,
        )
    elif provider == "deepseek":
        # Use free local tool-calling wrapper
        return ToolLoggingWrapper(
            model_name=kwargs.get("model_name", "deepseek-chat"),
            temperature=kwargs.get("temperature", 0.0),
        )
    
        
    elif provider == "google":
        try:
            return GeminiAgenticWrapper(
                model_name=kwargs.get("model_name", "gemini-2.0-flash"),
                temperature=kwargs.get("temperature", 0.0),
                api_key=api_key,
            )
        except Exception as e:
            print(f"[WARN] Gemini model failed to load ({e}), retrying Gemini directly (no fallback).")
            return GeminiAgenticWrapper(
                model_name="gemini-2.0-flash",
                temperature=kwargs.get("temperature", 0.0),
                api_key=api_key,
            )    


    elif provider == "ollama":
        if not kwargs.get("base_url", ""):
            base_url = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
        else:
            base_url = kwargs.get("base_url")

        if "deepseek-r1" in kwargs.get("model_name", "qwen2.5:7b"):
            return DeepSeekR1ChatOllama(
                model=kwargs.get("model_name", "deepseek-r1:14b"),
                temperature=kwargs.get("temperature", 0.0),
                num_ctx=kwargs.get("num_ctx", 32000),
                base_url=base_url,
            )
        else:
            return ChatOllama(
                model=kwargs.get("model_name", "qwen2.5:7b"),
                temperature=kwargs.get("temperature", 0.0),
                num_ctx=kwargs.get("num_ctx", 32000),
                num_predict=kwargs.get("num_predict", 1024),
                base_url=base_url,
            )
    elif provider == "azure_openai":
        if not kwargs.get("base_url", ""):
            base_url = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        else:
            base_url = kwargs.get("base_url")
        api_version = kwargs.get("api_version", "") or os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
        return AzureChatOpenAI(
            model=kwargs.get("model_name", "gpt-4o"),
            temperature=kwargs.get("temperature", 0.0),
            api_version=api_version,
            azure_endpoint=base_url,
            api_key=api_key,
        )
    elif provider == "alibaba":
        if not kwargs.get("base_url", ""):
            base_url = os.getenv("ALIBABA_ENDPOINT", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        else:
            base_url = kwargs.get("base_url")

        return ChatOpenAI(
            model=kwargs.get("model_name", "qwen-plus"),
            temperature=kwargs.get("temperature", 0.0),
            base_url=base_url,
            api_key=api_key,
        )
    elif provider == "ibm":
        parameters = {
            "temperature": kwargs.get("temperature", 0.0),
            "max_tokens": kwargs.get("num_ctx", 32000)
        }
        if not kwargs.get("base_url", ""):
            base_url = os.getenv("IBM_ENDPOINT", "https://us-south.ml.cloud.ibm.com")
        else:
            base_url = kwargs.get("base_url")

        return ChatWatsonx(
            model_id=kwargs.get("model_name", "ibm/granite-vision-3.1-2b-preview"),
            url=base_url,
            project_id=os.getenv("IBM_PROJECT_ID"),
            apikey=os.getenv("IBM_API_KEY"),
            params=parameters
        )
    elif provider == "moonshot":
        return ChatOpenAI(
            model=kwargs.get("model_name", "moonshot-v1-32k-vision-preview"),
            temperature=kwargs.get("temperature", 0.0),
            base_url=os.getenv("MOONSHOT_ENDPOINT"),
            api_key=os.getenv("MOONSHOT_API_KEY"),
        )
    elif provider == "unbound":
        return ChatOpenAI(
            model=kwargs.get("model_name", "gpt-4o-mini"),
            temperature=kwargs.get("temperature", 0.0),
            base_url=os.getenv("UNBOUND_ENDPOINT", "https://api.getunbound.ai"),
            api_key=api_key,
        )
    elif provider == "siliconflow":
        if not kwargs.get("api_key", ""):
            api_key = os.getenv("SiliconFLOW_API_KEY", "")
        else:
            api_key = kwargs.get("api_key")
        if not kwargs.get("base_url", ""):
            base_url = os.getenv("SiliconFLOW_ENDPOINT", "")
        else:
            base_url = kwargs.get("base_url")
        return ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model_name=kwargs.get("model_name", "Qwen/QwQ-32B"),
            temperature=kwargs.get("temperature", 0.0),
        )
    elif provider == "modelscope":
        if not kwargs.get("api_key", ""):
            api_key = os.getenv("MODELSCOPE_API_KEY", "")
        else:
            api_key = kwargs.get("api_key")
        if not kwargs.get("base_url", ""):
            base_url = os.getenv("MODELSCOPE_ENDPOINT", "")
        else:
            base_url = kwargs.get("base_url")
        return ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model_name=kwargs.get("model_name", "Qwen/QwQ-32B"),
            temperature=kwargs.get("temperature", 0.0),
        )
    
    elif provider == "local_free":
        return ToolLoggingWrapper(
            model_name=kwargs.get("model_name", "qwen2.5:7b"),  # or any local free model
            temperature=kwargs.get("temperature", 0.0),
        )


    elif provider == "bedrock":
        return ChatBedrock(
            model=kwargs.get("model_name", "anthropic.claude-3-sonnet-20240229-v1:0"),
            temperature=kwargs.get("temperature", 0.0),
        )
    else:
        raise ValueError(f"❌ Unsupported provider: {provider}")
