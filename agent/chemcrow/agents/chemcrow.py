from typing import Optional

import langchain
from dotenv import load_dotenv
from langchain import PromptTemplate, chains
from langchain.chat_models import ChatOllama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from pydantic import ValidationError
from rmrkl import ChatZeroShotAgent, RetryAgentExecutor

from chemcrow.agents.agent_core import RetroPlannerChatZeroShotOutputParser

from .prompts import FORMAT_INSTRUCTIONS, QUESTION_PROMPT, REPHRASE_TEMPLATE, SUFFIX
from .tools import make_tools


def _make_llm(
    model,
    temp,
    api_key,
    streaming: bool = False,
    ollama_base_url: str = "http://localhost:11434",
    cache=True,
):
    if model.startswith("gpt-3.5-turbo") or model.startswith("gpt-4"):
        llm = langchain.chat_models.ChatOpenAI(
            temperature=temp,
            model_name=model,
            request_timeout=1000,
            streaming=streaming,
            callbacks=[StreamingStdOutCallbackHandler()],
            openai_api_key=api_key,
            cache=cache,
        )
    elif model.startswith("text-"):
        llm = langchain.OpenAI(
            temperature=temp,
            model_name=model,
            streaming=streaming,
            callbacks=[StreamingStdOutCallbackHandler()],
            openai_api_key=api_key,
            cache=cache,
        )
    elif model.startswith("llama"):
        llm = ChatOllama(
            temperature=temp,
            model=model,
            base_url=ollama_base_url,
            streaming=streaming,
            callbacks=[StreamingStdOutCallbackHandler()],
            openai_api_key=api_key,
            cache=cache,
            num_ctx=20480,   # Important !
        )
    else:
        raise ValueError(f"Invalid model name: {model}")
    return llm


class ChemCrow:
    def __init__(
        self,
        tools=None,
        model="gpt-4-0613",
        tools_model="gpt-3.5-turbo-0613",
        temp=0.1,
        max_iterations=40,
        verbose=True,
        streaming: bool = True,
        openai_api_key: Optional[str] = None,
        api_keys: dict = {},
        ollama_base_url: str = "http://localhost:11434",
        retroplanner_base_url: str = "http://localhost:8001",
        llm_cache=True,
    ):
        """Initialize ChemCrow agent."""

        load_dotenv()
        try:
            if model.startswith("llama"):
                self.llm = _make_llm(
                    model, temp, None, streaming, ollama_base_url=ollama_base_url, cache=llm_cache,
                )
            else:
                self.llm = _make_llm(model, temp, openai_api_key, streaming, cache=llm_cache)
        except ValidationError:
            raise ValueError("Invalid base url")

        self.retroplanner_base_url = retroplanner_base_url

        if tools is None:
            api_keys["OPENAI_API_KEY"] = openai_api_key
            if model.startswith("llama"):
                tools_llm = _make_llm(
                    tools_model, temp, None, streaming, ollama_base_url=ollama_base_url, cache=llm_cache
                )
                tools = make_tools(
                    tools_llm,
                    api_keys=api_keys,
                    verbose=verbose,
                    retroplanner_base_url=retroplanner_base_url,
                )
            else:
                # api_keys["OPENAI_API_KEY"] = openai_api_key
                tools_llm = _make_llm(tools_model, temp, openai_api_key, streaming, cache=llm_cache)
                tools = make_tools(
                    tools_llm,
                    api_keys=api_keys,
                    verbose=verbose,
                    retroplanner_base_url=retroplanner_base_url,
                )

        # Initialize agent
        self.agent_executor = RetryAgentExecutor.from_agent_and_tools(
            tools=tools,
            agent=ChatZeroShotAgent.from_llm_and_tools(
                self.llm,
                tools,
                suffix=SUFFIX,
                format_instructions=FORMAT_INSTRUCTIONS,
                question_prompt=QUESTION_PROMPT,
                output_parser=RetroPlannerChatZeroShotOutputParser(),
            ),
            verbose=True,
            max_iterations=max_iterations,
        )

        rephrase = PromptTemplate(
            input_variables=["question", "agent_ans"], template=REPHRASE_TEMPLATE
        )

        self.rephrase_chain = chains.LLMChain(prompt=rephrase, llm=self.llm)

    def run(self, prompt):
        outputs = self.agent_executor({"input": prompt})
        return outputs["output"]
