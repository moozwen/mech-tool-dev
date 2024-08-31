# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024 Valory AG
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------

from typing import Dict, Optional, Literal, List, Any, Callable
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
    RunnableSerializable,
)
from langchain_core.output_parsers import StrOutputParser
import os
from functools import partial


class ChatOpenRouter(ChatOpenAI):
    def __init__(
        self,
        model_name: str,
        openrouter_api_key: Optional[str] = None,
        openrouter_api_base: str = "https://openrouter.ai/api/v1",
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            openai_api_key=openrouter_api_key or os.getenv("OPENROUTER_API_KEY"),
            openai_api_base=openrouter_api_base,
            **kwargs,
        )


class AgentFactory:
    # ToDo: いずれか一つのAPIキーでも受け付けられるようにする。
    def __init__(
        self,
        openrouter_api_key: str,
    ):
        self.openrouter_api_kei = openrouter_api_key
        self.groq_api_key = groq_api_key

    def create_openrouter_agent(self, model_name: str, **kwargs):
        return ChatOpenRouter(
            model_name=model_name,
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
            **kwargs
        )

    def crate_groq_agent(self, model_name: str, **kwargs):
        return ChatGroq(model=model_name, **kwargs)


# メイン処理は、MoAgentインスタンスを受け取って、MoAのコアロジックを実行する
class MoAgent:
    def __init__(
        self,
        main_agent: RunnableSerializable[Dict, str],
        layer_agent: RunnableSerializable[Dict, Dict]
    ):
        self.main_agent = main_agent
        self.layer_agent = layer_agent


def configure_layer_agents(
    layer_agent_config: Optional[Dict] = None,
    provider: Literal["groq", "openrouter"] = "groq",
) -> RunnableSerializable[Dict, Dict]:
    if layer_agent_config is None:
        layer_agent_config = get_default_layer_agent_config(provider=provider)

    agent_factory = AgentFactory(
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    create_agent_func = agent_factory.crate_groq_agent if provider == "groq" else agent_factory.create_openrouter_agent
    parallel_chain_map = {
        key: RunnablePassthrough() | partial(create_agent, create_client_func=create_agent_func, **value)
        for key, value in layer_agent_config.items()
    }

    return parallel_chain_map | RunnableLambda(concat_responses)


def get_default_layer_agent_config(provider: str) -> Dict:
    if provider == "groq":
        return {
            "layer_agent_1": {
                "system_prompt": SYSTEM_PROMPT,
                "model_name": "llama3-8b-8192"
            },
            "layer_agent_2": {
                "system_prompt": SYSTEM_PROMPT,
                "model_name": "gemma-7b-it"
            },
            "layer_agent_3": {
                "system_prompt": SYSTEM_PROMPT,
                "model_name": "mixtral-8x7b-32768"
            }
        }
    else:
        return {
            "layer_agent_1": {
                "system_prompt": SYSTEM_PROMPT,
                "model_name": "meta-llama/llama-3-8b-instruct:free"
            },
            "layer_agent_2": {
                "system_prompt": SYSTEM_PROMPT,
                "model_name": "google/gemma-2-9b-it:free"
            },
            "layer_agent_3": {
                "system_prompt": SYSTEM_PROMPT,
                "model_name": "nousresearch/nous-capybara-7b:free"
            },
        }


def create_agent(
    system_prompt: str,
    create_agent_func: Callable[[str], Any],
    model_name: str,
    **llm_kwargs,
) -> RunnableSerializable[Dict, str]:
    prompt = build_prompt_template(system_prompt)
    llm = create_agent_func(model_name=model_name, **llm_kwargs)
    return prompt | llm | StrOutputParser()


def build_prompt_template(system_prompt: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages", optional=True),
        ("human", "{input}")
    ])


def concat_responses(
    inputs: Dict[str, str],
    reference_system_prompt: Optional[str] = None
) -> Dict:
    reference_system_prompt = reference_system_prompt or REFERENCE_SYSTEM_PROMPT
    responses = "\n".join([f"{i}. {out}" for i, out in enumerate(inputs.values())])
    formatted_prompt = reference_system_prompt.format(responses=responses)
    return {
        "formatted_response": formatted_prompt,
        "responses": list(inputs.values())
    }


SYSTEM_PROMPT = """\
You are a personal assistant that is helpful.
{helper_response}\
"""

REFERENCE_SYSTEM_PROMPT = """\
You have been provided with a set of responses from various open-source models to the latest user query.
Your task is to synthesize these responses into a single, high-quality response.
It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect.
Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction.
Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.
Responses from models:
{responses}
"""


class MOAgentConfig(BaseModel):
    main_model: Optional[str] = None
    system_prompt: Optional[str] = None
    cycles: int = Field(...)
    layer_agent_config: Optional[Dict[str, Any]] = None
    reference_system_prompt: Optional[str] = None
    max_tokens: Optional[int] = None


load_dotenv()

valid_model_names = Literal[
    "llama3-70b-8192",
    "llama3-8b-8192",
    "gemma-7b-it",
    "gemma2-9b-it",
    "mixtral-8x7b-32768",
    "llama-3.1-8b-instant",
    "llama-3.1-70b-versatile",
]

or_valid_model_names = Literal[
    "meta-llama/llama-3-8b-instruct:free",
    "google/gemma-2-9b-it:free",
    "nousresearch/nous-capybara-7b:free",
]


class MOAgent:
    def __init__(
        self,
        main_agent: RunnableSerializable[Dict, str],
        layer_agent: RunnableSerializable[Dict, Dict],
        reference_system_prompt: Optional[str] = None,
        cycles: Optional[int] = None,
        chat_memory: Optional[ConversationBufferMemory] = None,
    ) -> None:
        self.reference_system_prompt = (
            reference_system_prompt or REFERENCE_SYSTEM_PROMPT
        )
        self.main_agent = main_agent
        self.layer_agent = layer_agent
        self.cycles = cycles or 1
        self.chat_memory = chat_memory or ConversationBufferMemory(
            memory_key="messages", return_messages=True
        )

    @staticmethod
    def concat_response(
        inputs: Dict[str, str], reference_system_prompt: Optional[str] = None
    ):
        reference_system_prompt = reference_system_prompt or REFERENCE_SYSTEM_PROMPT

        responses = ""
        res_list = []
        for i, out in enumerate(inputs.values()):
            responses += f"{i}. {out}\n"
            res_list.append(out)

        formatted_prompt = reference_system_prompt.format(responses=responses)
        return {"formatted_response": formatted_prompt, "responses": res_list}

    @classmethod
    def from_config(
        cls,
        main_model: Optional[valid_model_names] = "llama3-70b-8192",
        or_main_model: Optional[
            or_valid_model_names
        ] = "meta-llama/llama-3-8b-instruct:free",
        system_prompt: Optional[str] = None,
        cycles: int = 1,
        layer_agent_config: Optional[Dict] = None,
        reference_system_prompt: Optional[str] = None,
        **main_model_kwargs,
    ):
        reference_system_prompt = reference_system_prompt or REFERENCE_SYSTEM_PROMPT
        system_prompt = system_prompt or SYSTEM_PROMPT
        layer_agent = MOAgent._configure_layer_agent(layer_agent_config)
        main_agent = MOAgent._create_agent_from_system_prompt(
            system_prompt=system_prompt,
            model_name=main_model,
            or_model_name=or_main_model,
            **main_model_kwargs,
        )

        return cls(
            main_agent=main_agent,
            layer_agent=layer_agent,
            reference_system_prompt=reference_system_prompt,
            cycles=cycles,
        )

    @staticmethod
    def _configure_layer_agent(
        layer_agent_config: Optional[Dict] = None,
        or_layer_agent_config: Optional[Dict] = None,
    ) -> RunnableSerializable[Dict, Dict]:
        if not layer_agent_config:
            layer_agent_config = {
                "layer_agent_1": {
                    "system_prompt": SYSTEM_PROMPT,
                    "model_name": "llama3-8b-8192",
                },
                "layer_agent_2": {
                    "system_prompt": SYSTEM_PROMPT,
                    "model_name": "gemma-7b-it",
                },
                "layer_agent_3": {
                    "system_prompt": SYSTEM_PROMPT,
                    "model_name": "mixtral-8x7b-32768",
                },
            }

        if not or_layer_agent_config:
            or_layer_agent_config = {
                "layer_agent_1": {
                    "system_prompt": SYSTEM_PROMPT,
                    "model_name": "meta-llama/llama-3-8b-instruct:free",
                },
                "layer_agent_2": {
                    "system_prompt": SYSTEM_PROMPT,
                    "model_name": "google/gemma-2-9b-it:free",
                },
                "layer_agent_3": {
                    "system_prompt": SYSTEM_PROMPT,
                    "model_name": "nousresearch/nous-capybara-7b:free",
                },
            }

        parallel_chain_map = dict()

        for key, value in or_layer_agent_config.items():
            chain = MOAgent._create_agent_from_system_prompt(
                system_prompt=value.pop("system_prompt", SYSTEM_PROMPT),
                or_model_name=value.pop(
                    "model_name", "meta-llama/llama-3-8b-instruct:free"
                ),
                **value,
            )
            parallel_chain_map[key] = RunnablePassthrough() | chain

        chain = parallel_chain_map | RunnableLambda(MOAgent.concat_response)

        return chain

    @staticmethod
    def _create_agent_from_system_prompt(
        system_prompt: str = SYSTEM_PROMPT,
        model_name: str = "llama3-8b-8192",
        or_model_name: str = "meta-llama/llama-3-8b-instruct:free",
        **llm_kwargs,
    ) -> RunnableSerializable[Dict, str]:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="messages", optional=True),
                ("human", "{input}"),
            ]
        )

        assert "helper_response" in prompt.input_variables

        llm = ChatGroq(model=model_name, **llm_kwargs)

        # llm = ChatOpenRouter(model_name=or_model_name)

        chain = prompt | llm | StrOutputParser()

        return chain






    # def chat(
    #     self,
    #     input: str,
    #     messages: Optional[List[BaseMessage]] = None,
    #     cycles: Optional[int] = None,
    #     save: bool = True,
    #     output_format: Literal["string", "json"] = "string",
    # ) -> str:
    #     cycles = cycles or self.cycles

    #     llm_inp = {
    #         "input": input,
    #         "messages": messages
    #         or self.chat_memory.load_memory_variables({})["messages"],
    #         "helper_response": "",
    #     }

    #     response = ""

    #     for cyc in range(cycles):
    #         layer_output = self.layer_agent.invoke(llm_inp)
    #         l_frm_resp = layer_output["formatted_response"]
    #         # l_resps = layer_output["responses"]

    #         llm_inp = {
    #             "input": input,
    #             "messages": self.chat_memory.load_memory_variables({})["messages"],
    #             "helper_response": l_frm_resp,
    #         }

    #         stream = self.main_agent.stream(llm_inp)
    #         for chunk in stream:
    #             response += chunk

    #     if save:
    #         self.chat_memory.save_context({"input": input}, {"output": response})

    #     return response
