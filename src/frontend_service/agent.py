# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from datetime import date
from typing import Dict

import aiohttp
from langchain.agents import AgentType, initialize_agent
from langchain.agents.agent import AgentExecutor
from langchain.globals import set_verbose  # type: ignore
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core import messages
from langchain_google_vertexai import VertexAI

from tools import initialize_tools

set_verbose(bool(os.getenv("DEBUG", default=False)))

# URL to connect to the backend service
BASE_URL = os.getenv("BASE_URL", default="http://127.0.0.1:8080")

# aiohttp context
connector = None

CLOUD_RUN_AUTHORIZATION_TOKEN = None


# Class for setting up a dedicated llm agent for each individual user
class UserAgent:
    client: aiohttp.ClientSession
    agent: AgentExecutor

    def __init__(self, client, agent) -> None:
        self.client = client
        self.agent = agent


user_agents: Dict[str, UserAgent] = {}


async def get_connector():
    global connector
    if connector is None:
        connector = aiohttp.TCPConnector(limit=100)
    return connector


async def handle_error_response(response):
    if response.status != 200:
        return f"Error sending {response.method} request to {str(response.url)}): {await response.text()}"


async def create_client_session() -> aiohttp.ClientSession:
    return aiohttp.ClientSession(
        connector=await get_connector(),
        connector_owner=False,
        headers={},
        raise_for_status=True,
    )


# Agent
async def init_agent(history: list[messages.BaseMessage]) -> UserAgent:
    """Load an agent executor with tools and LLM"""
    print("Initializing agent..")
    llm = VertexAI(max_output_tokens=512, model_name="gemini-1.5-pro-001")
    memory = ConversationBufferMemory(
        chat_memory=ChatMessageHistory(messages=history),
        memory_key="chat_history",
        input_key="input",
        output_key="output",
    )
    client = await create_client_session()
    tools = await initialize_tools(client)
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        memory=memory,
        handle_parsing_errors=True,
        max_iterations=3,
        early_stopping_method="generate",
        return_intermediate_steps=True,
    )
    # Create new prompt template
    tool_strings = "\n".join([f"> {tool.name}: {tool.description}" for tool in tools])
    tool_names = ", ".join([tool.name for tool in tools])
    format_instructions = FORMAT_INSTRUCTIONS.format(
        tool_names=tool_names,
    )
    today_date = date.today().strftime("%Y-%m-%d")
    today = f"Today is {today_date}."
    template = "\n\n".join([PREFIX, tool_strings, format_instructions, SUFFIX, today])
    human_message_template = "{input}\n\n{agent_scratchpad}"
    prompt = ChatPromptTemplate.from_messages(
        [("system", template), ("human", human_message_template)]
    )
    agent.agent.llm_chain.prompt = prompt  # type: ignore

    return UserAgent(client, agent)


PREFIX = """Michael's Assistant helps users to find out about Michael.
Assistant is designed to be able to answer accurate and verified information about Michael.
Assistant do not respond to irrelevant or nonsensical questions.
Assistant use any provided context about Michael's experience, such as work experience, past publications, and personal project.
Assistant do not mention that the context was used to generate the response. 
Assistant only include information directly relevant to the user's inquiry.
"""

SUFFIX = """Previous conversation history:
{chat_history}
"""
