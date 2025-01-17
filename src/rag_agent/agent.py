import os

from langchain import hub
from langchain.agents import AgentExecutor, Tool, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from src.rag_agent.tools import get_product_description


AGENT_MODEL = os.getenv("AGENT_MODEL")

agent_prompt = hub.pull("hwchase17/openai-functions-agent")

tools = [
    Tool(
        name="Dummy",
        func=get_product_description,
        description="""Useful when you are asked about the description or
        characteristics of a specific product. This tool can only get the description
        of one product at a time and nothing else. Pass only the product id as input.
        For example, if the prompt is "Tell me about product B01CUPMQZE", the
        input should be "B01CUPMQZE".
        """,
    ),
]

chat_model = ChatOpenAI(
    model=AGENT_MODEL,
    temperature=0,
)

rag_agent = create_openai_functions_agent(
    llm=chat_model,
    prompt=agent_prompt,
    tools=tools,
)

rag_agent_executor = AgentExecutor(
    agent=rag_agent,
    tools=tools,
    return_intermediate_steps=True,
    verbose=True,
)
