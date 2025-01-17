import os

from chains.review_chain import reviews_vector_chain
from langchain import hub
from langchain.agents import AgentExecutor, Tool, create_openai_functions_agent
from langchain_openai import ChatOpenAI


AGENT_MODEL = os.getenv("AGENT_MODEL")

agent_prompt = hub.pull("hwchase17/openai-functions-agent")

tools = [
    Tool(
        name="Dummy",
        func=reviews_vector_chain.invoke,
        description="""Useful all the time.""",
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
