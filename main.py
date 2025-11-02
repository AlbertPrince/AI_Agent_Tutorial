from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor

load_dotenv()


class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]
    

llm = ChatOpenAI(model_name="gpt-4o-mini")
# llm2 = ChatAnthropic(model="claude-4-5-sonne-20241022")

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt_template = ChatPromptTemplate.from_messages([
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper. 
            Answer the user query and use necessary tools. 
            Wrap the output in this format and provide no other text \n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
]).partial(format_instructions=parser.get_format_instructions())

agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt_template,
    # output_parser=parser,
    tools=[],
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=[],
    verbose=True,
)

raw_response = agent_executor.invoke(
    {"query": "Summarize the latest research on quantum computing."}
)

# response = llm.invoke("What is the meaning of life?")
# response2 = llm2.invoke("Tell me a joke about programming.")

# print("Response from OpenAI GPT-4:", response)
# print("Response from Anthropic Claude-3-5-Sonnet:", response2)