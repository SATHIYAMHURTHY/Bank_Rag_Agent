"""
graph.py
--------
LangGraph agent that decides which retrieval tool to call based on the user query.
LLM-agnostic — accepts any LangChain chat model from llm_factory.py.
"""

from langgraph.prebuilt import create_react_agent
from src.agent.tools import TOOLS


def build_agent(llm):
    """
    Build and return a LangGraph ReAct agent.

    Args:
        llm: any LangChain BaseChatModel (from llm_factory.get_llm)

    Returns:
        A compiled LangGraph agent (callable with {"messages": [...]})
    """
    system_prompt = """You are a helpful banking assistant specialising in education loans.
You have access to a knowledge base containing education loan information from three Indian banks: HDFC, ICICI, and SBI.

You have three tools available:
- search_bank_policy: use when the user asks about ONE specific bank
- compare_schemes_across_banks: use when the user wants to compare banks or asks a general question not tied to a specific bank
- list_available_banks_and_schemes: use when the user asks what banks or schemes are available

Guidelines:
- Always use a tool before answering — never answer from memory alone
- For comparison questions, always use compare_schemes_across_banks so all banks get equal representation
- Be concise and structured in your responses
- If information is not in the knowledge base, say so clearly
- Present interest rates, fees, and eligibility criteria in an easy-to-read format
- Always mention which bank each piece of information comes from
"""

    agent = create_react_agent(
        model=llm,
        tools=TOOLS,
        prompt=system_prompt,
    )
    return agent


def run_agent(agent, user_message: str) -> str:
    """
    Run the agent with a user message and return the final text response.
    """
    result = agent.invoke({
        "messages": [{"role": "user", "content": user_message}]
    })

    final_message = result["messages"][-1]
    content = final_message.content

    # Gemini 2.5 returns a list of content blocks instead of a plain string
    if isinstance(content, list):
        text_parts = [
            block["text"] for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n".join(text_parts)

    return content

