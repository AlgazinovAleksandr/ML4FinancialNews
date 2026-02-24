from typing import TypedDict
from langgraph.graph import StateGraph, END
from app.agent import build_chain, PydanticOutputParser
from .schemas import AgentInput, AgentOutput
from .agent import build_chain


class AgentState(TypedDict):
    input: AgentInput
    output: AgentOutput


def build_graph():
    chain = build_chain()
    parser = PydanticOutputParser(pydantic_object=AgentOutput)


    def run_agent(state: AgentState):
        result = chain.invoke(
            {
                "news": state["input"].news,
                "ticker": state["input"].ticker,
                "current_price": state["input"].current_price,
                "model_forecast_pct": state["input"].model_forecast_pct,
                "metadata": state["input"].metadata,
                "format_instructions": parser.get_format_instructions(),
            }
        )
        return {"output": result}

    graph = StateGraph(AgentState)
    graph.add_node("agent_node", run_agent)

    graph.set_entry_point("agent_node")
    graph.add_edge("agent_node", END)

    return graph.compile()