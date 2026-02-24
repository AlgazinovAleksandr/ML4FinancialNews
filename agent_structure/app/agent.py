from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from .schemas import AgentOutput
from .config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, MODEL_NAME


def build_llm():
    return ChatOpenAI(
        model=MODEL_NAME,
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
        temperature=0.2,
    )


def build_chain():
    parser = PydanticOutputParser(pydantic_object=AgentOutput)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a quantitative equity analyst.
                Use the news context, model forecast, and price information
                to generate a structured investment recommendation.

                Be conservative and risk-aware.
                Return ONLY valid JSON following the schema.
                """,
            ),
            (
                "human",
                """
    News:
    {news}

    Ticker: {ticker}
    Current price: {current_price}
    Model forecast (change percentage): {model_forecast_pct}

    Metadata:
    {metadata}

    {format_instructions}
                """,
            ),
        ]
    )

    llm = build_llm()

    chain = prompt | llm | parser
    return chain