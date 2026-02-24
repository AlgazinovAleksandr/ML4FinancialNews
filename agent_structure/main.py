from app.schemas import AgentInput
from app.graph import build_graph
import argparse
from ast import literal_eval

def parse_args():
    parser = argparse.ArgumentParser(description="Run stock recommendation agent.")
    parser.add_argument("--news", type=str, required=True, help="News text")
    parser.add_argument("--ticker", type=str, required=True, help="Stock ticker")
    parser.add_argument("--current_price", type=float, required=True, help="Current price")
    parser.add_argument("--model_forecast_pct", type=float, required=True, help="Model forecast % change")
    parser.add_argument("--metadata", type=str, default="{}", help="Optional metadata as dict string")
    return parser.parse_args()

if __name__ == "__main__":
    graph = build_graph()
    args = parse_args()
    metadata_dict = literal_eval(args.metadata)

    """
    sample_input = AgentInput(
        news="Company reported strong earnings beating expectations.",
        ticker="AAPL",
        current_price=180.0,
        model_forecast_pct=5.4,
        metadata={"sector": "technology"},
    )
    """
    
    sample_input = AgentInput(
        news=args.news,
        ticker=args.ticker,
        current_price=args.current_price,
        model_forecast_pct=args.model_forecast_pct,
        metadata=metadata_dict,
    )

    result = graph.invoke({"input": sample_input})

    print(result["output"].model_dump_json(indent=2))