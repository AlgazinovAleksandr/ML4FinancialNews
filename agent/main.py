"""
Usage EXAMPLE (обожаю примеры с ними так хорошо а без них так тяжело):
    python -m agent.main --ticker AAPL --news "Apple reported record quarterly revenue..."
    python    agent/main.py --ticker AAPL --news "..."
"""

import argparse
import sys

from .agent import run_agent


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stock news recommendation agent",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--ticker",
        required=True,
        help="Stock ticker symbol, e.g. AAPL",
    )
    parser.add_argument(
        "--news",
        required=True,
        help="News article text to analyse",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    try:
        result = run_agent(ticker=args.ticker, news_text=args.news)
        print(result.model_dump_json(indent=2))
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
