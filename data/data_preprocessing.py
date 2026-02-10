from pathlib import Path
import pandas as pd
import json as _json


def article_to_rows(article: dict) -> list[dict]:
    rows = []

    # ---- extract high-confidence entity words (article-level) ----
    entities = [
        ent["word"]
        for ent in article.get("named_entities", [])
        if ent.get("score", 0) > 0.8 and "word" in ent
    ]
    
    mentioned = article.get("mentioned_companies", [])
    related = article.get("related_companies", [])
    industries = article.get("industries", [])
    
    

    # serialize for CSV
    entities_json = _json.dumps(entities, ensure_ascii=False)
    mentioned_json = _json.dumps(mentioned, ensure_ascii=False)
    related_json = _json.dumps(related, ensure_ascii=False)
    industries_json = _json.dumps(industries, ensure_ascii=False)

    sentiment = article.get("sentiment", {})
    emotion = article.get("emotion", {})

    base_fields = {
        "date_publish": article.get("date_publish"),
        "date_download": article.get("date_download"),
        "source_domain": article.get("source_domain"),
        "news_outlet": article.get("news_outlet"),
        "title": article.get("title"),
        "description": article.get("description"),
        "maintext": article.get("maintext"),
        "url": article.get("url"),
        "language": article.get("language"),

        # sentiment (article-level)
        "sentiment_negative": sentiment.get("negative"),
        "sentiment_neutral": sentiment.get("neutral"),
        "sentiment_positive": sentiment.get("positive"),

        # emotion (article-level)
        "emotion_anger": emotion.get("anger"),
        "emotion_fear": emotion.get("fear"),
        "emotion_joy": emotion.get("joy"),
        "emotion_sadness": emotion.get("sadness"),
        "emotion_disgust": emotion.get("disgust"),
        "emotion_surprise": emotion.get("surprise"),
        "emotion_neutral": emotion.get("neutral"),

        # named entities (article-level)
        "named_entities": entities_json,
        "mentioned_companies": mentioned_json,
        "related_companies": related_json,
        "industries": industries_json
    }

    for ticker in article.get("mentioned_companies", []):
        rows.append({
            **base_fields,
            "ticker": ticker,
            "prev_day_price": article.get(f"prev_day_price_{ticker}"),
            "curr_day_price": article.get(f"curr_day_price_{ticker}"),
            "next_day_price": article.get(f"next_day_price_{ticker}"),
        })

    return rows



def jsons_to_csv(output_csv="news_prices.csv"):
    all_rows = []

    for path in Path(".").glob("*.json"):
        with open(path, "r", encoding="utf-8") as f:
            print(f.name)
            data = _json.load(f)

        if isinstance(data, list):
            for article in data:
                all_rows.extend(article_to_rows(article))
        else:
            all_rows.extend(article_to_rows(data))

    df = pd.DataFrame(all_rows)
    df.to_csv(output_csv, index=False)

    print(f"Saved {len(df)} rows to {output_csv}")


if __name__ == "__main__":
    jsons_to_csv()
