import re
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
import torch
from transformers import AutoTokenizer, AutoModel
from pathlib import Path


def preprocess(text):
    text = text.lower()
    text = re.sub(r'\$([a-z]+)', r'\1', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def vectorize(corpus, method):
    preprocessed = [preprocess(text) for text in corpus]
    tokenized = [text.split() for text in preprocessed]

    if method == 'tf-idf':
        vectorizer = TfidfVectorizer(
            ngram_range=(1,2),
            max_df=0.9,
            min_df=5,
            max_features=50000,
            sublinear_tf=True,
            norm='l2'
        )
        X = vectorizer.fit_transform(preprocessed)
        return X, vectorizer

    elif method == 'tf-idf-svd':
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=50000)),
            ('svd', TruncatedSVD(n_components=300, random_state=42))
        ])
        X = pipeline.fit_transform(preprocessed)
        return X, pipeline

    elif method == 'word2vec':
        model = Word2Vec(
                vector_size=300,
                window=5,
                min_count=5,
                sg=1,
                negative=10,
                sample=1e-4,
                epochs=10
            )

        def doc_vector(tokens):
            vectors = [model.wv[word] for word in tokens if word in model.wv]
            if len(vectors) == 0:
                return np.zeros(model.vector_size)
            return np.mean(vectors, axis=0)

        X = np.array([doc_vector(tokens) for tokens in tokenized])
        return X, model

    elif method == 'word2vec-tfidf':
        # train word2vec
        w2v_model = Word2Vec(
            sentences=tokenized,
            vector_size=300,
            window=5,
            min_count=5,
            workers=4,
            sg=1
        )

        # build tf-idf for weighting
        tfidf = TfidfVectorizer(min_df=5)
        tfidf_matrix = tfidf.fit_transform(preprocessed)
        tfidf_vocab = tfidf.vocabulary_

        def doc_vector(tokens, row):
            weights = []
            vectors = []

            for word in tokens:
                if word in w2v_model.wv and word in tfidf_vocab:
                    word_index = tfidf_vocab[word]
                    weight = row[0, word_index]
                    weights.append(weight)
                    vectors.append(w2v_model.wv[word])

            if len(vectors) == 0:
                return np.zeros(w2v_model.vector_size)

            weights = np.array(weights)
            vectors = np.array(vectors)

            return np.average(vectors, axis=0, weights=weights)

        X = np.array([
            doc_vector(tokenized[i], tfidf_matrix[i])
            for i in range(len(tokenized))
        ])

        return X, w2v_model
    
    elif method == 'finbert':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = Path(__file__).parent / "finbert-pretrain"
        model_path_str = str(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"FinBERT model directory not found at {model_path_str}")

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path_str, local_files_only=True, use_fast=False)
        except Exception:
            from transformers import BertTokenizer
            tokenizer = BertTokenizer.from_pretrained(model_path_str, local_files_only=True)

        model = AutoModel.from_pretrained(model_path_str, local_files_only=True)
        model.to(device)
        model.eval()

        def batch_embeddings(texts, batch_size=16):
            all_embeddings = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]

                inputs = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )

                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs)

                # CLS token embedding
                cls_embeddings = outputs.last_hidden_state[:, 0, :]

                all_embeddings.append(cls_embeddings.cpu().numpy())

            return np.vstack(all_embeddings)

        X = batch_embeddings(preprocessed, batch_size=16)
        return X, model

    else:
        raise ValueError(f"Unknown method: {method}")
    
    




df = pd.read_csv("news_prices.csv")
corpus = df["maintext"].dropna().astype(str).tolist()[:1000]
print(vectorize(corpus, 'finbert'))

