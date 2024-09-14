"""## Searching for synonyms in dataset ingredients"""

def calculate_similarity(word, ingredients_noun):
    similarities = []

    for ingredient in ingredients_noun:
      if ingredient in model and ingredient != word:
        similarity = model.similarity(word, ingredient)
        similarities.append((ingredient, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)
    top_similarities = similarities[:3]

    return top_similarities

def get_rubert_embedding(word_list, max_length=128):
    inputs = tokenizer(' '.join(word_list), return_tensors='pt', padding=True, truncation=True, max_length=max_length)
    outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().detach().numpy()
    return cls_embedding

def perform_search(collection_name, query_vector, limit=5):
    response = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=limit,
        with_payload=True
    )
    return response


def searching_for_ingredients():
    synonyms_predictions = []

    for word in predictions_noun:
      if word in model:
        top_similarities = calculate_similarity(word, ingredients_noun)
        print(f"The most similar words to '{word}' are:")
        if len(top_similarities) != 0:

          synonyms_predictions.append(top_similarities[0][0][:-5] if top_similarities[0][1] > 0.6 else word[:-5])

          for ingredient, similarity in top_similarities:
              print(f"- {ingredient} (similarity: {similarity:.2f})")
          print()
        else:
          print(f"No similarities for '{word}'")
          print()
      else:
        synonyms_predictions.append(word[:-5])
        print(f"No such word as '{word}'")
        print()

    """# Initial ingredients vs synonims"""

    print("Initial list: ", [prediction[:-5] for prediction in predictions_noun])
    print("Synonyms list: ", synonyms_predictions)

    """# Setting up database access"""

    # !pip install qdrant-client

    from qdrant_client import QdrantClient
    from qdrant_client.http import models

    api_key = 'id700l8bqyjwxINKMNeM4PjxQRHAj8O-pJ7CHIyYnViEzUsO9RPuNw'
    url = 'https://3368514d-ddd6-417d-8f79-d027e32cea9d.us-east4-0.gcp.cloud.qdrant.io:6333'
    client = QdrantClient(api_key=api_key, url=url)

    """## Turning word lists into embeddings"""

    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
    model = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased")

    # !pip install torch

    import torch
    from sklearn.decomposition import PCA
    import numpy as np
    from sklearn.decomposition import TruncatedSVD
    import joblib

    tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
    model = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased")


    raw_prediction = predictions_rus
    raw_synonyms = synonyms_predictions


    prediction_embedding = get_rubert_embedding(raw_prediction)
    synonyms_embedding = get_rubert_embedding(raw_synonyms)
    prediction_embedding = prediction_embedding.reshape(1, -1)
    synonyms_embedding = synonyms_embedding.reshape(1, -1)
    pca = joblib.load("pca_model.pkl")

    prediction_embedding_70d = pca.transform(prediction_embedding)
    synonyms_embedding_70d = pca.transform(synonyms_embedding)

    prediction_embedding_70d = prediction_embedding_70d.reshape(70)
    synonyms_embedding_70d = synonyms_embedding_70d.reshape(70)

    """# List of recipes"""

    import time
    from qdrant_client import QdrantClient, models

    collection_name_70 = "recipes_70"
    print("INITIAL", raw_prediction)

    search_results_70 = perform_search(collection_name_70, prediction_embedding_70d )
    print("\nSearch results for INITIAL :")
    for result in search_results_70:
        print(result.payload)

    print("\n\nSYNONYMS", raw_synonyms)
    search_results_70 = perform_search(collection_name_70, synonyms_embedding_70d)
    print("\nSearch results for SYNONYMS :")
    for result in search_results_70:
        print(result.payload)