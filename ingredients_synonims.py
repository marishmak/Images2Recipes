"""# Loading RusVectōrēs model"""

import zipfile
import wget
import gensim

def process_array(array):
    main_words = []

    for item in array:
      if len(item) > 0:
        doc = nlp(item)
        main_noun = None
        if len(doc) > 1:
            for token in doc:
                if token.pos_ == "NOUN":
                    main_noun = token.lemma_
                    break
        else:
            main_noun = doc[0].lemma_

        main_words.append(main_noun)

    main_words = {word for word in main_words if word is not None}

    main_words_set = set(main_words)

    return main_words_set

def get_unique_keys(filename):
    unique_keys = set()

    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['ingredients']:
                ingredients_dict = eval(row['ingredients'])
                unique_keys.update(ingredients_dict.keys())

    return unique_keys

def synonim_search():
    model_url = 'http://vectors.nlpl.eu/repository/11/180.zip'
    model_file = wget.download(model_url)
    
    with zipfile.ZipFile(model_file, 'r') as archive:
        stream = archive.extract('model.bin')
        model = gensim.models.KeyedVectors.load_word2vec_format(stream, binary=True, unicode_errors='replace')
    
    """## Extracting all unique ingredients from a recipe dataset"""
    
    import csv
    
    
    ingredients = get_unique_keys('/content/povarenok_recipes_2021_06_16.csv')
    
    """## Preproccessing predicted ingredients and ingredients from dataset"""
    
    # !python -m spacy download ru_core_news_md
    
    import spacy
    
    nlp = spacy.load("ru_core_news_md")
    
    
    
    ingredients_main_set = process_array(ingredients)
    predictions_main_set = process_array(predictions_rus)
    
    ingredients_noun = [ingredient + '_NOUN' for ingredient in ingredients_main_set]
    predictions_noun = [prediction + '_NOUN' for prediction in predictions_main_set]
    
    print("Ingredients Main Set:", ingredients_noun)
    print("Predictions Main Set:", predictions_noun)