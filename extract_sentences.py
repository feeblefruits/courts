
# coding: utf-8

import pandas as pd
import numpy as np


df = pd.read_excel('criminal_df_v1.xlsx')

# to replace '\\'96'
# to add bail, compensation amounts



from textblob import TextBlob
import nltk
lemma = nltk.wordnet.WordNetLemmatizer()

def lemma_words(text):
    
    # returns clean text with infinitive verbs

    blob = TextBlob(text)
    blob_words = list(blob.words)
    
    blob_lst = []

    for word in blob_words:
        
        new_word = lemma.lemmatize(word, 'v')

        blob_lst.append(new_word)
        
    return ' '.join(blob_lst)


def get_sentence_years(blob):

    sentence_lst = []

    # extract ngrams that include keywords like years AND sentence, imprisonment AND NOT old and age

    for lst in blob.ngrams(3):
        if 'years' in lst and ('old' not in lst and 'age' not in lst):
            sentence_lst.append(list(lst))

    concatted = []

    for item in sentence_lst:     
        concatted.append(' '.join(item))

    concatted = ' '.join(concatted)
    sentence_phrase = list(dict.fromkeys(concatted.split()))

    # append ints that are mentioned in sentence_phrase 

    amount = []

    for word in sentence_phrase:
        try:
            amount.append(int(word))
        except ValueError:
            pass
    return amount


def get_verdict(text):
    
    text = text.replace('life sentence', '25 years sentence')
    text = text.replace('sentenced to life', '25 years sentence')
    text = text.replace('life imprisonment', '25 years sentence')
    text = text.replace('life in prison', '25 years sentence')
    text = text.replace('sentence to life', '25 years sentence')
    
    text = lemma_words(text)
    
    blob = TextBlob(text)
    
    blob_lst = blob.ngrams(4)
    
    try:
        
        amount = get_sentence_years(blob)
        
        for item in blob_lst:

            if 'remit' in item:
                verdict = 'Remitted'

            elif 'inadmissible' in item or ('law' in item and 'position' in item and 'restored' in item):
                verdict = 'Inadmissible'

            elif 'set' in item and 'aside' in item:
                verdict = 'Set Aside'

            elif 'uphold' in item and 'grant' in item:
                verdict = 'Appeal Granted'

            elif 'uphold' in item and 'appeal' in item:
                verdict = 'Appeal Upheld'

            elif 'appeal' in item and 'dismiss' in item:
                verdict = 'Appeal Dismissed'

            elif 'award' in item or 'compensate' in item or 'compensation' in item:
                verdict = 'Compensation'

            elif 'suspend' in item and ('court' in item or 'sentence' in item):
                verdict = 'Suspended'
                
            elif 'sentence' in item or 'imprisonment' in item:
                verdict = 'Sentenced'
                
        return verdict, amount
    
    except UnboundLocalError:
        return np.nan, amount



get_verdict(df['Summary'][50])


for words in text_test.split(' '):
    try:
        result =  re.search("^r+", words) and re.search("\d+", words)
        print(result.group())
    except AttributeError:
        pass

# test and apply

df['Verdict'] = df['Summary'].apply(get_verdict)

