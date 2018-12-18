# coding: utf-8
from __future__ import division

import nltk
from nltk.corpus import stopwords
from nltk.util import skipgrams
from nltk.stem import PorterStemmer
from textblob import TextBlob, Word
import pandas as pd
import numpy as np

DIGITS = {
    'one': 1,
    'two': 2,
    'three': 3,
    'four': 4,
    'five': 5,
    'six': 6,
    'seven': 7,
    'eight': 8,
    'nine': 9,
    'ten': 10,
    'eleven': 11,
    'twelve': 12,
    'thirteen': 13,
    'fourteen': 14,
    'fifteen': 15,
    'sixteen': 16,
    'seventeen': 17,
    'eighteen': 18,
    'nineteen': 19,
}

CRIMES = [
    'Murder', 'Manslaughter', 'Mayhem', 'assault', 'Battery', 'Kidnapping',
    'abduction', 'Rape', 'Buggery', 'Arson', 'Larceny', 'Robbery', 'Burglary',
    'Trespass', 'Extortion', 'Cheating', 'Forgery', 'High treason',
    'Petty treason', 'treason', 'Sedition', 'Espionage', 'Riot', 'Mobbing',
    'Piracy', 'Rout', 'Affray', 'Blasphemy', 'Incitement', 'Champerty',
    'Embracery', 'Eavesdropping', 'Barratry', 'Conspiracy', 'Accessory',
    'Housebreaking', 'Theft', 'Contravening', 'attempted'
]

LEMMATIZED_CRIMES = [Word(i.lower()).lemmatize() for i in CRIMES]

df = pd.read_csv('court_cases_df.csv', encoding='utf-8')


def lower_case(text):
    return text.apply(lambda x: ' '.join(x.lower() for x in x.split()))


def remove_punctuation(text):
    return text.str.replace(r'[^\w\s]', '')


def remove_stopwords(text):
    stop_words = stopwords.words('english')
    return text.apply(
        lambda x: ' '.join(x for x in x.split() if x not in stop_words))


def correct_spelling(text):
    return text.apply(lambda x: str(TextBlob(x).correct()))


def lemmatize(text):
    return text.apply(
        lambda x: ' '.join([Word(word).lemmatize() for word in x.split()]))


def process_column(column_name):
    return remove_stopwords(remove_punctuation(lower_case(df[column_name])))


def convert_word_pos_tag_dict(pos_tags):
    tags_dict = {}
    for i in pos_tags:
        tags_dict[i[1]] = i[0]
    return tags_dict


def get_digit_from_sentence(lst):
    tags = nltk.pos_tag(lst)
    tags_dict = convert_word_pos_tag_dict(tags)
    # nltk pos tagger tags all numbers as CD
    digit = tags_dict.get('CD', 0)

    if isinstance(digit, int) or digit.isdigit():
        return digit
    try:
        return int(DIGITS.get(digit))
    except TypeError as e:
        return 0


def extract_sentence(text):
    for lst in TextBlob(text).ngrams(2):
        if 'months' in lst or 'years' in lst:
            length = ''
            try:
                length = int(DIGITS.get(lst[0], lst[0]))
            except ValueError as e:
                print(e)
                pass
            # print('{} {}'.format(' '.join(lst), length))
            return ' '.join(lst)
        if 'life' in lst:
            print(text)
            for s in skipgrams(text.split(), 3, 2):
                if 'sentenced' == s[0]:
                    pass
                    # print(s)
            print('\n\n')
    return ' '.join(lst)


def analyse_text(processed_text):

    # There could be a lot of words between `sentenced` and verdict such as
    # effective, accused, etc. so widen the ngrams
    for lst in TextBlob(processed_text).ngrams(6):
        if is_sentencing(lst):
            s = extract_sentence(processed_text)
            print(processed_text)

            # print(lst)
            print('\n\n')
            return s, s

        # Check for other types of verdicts

    return '', ''


def is_sentencing(lst):
    return 'sentenced' in lst and ('years' in lst or 'months' in lst
                                   or 'life' in lst)


def extract_number_of_counts(processed_text):
    """
    Assumption: the number always comes before the
    counts i.e [three counts] of murder else it's only
    one count
    """
    counts = []
    for lst in TextBlob(processed_text).ngrams(3):
        if ('counts' in lst or 'count' in lst) and lst[1] == 'counts':
            count = ''
            try:
                count = int(DIGITS.get(lst[0], lst[0]))
            except ValueError as e:
                if lst[0] == 'numerous':
                    count = 'numerous'
                elif lst[0] == 'multiple':
                    count = 'multiple'
                print(e)

            if lst[2] in LEMMATIZED_CRIMES and count:
                counts.append("{} counts {}".format(count, lst[2]))
                print("{} counts {}".format(count, lst[2]))
            elif count:
                counts.append("{} counts {}".format(count, ""))
    return len(counts) > 1, counts


def get_verdict(text):
    # We can only process single defendant
    if not has_multiple_defendants(text):
        sentences = get_sentences(text)
        if len(sentences) > 1:
            if is_concurrent_sentence(text):
                sentence = max(sentences)
            else:
                sentence = sum(sentences)
        elif sentences:
            sentence = sentences[0]
        else:
            # Couldn't extract sentence basically
            sentence = sentences

        if sentence:

            suspended = get_suspended(text)
            if suspended > 0 and suspended < sentence:
                sentence -= suspended

            years_aside = get_set_aside(text)
            if years_aside > 0 and years_aside < sentence:
                sentence -= years_aside
            return [sentence]
    return []


def has_multiple_defendants(processed_text):
    # We can assume if plural is used then we're talking about multiple
    # defendants
    # TODO: What about `accused`?
    # can't use accused because the plural remains accused
    return 'defendants' in processed_text or \
           'appellants' in processed_text


def get_sentences(text):
    sentences = []
    for lst in TextBlob(text).ngrams(4):
        # target ngram ---> "sentence 14 years imprisonment"
        if ('sentenced' in lst or 'sentence' in lst) and \
           'imprisonment' in lst and \
           ('years' in lst or 'months' in lst):

            digit = int(get_digit_from_sentence(lst))

            # because this is also picking out years, filter out numbers greater than 200
            # highly unlikely someone will get a sentence of 200 years

            # if it's a life sentence first?
            if digit and digit < 200:
                if 'months' in lst:
                    sentences.append(digit * 0.0833334)
                else:
                    sentences.append(digit)

    # Check for `rape`, `robbery` and `murder`
    if 'rape' in text:
        sentences.append(15)
    elif 'robbery' in text:
        sentences.append(10)
    elif 'murder' in text:
        sentences.append(25)

    return sentences


def is_concurrent_sentence(processed_text):
    return 'sentenced' in processed_text and \
            ('concurrent' in processed_text or \
             'concurrently' in processed_text or \
             ('along' in processed_text and 'side' in processed_text))


def get_suspended(processed_text):
    # assuming there will always be
    # [('suspended', 'VBN'), ('five', 'CD'), ('years', 'NNS')]
    if 'sentenced' in processed_text:
        for lst in TextBlob(processed_text).ngrams(4):
            tags = convert_word_pos_tag_dict(nltk.pos_tag(lst))
            if 'suspended' in lst and 'CD' in tags:
                time_suspended = ''
                try:
                    time_suspended = tags.get('CD')
                    if isinstance(tags.get('CD'), int):
                        time_suspended = tags.get('CD')
                    else:
                        time_suspended = int(DIGITS.get(tags.get('CD')))
                except ValueError as e:
                    pass
                except TypeError as ne:
                    pass

                if 'months' in lst or 'month' in lst:
                    return time_suspended * 0.0833334

                return time_suspended
    return 0


def get_set_aside(processed_text):
    # [('suspended', 'VBN'), ('five', 'CD'), ('years', 'NNS')]
    if 'sentenced' in processed_text:
        for lst in TextBlob(processed_text).ngrams(5):
            tags = convert_word_pos_tag_dict(nltk.pos_tag(lst))
            if ('set' in lst and 'aside' in lst and 'CD' in tags) \
             and ('year' in lst or 'years' in lst or 'month' in \
              lst or 'months' in lst) and not 'imprisonment' in lst:
                #       years
                time_aside = ''
                try:
                    time_aside = tags.get('CD')
                    if isinstance(tags.get('CD'), int):
                        time_aside = tags.get('CD')
                    else:
                        time_aside = int(DIGITS.get(tags.get('CD')))
                except ValueError as e:
                    pass
                except TypeError as ne:
                    pass

                if 'months' in lst or 'month' in lst:
                    return time_aside * 0.0833334
                return time_aside

    return 0


df['Verdict'] = process_column('Summary').apply(get_verdict)

df.to_csv('processed_court_cases.csv', index=False)