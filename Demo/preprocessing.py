# -*- coding: utf-8 -*-
"""
Updated on Tue Oct 17 

@author: conkennedy

This is a shortened preprocessing script that processes the input headline text in a similar manner to what was done for the data used to train the models. 
This is called from the demo.py script to process the input text before it is passed to the models for inference.
"""

#Import libraries
import pandas as pd

import re

import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

# FUNCTIONS FOR PREPROCESSING

# Outline Specific contractions to be applied to data
def specific_contractions():

    '''
    Get a dictionary of specific contractions and their corresponding replacements.

    Returns:
    general phrases (dict) : A dictionary containing regular expression patterns as keys and their corresponding replacements
    as values.

    '''

    GENERAL_PHRASES = {
        #covid
        r"\bcovid19\b": "covid",
        r"\bcovid 19\b": "covid",
        r"\bcoronavirus\b": "covid",
        r"\bcorona virus\b": "covid",
        r"\b[Uu]\.?[Ss]\.?'?s?\b ": "US ",
        r"\bU\.?K\.?\b" : "United Kingdom",
        r"&" : "and"}

    return GENERAL_PHRASES

def replace_custom_constractions(text):

    '''
    Replace specific contractions in the given text with their corresponding expansions.

    Args:
    text (str) : The input text where contractions will be replaced.

    Returns:
    text (str) : The input text with specific contractions replaced.

    '''

    dict_contractions = specific_contractions()

    for pattern, replacement in dict_contractions.items():
        text = re.sub(pattern, replacement, text)

    return text

#Import stopwords
def custom_stopwords():

    '''
    Get a set of custom stopwords by combining NLTK's English stopwords with additional stopwords.

    Returns:
        set (set): A set containing custom stopwords for text processing.
    '''

    from nltk.corpus import stopwords
    stopwords = set(stopwords.words('english'))


    extra = set([])# Placeholder to include regular expressions and custom stopwords later

    stopwords = stopwords.union(extra)
    return stopwords


def data_frame_pre_process(df_input, field, is_lemming = False, is_stemming = False, str_rejoin = True):

    '''
    Preprocess the text data in a DataFrame column.

    Args:
        df_input (pandas.DataFrame): The DataFrame containing the text data.
        field (str): The name of the column in the DataFrame containing the text data.
        is_lemming (bool): The flag to control if the text should be lemmatized - default is False.
        is_stemming (bool): The flag to control if the text should be stemmed - default is False.
        str_rejoin (bool): The flag to control if the text should be rejoined from a list at the end of the preprocessing - default is False.

    Returns:
        pandas.DataFrame: A DataFrame with the preprocessed text data in a new column named 'CLEAN_TEXT'.
    '''

    # Remove NA values from the dataset if present:
    df = df_input.dropna().copy()

    # Remove the basic contractions with the contractions package:
    df['CLEAN_TEXT'] = df[field].apply(lambda x: replace_custom_constractions(x))
    # df['CLEAN_TEXT'] = df['CLEAN_TEXT'].apply(lambda x: cns.fix(x)) #NOT SURE WHAT THE CNS PACKAGE IS HERE?ß

    # Remove 1st, 2nd, 3rd etc.:

    df['CLEAN_TEXT'] = df['CLEAN_TEXT'].apply(lambda x : re.sub(r"\b\d+(st|nd|rd|th)\b", '', x))

    # Remove the Amounts of money e.g. 100k, 200M etc.:
    df['CLEAN_TEXT'] = df['CLEAN_TEXT'].apply(lambda x : re.sub(r"\b\d+(\.\d+)?[kMBGTPEZY]\b", '', x))


    # Remove " 's " pattern:
    df['CLEAN_TEXT'] = df['CLEAN_TEXT'].apply(lambda x : re.sub(r"\b\w+'s\b", '', x))

    # remove currency numbers:
    df['CLEAN_TEXT'] = df['CLEAN_TEXT'].apply(lambda x : re.sub(r"[$€¥£]\d+[^ ]*", '', x))


    # Remove punctuation where the punctuation is internal - add space instead to maintain word breakpoint:
    pattern = r"(?<=\w)[!\"#$%&'()*+,-./:;<=>?@[\\\]^_`{|}~](?=\w)"
    df['CLEAN_TEXT'] = df['CLEAN_TEXT'].apply(lambda x: re.sub(pattern, ' ', x))

    # Remove punctuation internal to words and replace with space
    df['CLEAN_TEXT'] = df['CLEAN_TEXT'].apply(lambda x : re.sub(r"[^a-zA-Z0-9\s]+", '', x))

    # Remove Numbers:
    df['CLEAN_TEXT'] = df['CLEAN_TEXT'].apply(lambda x : re.sub(r'\d+', '', x))


    # Remove multiple spaces where they occur:
    df['CLEAN_TEXT'] = df['CLEAN_TEXT'].apply(lambda x : re.sub(r'^\s+', '', x))
    df['CLEAN_TEXT'] = df['CLEAN_TEXT'].apply(lambda x : re.sub(r'\s+', ' ', x))

    # Make the Text Lower
    df['CLEAN_TEXT'] = df['CLEAN_TEXT'].apply(lambda x : x.lower())

    # Tokenize the text:
    df['CLEAN_TEXT'] = df['CLEAN_TEXT'].apply(lambda x : nltk.word_tokenize(x))

    # Remove set of default stopwords with stopwords library:
    stop = custom_stopwords()
    df['CLEAN_TEXT'] = df['CLEAN_TEXT'].apply(lambda x : [word for word in x if word not in stop])

    # Lemmatisation:
    if is_lemming:

        lem = WordNetLemmatizer()
        df['CLEAN_TEXT'] = df['CLEAN_TEXT'].apply(lambda x : [lem.lemmatize(w) for w in x])

    # Stemming
    if is_stemming:

        ps = PorterStemmer()
        df['CLEAN_TEXT'] = df['CLEAN_TEXT'].apply(lambda x : [ps.stem(w) for w in x])

    # If you want the tokens to be made back into a string:
    if str_rejoin:
        df['CLEAN_TEXT'] = df['CLEAN_TEXT'].apply(lambda x : " ".join(x))

    return df

# POSSIBLE FUTURE FUNCTIONS

# def feature_engineer(df,target, unwanted = False):

#     """
#     Perform feature engineering on the given DataFrame for increased accuracy in classification.

#     Args:
#         df (pandas.DataFrame): The DataFrame containing pre-processed text data.
#         target (str): The String name of the target categorical Label.
#         unwanted (list): The list of unwanted columns Strings to drop

#     Returns:
#         X (pandas.DataFrame): The DataFrame with added features based on feature engineering.
#         Y (np.array): Label Encoding of Categorical Target

#     Note:
#         This function adds various features to the DataFrame including word count, character count,
#         diversity score, punctuation count, polarity, subjectivity, counts of specific parts of speech (POS) tags,
#         and more.

#     """
#     y = df[target]


#     # Feature Engineering for increased accuracy in classification:
#     # Now we have processed and pre-processed text in our dataframe. Let's start making features from the above data.
#     if unwanted:
#         unwanted.append(target)
#         X = df.drop(unwanted, axis = 1)

#     # Feature 1 - Length of the input OR count of the words in the statement (Vocab size).
#     X['WORD_COUNT'] = X['TITLE'].apply(lambda x: len(str(x).split()))  # Feature 1

#     # Feature 2 - Count of characters in a statement
#     X['CHARACTER_COUNT'] = X['TITLE'].apply(lambda x: len(str(x)))  # Feature 2

#     # Feature 3 - Diversity_score i.e. Average length of words used in statement
#     X['AVERAGE_LENGTH'] = X['CHARACTER_COUNT'] / X['WORD_COUNT']  # Feature 3

#     # Feature 4: Count of punctuations in the input.
#     X['PUNCTUATION_COUNT'] = X['TITLE'].apply(lambda x: len([w for w in str(x) if w in string.punctuation]))  # Feature 4

#     # Change df_small to df to create these features on the complete dataframe
#     X['polarity'] = X['TITLE'].apply(get_polarity)  # Feature 5: Polarity
#     X['subjectivity'] = X['TITLE'].apply(get_subjectivity)  # Feature 6: Subjectivity

#     # Tokenize all text without stopwords
#     all_text_without_sw = ''
#     for i in df.itertuples():
#         all_text_without_sw = all_text_without_sw + str(i.TITLE)

#     tokenized_all_text = word_tokenize(all_text_without_sw)  # tokenize the text

#     # Adding POS Tags to tokenized words
#     list_of_tagged_words = nltk.pos_tag(tokenized_all_text)
#     set_pos = set(list_of_tagged_words)  # set of POS tags & words

#     # Counting specific POS tags
#     nouns = ['NN', 'NNS', 'NNP', 'NNPS']  # POS tags of nouns
#     list_of_words = set(map(lambda tuple_2: tuple_2[0], filter(lambda tuple_2: tuple_2[1] in nouns, set_pos)))
#     X['NOUN'] = X['TITLE'].apply(lambda x: len([w for w in str(x).lower().split() if w in list_of_words]))  # Feature 7

#     # Counting pronouns
#     pronouns = ['PRP', 'PRP$', 'WP', 'WP$']  # POS tags of pronouns
#     list_of_words = set(map(lambda tuple_2: tuple_2[0], filter(lambda tuple_2: tuple_2[1] in pronouns, set_pos)))
#     df['PRONOUN_COUNT'] = df['TITLE'].apply(lambda x: len([w for w in str(x).lower().split() if w in list_of_words]))  # Feature 8

#     # Counting verbs
#     verbs = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']  # POS tags of verbs
#     list_of_words = set(map(lambda tuple_2: tuple_2[0], filter(lambda tuple_2: tuple_2[1] in verbs, set_pos)))
#     X['VERBS_COUNT'] = X['TITLE'].apply(lambda x: len([w for w in str(x).lower().split() if w in list_of_words]))  # Feature 9

#     # Counting adverbs
#     adverbs = ['RB', 'RBR', 'RBS', 'WRB']  # POS tags of adverbs
#     list_of_words = set(map(lambda tuple_2: tuple_2[0], filter(lambda tuple_2: tuple_2[1] in adverbs, set_pos)))
#     X['ADVERBS_COUNT'] = X['TITLE'].apply(lambda x: len([w for w in str(x).lower().split() if w in list_of_words]))  # Feature 10

#     # Counting adjectives
#     adjectives = ['JJ', 'JJR', 'JJS']  # POS tags of adjectives
#     list_of_words = set(map(lambda tuple_2: tuple_2[0], filter(lambda tuple_2: tuple_2[1] in adjectives, set_pos)))
#     X['ADJECTIVE_COUNT'] = X['TITLE'].apply(lambda x: len([w for w in str(x).lower().split() if w in list_of_words]))  # Feature 11

#     encoder = LabelEncoder()
#     y = encoder.fit_transform(y)

#     return X,y

#############################################################

# Download nltk (Natural Language ToolKit) - Action may be required
# nltk.download()

# Preprocessing Flow
class Headline:
    def __init__(self, headline):
        self.headline = headline

    def process(self):
        df = pd.DataFrame()
        # df['TITLE'] = None
        # df['CLEAN_TEXT'] = None
        df.loc[0, 'TITLE'] = str(self.headline)
        df['TITLE'][0] = str(self.headline)
        field = 'TITLE'
        df = data_frame_pre_process(df, field, is_lemming = False, is_stemming = False, str_rejoin = True)
        return df['CLEAN_TEXT'][0]