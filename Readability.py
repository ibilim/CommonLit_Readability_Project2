
import nltk
import re
import pandas as pd
import numpy as np
from collections import Counter
from nltk.tokenize.api import TokenizerI
from nltk.corpus import words
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV 
from nltk import WordNetLemmatizer
from nltk import pos_tag
class LegalitySyllableTokenizer(TokenizerI):
    """
    Syllabifies words based on the Legality Principle and Onset Maximization.

        >>> from nltk.tokenize import LegalitySyllableTokenizer
        >>> from nltk import word_tokenize
        >>> from nltk.corpus import words
        >>> text = "This is a wonderful sentence."
        >>> text_words = word_tokenize(text)
        >>> LP = LegalitySyllableTokenizer(words.words())
        >>> [LP.tokenize(word) for word in text_words]
        [['This'], ['is'], ['a'], ['won', 'der', 'ful'], ['sen', 'ten', 'ce'], ['.']]
    """

    def __init__(self, tokenized_source_text, vowels="aeiouy", legal_frequency_threshold=.001):
        """
        :param tokenized_source_text: List of valid tokens in the language
        :type tokenized_source_text: list(str)
        :param vowels: Valid vowels in language or IPA represenation
        :type vowels: str
        :param legal_frequency_threshold: Lowest frequency of all onsets to be considered a legal onset
        :type legal_frequency_threshold: float
        """
        self.legal_frequency_threshold = legal_frequency_threshold
        self.vowels = vowels
        self.legal_onsets = self.find_legal_onsets(tokenized_source_text)
    def find_legal_onsets(self, words):
       
        onsets = [self.onset(word) for word in words]
        legal_onsets = [k for k,v 
                        in Counter(onsets).items()
                        if (v / len(onsets)) > self.legal_frequency_threshold]
        return set(legal_onsets)
    
    def onset(self, word):
        onset = ""
        for c in word.lower():
            if c in self.vowels:
                return onset
            else:
                onset += c
        return onset
    def tokenize(self, token):
        syllables = []
        syllable, current_onset = "", ""
        vowel, onset = False, False
        for char in token[::-1]:
            char_lower = char.lower()
            if not vowel:
                syllable += char
                vowel = bool(char_lower in self.vowels)
            else:
                if char_lower + current_onset[::-1] in self.legal_onsets:
                    syllable += char
                    current_onset += char_lower
                    onset = True
                elif char_lower in self.vowels and not onset:
                    syllable += char
                    current_onset += char_lower
                else:
                    syllables.append(syllable)
                    syllable = char
                    current_onset = ""
                    vowel = bool(char_lower in self.vowels)
        syllables.append(syllable)
        syllables_ordered = [syllable[::-1] for syllable in syllables][::-1]
        return syllables_ordered

def get_trained_model():
    trained_data=pd.read_csv('trained_model_outlier_free.csv')
    #neuro_model
    X=trained_data[['lexical_diversity','text_lenght','num_words_per_sent','ave_syllabi_per_word','gsl_calculation_ave']] 
    y=trained_data['target']
    X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
    linear_reg=LinearRegression().fit(X_train,y_train) 
    return linear_reg                                           
    

def process_test_data(test_file):
    #read test file
    test_data=pd.read_csv(test_file)
    #refine test data
    test_data['excerpt_new']=['. '.join([re.sub(r'\W+',' ',i) for i in nltk.sent_tokenize(text)]) 
                              for text in test_data['excerpt']]
    
    # Dimension 1  text lenght
    test_data['text_lenght']=test_data['excerpt_new'].str.len()
    
    #Dimention 2 Average Word lenght
    test_data['word_lenght_ave']=[sum([len(word) for word in nltk.word_tokenize(i)])/
                               len([word for word in nltk.word_tokenize(i)]) for i in test_data['excerpt_new']]
    
    #Dimention 3 lexical diversity of a tex: number of tokens/ number of unique tokens
    test_data['words_unique']=[len(set([ word for word in nltk.word_tokenize(i) ])) for i in test_data['excerpt_new']]
    test_data['words_number']=[len([ word for word in nltk.word_tokenize(i) ]) for i in test_data['excerpt_new']]
    test_data['lexical_diversity']=test_data['words_number']/test_data['words_unique']
    
    # Dimention 4 Average number of words per sentence
    test_data['words_count']=[len( [word for word in nltk.word_tokenize(i) ]) for i in test_data['excerpt_new']]
    test_data['sent_count']=[len(nltk.sent_tokenize(i)) for i in test_data['excerpt_new']]
    test_data['num_words_per_sent']=test_data['words_count']/test_data['sent_count']
    
    #Dimention 5 average syllabus per word
    Leg_Syl_tok=LegalitySyllableTokenizer(words.words())
    test_data['ave_syllabi_per_word']=[sum([len(Leg_Syl_tok.tokenize(word)) for word in nltk.word_tokenize(text)])/
                                        len([i for i in nltk.word_tokenize(text) ]) for text in test_data['excerpt_new']]
    
    # Dimention 6 GSL calculation but first Lemmatization of text
    #lemmatize test data
    lemmatizer=nltk.WordNetLemmatizer()
    test_data['lemmatized']=[[lemmatizer.lemmatize(i,j[0].lower()) if j[0].lower() in ['a','n','v'] 
                              else lemmatizer.lemmatize(i).lower() 
                               for i,j in pos_tag(nltk.word_tokenize(text))] for text in test_data['excerpt_new']]
    test_data['lemmatized_set']=[list(set(i)) for i in test_data['lemmatized']]
    
    #General Service List: Read 
    words_corpus=pd.read_csv('NGSL_Spoken_1.01.csv')
    #create GSL corpus dictionary
    wc_dict={words_corpus['Lemma'][i]:words_corpus['Cumulative'][i] for i in range(0,len(words_corpus['Lemma']))}
    
    # Dimention 6 GSL score averages
    test_data['gsl_calculation']=[[wc_dict.get(word,0) for word in wc_dict.keys()  if word in i] 
                                  for i in test_data['lemmatized_set']]
    test_data['gsl_calculation_ave']=[[sum(i)/len(i)][0] for i in test_data['gsl_calculation']]
    
    return test_data[['lexical_diversity','text_lenght','num_words_per_sent','ave_syllabi_per_word','gsl_calculation_ave']] 
    
def get_predictions(test_file):
    
    predict_values=pd.DataFrame([i for i in pd.read_csv(test_file)['id']]).rename(columns={0:'id'})
    predict_values['target']=[round (i,1) for i in get_trained_model().predict(process_test_data(test_file))]
    
    return predict_values.to_csv('submission.csv',index=False)
