# -*- coding: utf-8 -*-
import re
import nltk
import pymorphy2

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


class TextPreprocessing():
    
    def __init__(self, setting: dict):
        
        nltk.download('punkt') 
        nltk.download('stopwords')
        
        self.stopwords = setting['stopwords']
        self.morph = pymorphy2.MorphAnalyzer(lang='ru')
        
    
    def tokenize_text(self, text: str) -> list:
        text = text.lower().replace("ё", "е")
        text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '', text)
        text = re.sub('@[^\s]+', '', text)
        text = re.sub('[^a-zA-Zа-яА-Я1-9]+', ' ', text)
        text = re.sub(' +', ' ', text)
        
        tokenize = [token for token in nltk.word_tokenize(text)]
        remove_stopW = [clear_txt for clear_txt in tokenize if clear_txt not in self.stopwords]
        morph_text = [self.morph.parse(m_text)[0].normal_form for m_text in remove_stopW]
        
        return morph_text
    
    
    def cosine_vector(self, vectors): 
        vec1 = vectors[0].reshape(1, -1)
        vec2 = vectors[1].reshape(1, -1)

        return round(cosine_similarity(vec1, vec2)[0][0] * 100, 2)
    
    
    def vectorize_text(self, array_text: list) -> list:
        ohe = CountVectorizer(tokenizer=self.tokenize_text).fit_transform(array_text)
        
        return ohe.toarray()
