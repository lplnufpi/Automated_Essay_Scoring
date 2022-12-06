from lib2to3.pgen2 import token
from click import prompt
import stanza
import pandas as pd
import numpy as np
from nltk.probability import FreqDist
from nltk import Text, corpus, word_tokenize
from nltk.corpus import stopwords
from nltk.translate.bleu_score import sentence_bleu
from collections import defaultdict
from gensim import corpora
from gensim import models
from gensim import similarities
import rouge
import spacy
from math import log
from lsa.lsa import LsaSpace


class FeatsExtractor:
    def __init__(self):
        self.stz = stanza.Pipeline(lang='pt', processors='tokenize,mwt,pos')
        self.entset = spacy.load("pt_core_news_lg")

        # self.temas = []
        # self.prompts = pd.read_csv('corpus/new_essay_br/new_prompts.csv')
        # for tema in self.prompts.itertuples():
        #     t = '\n'.join(tema.description)
        #     self.temas.append(t)
        # processed_corpus = [self.remove_stopword(t) for t in self.temas]
        # dictionary = corpora.Dictionary(processed_corpus)
        # bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]
        # self.tfidf = models.TfidfModel(bow_corpus)

        self.temas = []
        self.prompts = pd.read_csv('prompts.csv', converters={'description': eval})
        for tema in self.prompts.itertuples():
            t = '\n'.join(tema.description)
            self.temas.append(self.remove_stopword(t))

        #carrega modelo LSA para calcular similaridade entre sentenças e parágrafos
        self.lsa = LsaSpace('lsa/brwac_full_lsa_word_dict.pkl')


    def remove_stopword(self, txt):
        stoplist = corpus.stopwords.words("portuguese")
        txt = [p for p in word_tokenize(txt, language='portuguese') if p.isalpha() and p not in stoplist]
        return txt


    def essay_prompt_similarity(self, id_tema, redacao):
        redacao = self.remove_stopword(redacao)
        sims = self.lsa.compute_similarity(redacao, self.temas[id_tema])
        return sims
    

    def extractText(self,essay, id_tema):
        paragrafos = essay.split('\n')
        tokens = [p for p in word_tokenize(essay, language='portuguese') if p.isalpha()]
        
        quant_paragrafos = len(paragrafos) #quantidade de parágrafos
               
        tam_med_paragrafos = len(tokens)/quant_paragrafos  #tamanho (quantidade de palavras) médio dos parágrafos
        
        diver_lex = len(set(tokens))/len(tokens) #diversidade lexical

        quant_en = 0
        quant_verb_pron_p1 = 0
        doc = self.stz(essay)
        for sent in doc.sentences:
            for word in sent.words:
                if word.pos == 'VERB' or word.pos == 'AUX' or word.pos == 'PRON':
                    if word.feats:
                        if 'Person=1' in word.feats:
                            quant_verb_pron_p1 += 1 #quantidade de verbos e pronomes na primeira pessoa 
                

        #entidades nomeadas na redação
        ents_nom = []
        en = self.entset(essay)
        for ent in en.ents:
            ents_nom.append(ent.text)
        quant_en = len(ents_nom)    

        #similaridade com os textos motivadores    
        similar_essay_prompt = self.essay_prompt_similarity(id_tema, essay)


        #Estatística de Honoré
        hpx = FreqDist(essay).hapaxes()
        honore = 100 * log(len(tokens), 10) / (1 - len(hpx) / len(set(tokens)))
        
        return [quant_paragrafos, tam_med_paragrafos, quant_en, quant_verb_pron_p1, diver_lex, similar_essay_prompt, honore] 


    def extractCorpus(self, corpus):
        
        featsets = []
        targetset = []
        i = 0

        for redacao in corpus.itertuples():
            i += 1
            print('Redação: ', end='')
            print(i, end='')
            print('\r', end='') 

            texto = '\n'.join(redacao.essay)
                
            if texto != "":
                try:
                    feats = self.extractText(texto, redacao.prompt) #obtem as features da redação
                    featsets.append(feats) #adiciona no vetor de features
                    targetset.append(redacao.c2) #adiciona o target no vetor de targets
                except:
                    print('====> erro na redação', i)
                
        vecfeats = np.array(featsets) #converte o vetor de features para um array numpy
 

        return vecfeats, targetset


if __name__ == '__main__':

    train = pd.read_csv('essay-br.csv', converters={'essay': eval})

    ft = FeatsExtractor()

    X, y = ft.extractCorpus(train.loc[0:10])

    print(X, y)
