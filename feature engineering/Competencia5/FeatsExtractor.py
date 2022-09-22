from lib2to3.pgen2 import token
import stanza
import pandas as pd
import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords
from gensim import corpora, models, similarities
from gensim.matutils import cossim, sparse2full, full2sparse
import spacy
import re
from lsa.lsa import LsaSpace
from itertools import chain
from numpy import dot
from scipy.linalg import pinv


class FeatsExtractor:
    def __init__(self):
        self.stz = stanza.Pipeline(lang='pt', processors='tokenize,mwt,pos')
        self.entset = spacy.load("pt_core_news_lg")
       
        #carrega arquivo com os operadores argumentativos
        arq = open('operadores_argumentativos.txt',"r")
        self.op_arg = [c.strip() for c in arq.readline().split(';')]
        arq.close()


        self.temas = []
        self.prompts = pd.read_csv('prompts.csv', converters={'description': eval})
        for tema in self.prompts.itertuples():
            t = '\n'.join(tema.description)
            self.temas.append(self.remove_stopword(t))

        #carrega modelo LSA para calcular similaridade entre sentenças e parágrafos
        self.lsa = LsaSpace('lsa/brwac_full_lsa_word_dict.pkl')

        self.operadores = ['logo', 'portanto', 'então','assim', 'enfim', 'consequentemente', 'por isso', 'por conseguinte', 'de modo que', 'por fim', 'conclui-se', 'é necessário', 'é preciso','é importante']

    

    def remove_stopword(self, txt):
        stoplist = stopwords.words("portuguese")
        txt = [p.lower() for p in word_tokenize(txt, language='portuguese') if p.isalpha() and p not in stoplist]
        return txt


    def essay_prompt_similarity(self, id_tema, redacao):
        redacao = self.remove_stopword(redacao)
        sims = self.lsa.compute_similarity(redacao, self.temas[id_tema])
        return sims


    
    def similarity_paragraphs(self, paragrafos):
        stoplist = stopwords.words("portuguese")
        tokens_paragraphs = [[t.lower() for t in word_tokenize(p, language='portuguese') if t.isalpha() and t not in stoplist]  for p in paragrafos]
        similarities = []
        parags = tokens_paragraphs.pop()
        for p in parags:
            similarities.append(self.lsa.compute_similarity(tokens_paragraphs[-1], p))
        return sum(similarities) / len(similarities) if similarities else 0





    def contatodor(self,i):
        print('Redação: ', end='')
        print(i, end='')
        print('\r', end='') 
        

    


    def extractText(self, essay, id_tema):
        doc = self.stz(essay)

        stoplist = stopwords.words("portuguese")

        paragrafos = essay.split('\n')
        sentences = [s.text for s in doc.sentences]
        
        cont_verb = 0
        cont_verb_1_pers = 0
        cont_verb_imp  = 0
        for sentence in doc.sentences:
            for word in sentence.words:
                if word.upos == 'VERB':
                    cont_verb += 1
                    if 'Person=1' in sentence.words[0].feats: 
                        cont_verb_1_pers += 1
                    if 'Mood=Imp' in sentence.words[0].feats: 
                        cont_verb_imp += 1

        #proporção de verbos na primeira pessoa em relação à quantidade total de verbos
        prop_verb_pers_1_verb = cont_verb_1_pers/cont_verb

        #proporção de verbos no imperativo em relação à quantidade de total de verbos
        prop_verb_imp_verb = cont_verb_imp/cont_verb
           

        #Proporção de operadores conclusivos em relação à quantidade de palavras de sentenças
        quant_op_arg = 0
        for op in self.operadores:
            if re.search("[^a-zA-Z0-9_]"+op+"[^a-zA-Z0-9_]", essay):
                quant_op_arg += 1
        op_arg_ratio = quant_op_arg/len(sentences)

        #similaridade com os textos motivadores    
        similar_essay_prompt = self.essay_prompt_similarity(id_tema, essay)

        #Similaridade entre o ultimo parágrafo e o restante dos parágarfos
        lsa_adjacent_paragraphs = self.similarity_paragraphs(paragrafos)        


        return [op_arg_ratio, similar_essay_prompt, lsa_adjacent_paragraphs, prop_verb_pers_1_verb, prop_verb_imp_verb] 



    def extractCorpus(self, corpus):
        featsets = []
        targetset = []
        i = 0

        for redacao in corpus.itertuples():
            i += 1
            self.contatodor(i)
            texto = '\n'.join(redacao.essay)   
            if texto != "":
                feats = self.extractText(texto, redacao.prompt) #obtem as features da redação
                featsets.append(feats) #adiciona no vetor de features
                targetset.append(redacao.c5) #adiciona o target no vetor de targets

        vecfeats = np.array(featsets) #converte o vetor de features para um array numpy

        return vecfeats, targetset


if __name__ == '__main__':

    train = pd.read_csv('essay-br.csv', converters={'essay': eval})

    ft = FeatsExtractor()

    X, y = ft.extractCorpus(train.loc[0:10])

    print(X, y)

    
