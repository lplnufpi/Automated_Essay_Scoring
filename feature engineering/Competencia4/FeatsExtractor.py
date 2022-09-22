from lib2to3.pgen2 import token
from click import prompt
import stanza
import pandas as pd
import numpy as np
from numpy import dot
from scipy.linalg import pinv
from nltk.probability import FreqDist
from nltk import Text, corpus, word_tokenize
from nltk.corpus import stopwords
from nltk.translate.bleu_score import sentence_bleu
from collections import defaultdict
from gensim import corpora
from gensim import models
from gensim import similarities
from gensim.matutils import cossim, sparse2full, full2sparse
import rouge
import spacy
from math import log
import re
from lsa.lsa import LsaSpace
from itertools import chain

from anaforas import *
from coref import *


class FeatsExtractor:
    def __init__(self):
        self.stz = stanza.Pipeline(lang='pt', processors='tokenize,mwt,pos')

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

    

    def remove_stopword(self, txt):
        stoplist = corpus.stopwords.words("portuguese")
        txt = [p for p in word_tokenize(txt, language='portuguese') if p.isalpha() and p not in stoplist]
        return txt


    def essay_prompt_similarity(self, id_tema, redacao):
        redacao = self.remove_stopword(redacao)
        sims = self.lsa.compute_similarity(redacao, self.temas[id_tema])
        return sims
 

    def contatodor(self,i):
        print('Redação: ', end='')
        print(i, end='')
        print('\r', end='') 
        


    def extractText(self, essay, id_tema):
        doc = self.stz(essay)

        paragrafos = essay.split('\n')
        sentences = [s.text for s in doc.sentences]
        tokens = [p for p in word_tokenize(essay, language='portuguese') if p.isalpha()]

        
        #quantidade de orações e quantidade de caracteres de pontuação
        qt_oracoes = 0
        qt_pontuacao = 0
        for sent in doc.sentences:
            for w in sent.words:
                if w.upos == 'VERB':
                    qt_oracoes += 1
                elif w.upos == 'PUNCT':
                    qt_pontuacao += 1

        #quantidade de parágrafos
        at_paragrafos = len(paragrafos)

        #quantidade de sentenças
        qt_sents = len(sentences)

        #quantidade de palavras
        qt_tokens = len(tokens)        
   

        #Proporção de operadores argumentativos em relação à quantidade de palavras do texto
        quant_op_arg = 0
        for op in self.op_arg:
            if re.search("[^a-zA-Z0-9_]"+op+"[^a-zA-Z0-9_]", essay):
                quant_op_arg += 1
        op_arg_ratio = quant_op_arg/len(tokens)


        #COESÃO SEMÂNTICA
        
        #similaridade com os textos motivadores    
        similar_essay_prompt = self.essay_prompt_similarity(id_tema, essay)

        #similaridade entre os pares de sentenças adjacentes
        lsa_adjacent_sentences = self.lsa.similarity_adjacent_pairs(sentences) 
        
        #Média da similaridade entre todos os pares de sentenças
        lsa_sent_all_mean = self.lsa.similarity_pair_all_sentences(sentences)

        #Similaridade entre pares de parágrafos adjacentes
        lsa_adjacent_paragraphs = self.lsa.similarity_adjacent_pairs(paragrafos)        

        #Média do *givenness* da cada sentença do texto (Similiaridade entre a sentença e todas as outras anteriosres a ela)
        givenness_mean = self.lsa.calc_givenness_mean(sentences)

        #Média do *span* da cada sentença do texto, a partir da segunda
        span_mean = self.lsa.calc_span_mean(sentences)


        #COESÃO REFERENCIAL
        #Média das proporções de candidatos a referentes na sentença anterior em relação aos pronomes pessoais do caso reto nas sentenças
        adj_sents_anaf = Cand_refs_adj_sents(1).value_for_text(doc.sentences)

        #Média de candidatos a referente, na sentença anterior, por pronome anafórico do caso reto
        adj_sents_pronoum_anaf = Cand_refs_pronoum_adj_sents().value_for_text(doc.sentences)

        #Média de candidatos a referente, na sentença anterior, por pronome demonstrativo anafórico
        adj_sents_dempron_anaf = Cand_refs_demo_pronoum_adj_sents().value_for_text(doc.sentences)

        #Média das proporções de candidatos a referentes nas 5 sentenças anteriores em relação aos pronomes anafóricos das sentenças
        anaf_sents_5 = Cand_refs_adj_sents(5).value_for_text(doc.sentences)

        #Quantidade média de referentes que se repetem nos pares de sentenças adjacentes do texto
        ref_repete_adj_sents = Refer_adj_sents().value_for_text(doc.sentences)

        #Quantidade média de palavras de conteúdo que se repetem nos pares de sentenças adjacentes do texto
        cont_repete_adj_sents = Contword_adj_sents().value_for_text(doc.sentences)

        #Quantidade média de radicais de palavras de conteúdo que se repetem nos pares de sentenças adjacentes do texto.
        stem_repete_adj_sents = Stem_adj_sents().value_for_text(doc.sentences)

        #Quantidade média de referentes que se repetem nos pares de sentenças do texto
        cont_repete_all_sents = Refer_all_pair_sents().value_for_text(doc.sentences)

        #Quantidade média de radicais de palavras de conteúdo que se repetem nos pares de sentenças do texto
        stem_repete_all_sents = Stem_all_pair_sents().value_for_text(doc.sentences)


        return [op_arg_ratio, similar_essay_prompt, lsa_adjacent_sentences,
                lsa_sent_all_mean, lsa_adjacent_paragraphs, givenness_mean, span_mean, at_paragrafos,
                qt_sents, qt_oracoes, qt_tokens, qt_pontuacao, adj_sents_anaf, adj_sents_pronoum_anaf,
                adj_sents_dempron_anaf, anaf_sents_5, ref_repete_adj_sents, cont_repete_adj_sents,
                stem_repete_adj_sents, cont_repete_all_sents, stem_repete_all_sents ] 




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
                targetset.append(redacao.c4) #adiciona o target no vetor de targets

        vecfeats = np.array(featsets) #converte o vetor de features para um array numpy  

        return vecfeats, targetset



if __name__ == '__main__':

    train = pd.read_csv('essay-br.csv', converters={'essay': eval})

    ft = FeatsExtractor()

    X, y = ft.extractCorpus(train.loc[0:10])

    print(X, y)