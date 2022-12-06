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

    

    def remove_stopword(self, txt):
        stoplist = stopwords.words("portuguese")
        txt = [p.lower() for p in word_tokenize(txt, language='portuguese') if p.isalpha() and p not in stoplist]
        return txt


    def essay_prompt_similarity(self, id_tema, redacao):
        redacao = self.remove_stopword(redacao)
        sims = self.lsa.compute_similarity(redacao, self.temas[id_tema])
        return sims


    def all_pairs_sentences(self, l):
        for i in range(len(l)):
            for j in range(i + 1, len(l)):
                yield l[i], l[j]

    
    def adjacent_pairs(self, l):
        for i in range(len(l) - 1):
            yield l[i], l[i + 1]

    
    def givenness_get_pairs(self, sentences):
        for i in range(1, len(sentences)):
            past_sentences = sentences[:i]
            past_tokens = list(chain.from_iterable(past_sentences))
            yield sentences[i], past_tokens

    
    def similarity_all_sentences(self, tokens_sentences):
        similarities = []
        for s1,s2 in self.all_pairs_sentences(tokens_sentences):
            similarities.append(self.lsa.compute_similarity(s1, s2))
        return sum(similarities) / len(similarities) if similarities else 0


    def similarity_adj_sentences(self, tokens_sentences):
        similarities = []
        for s1,s2 in self.adjacent_pairs(tokens_sentences):
            similarities.append(self.lsa.compute_similarity(s1, s2))
        return sum(similarities) / len(similarities) if similarities else 0

    
    def similarity_paragraphs(self, paragrafos):
        stoplist = stopwords.words("portuguese")
        tokens_paragraphs = [[t.lower() for t in word_tokenize(p, language='portuguese') if t.isalpha() and t not in stoplist]  for p in paragrafos]
        similarities = []
        for p1,p2 in self.adjacent_pairs(tokens_paragraphs):
            similarities.append(self.lsa.compute_similarity(p1, p2))
        return sum(similarities) / len(similarities) if similarities else 0



    def calc_givenness(self, tokens_sentences):
        similarities = []
        for s, pt in self.givenness_get_pairs(tokens_sentences):
            similarities.append(self.lsa.compute_similarity(s, pt))
        givenness = sum(similarities) / len(similarities) if similarities else 0
        return givenness

    

    def calc_span(self, tokens_sentences):
        space = self.lsa
        num_topics = space.num_topics

        if len(tokens_sentences) < 2:
            return 0
        spans = np.zeros(len(tokens_sentences) - 1)
        for i in range(1, len(tokens_sentences)):
            past_sentences = tokens_sentences[:i]
            span_dim = len(past_sentences)

            if span_dim > num_topics - 1:
                # It's not clear, from the papers I read, what should be done
                # in this case. I did what seemed to not imply in loosing
                # information.
                beginning = past_sentences[0:span_dim - num_topics]
                past_sentences[0] = list(chain.from_iterable(beginning))

            past_vectors = [sparse2full(space.get_vector(sent), num_topics)
                            for sent in past_sentences]

            curr_vector = sparse2full(space.get_vector(tokens_sentences[i]), num_topics)
            curr_array = np.array(curr_vector).reshape(num_topics, 1)

            A = np.array(past_vectors).transpose()

            projection_matrix = dot(dot(A,
                                        pinv(dot(A.transpose(),
                                                 A))),
                                    A.transpose())

            projection = dot(projection_matrix, curr_array).ravel()

            spans[i - 1] = cossim(full2sparse(curr_vector),
                                  full2sparse(projection))

        return spans.mean()




    def contatodor(self,i):
        print('Redação: ', end='')
        print(i, end='')
        print('\r', end='') 
        


    def extractText(self, essay, id_tema):
        doc = self.stz(essay)

        stoplist = stopwords.words("portuguese")

        paragrafos = essay.split('\n')
        sentences = [s.text for s in doc.sentences]
        tokens = [p for p in word_tokenize(essay, language='portuguese') if p.isalpha()]
        
        #diversidade lexical    
        diver_lex = len(set(tokens))/len(tokens)            

        #Quantidade de entidades nomeadas na redação
        ents_nom = []
        en = self.entset(essay)
        quant_en = len(en.ents)    

        #Proporção de operadores argumentativos em relação à quantidade de palavras de sentenças
        quant_op_arg = 0
        for op in self.op_arg:
            if re.search("[^a-zA-Z0-9_]"+op+"[^a-zA-Z0-9_]", essay):
                quant_op_arg += 1
        op_arg_ratio = quant_op_arg/len(sentences)

        tokens_sents = [[p.lower() for p in word_tokenize(s, language='portuguese') if p.isalpha() and p not in stoplist ]  for s in sentences]

        #similaridade com os textos motivadores    
        similar_essay_prompt = self.essay_prompt_similarity(id_tema, essay)

        #Média de similaridade entre pares de sentenças adjacentes

        lsa_sent_adj_mean = self.similarity_adj_sentences(tokens_sents)

        #Média da similaridade entre todos os pares de sentenças
        lsa_sent_all_mean = self.similarity_all_sentences(tokens_sents)

        #Similaridade entre pares de parágrafos adjacentes
        lsa_adjacent_paragraphs = self.similarity_paragraphs(paragrafos)        

        #Média do *givenness* da cada sentença do texto (Similiaridade entre a sentença e todas as outras anteriosres a ela)
        givenness_mean = self.calc_givenness(tokens_sents)

        #Média do *span* da cada sentença do texto, a partir da segunda
        spans_mean = self.calc_span(tokens_sents)

        return [quant_en, op_arg_ratio, diver_lex, similar_essay_prompt, lsa_sent_adj_mean, lsa_sent_all_mean, lsa_adjacent_paragraphs, givenness_mean, spans_mean] 



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
                targetset.append(redacao.c3) #adiciona o target no vetor de targets

        vecfeats = np.array(featsets) #converte o vetor de features para um array numpy


        return vecfeats, targetset


if __name__ == '__main__':

    train = pd.read_csv('essay-br.csv', converters={'essay': eval})

    ft = FeatsExtractor()

    X, y = ft.extractCorpus(train.loc[0:10])

    print(X, y)

    
