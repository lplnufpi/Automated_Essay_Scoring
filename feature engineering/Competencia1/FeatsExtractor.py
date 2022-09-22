
import stanza
import pandas as pd
import numpy as np
import re
from nltk import word_tokenize
import spacy
import hunspell
from cogroo4py.cogroo import Cogroo
import pickle
import statistics


class FeatsExtractor:
    def __init__(self):
        self.stz = stanza.Pipeline(lang='pt', processors='tokenize,mwt,pos,depparse', depparse_pretagged=True)
        
        self.spc = spacy.load("pt_core_news_lg")

        self.spellchecker = hunspell.HunSpell('aux/dicionarios/hunspell/pt_BR.dic','aux/dicionarios/hunspell/pt_BR.aff') 
        self.spellchecker.add_dic('aux/dicionarios/DELAF_PB_v2/Delaf2015v04.dic')

        with open('aux/conjuncoes.txt',"r") as f:
            self.conjuncoes = f.readline().split(';')
            f.close() 

        self.cogroo = Cogroo() 

        
    def grammar_check(self, text):
        try:
            gramerror = self.cogroo.grammar_check(text)
            cont = 0
            for erro in gramerror.mistakes:
                    if not str(erro).startswith('[space:'):
                        cont += 1    
            return cont
        except:
            print("=======> Erro ao conectar com Cogroo <==============") 
            return 0

    
    def ents_nom(self, essay):
        ents = []
        en = self.spc(essay)
        for ent in en.ents:
            if len(ent.text) > 1:
                for e in ent.text.split():
                    if len(e) > 2:
                        ents.append(e)
            else:
                ents.append(ent.text)
        return set(ents)

    
    def spell_check(self, tokens, ents_nom):
        cont = 0
        for t in tokens:
            if not self.spellchecker.spell(t) and t not in ents_nom:
                cont += 1
        return cont

    
    def contador(self,i):
        print('Redação: ', end='')
        print(i, end='')
        print('\r', end='') 
        


    def extractText(self, essay):
        doc = self.stz(essay)
        paragrafos = [p for p in essay.split('\n') if p != '']
        sentences = [s.text for s in doc.sentences]
        tokens = [t for t in word_tokenize(essay, language='portuguese') if t.isalpha()]
              
        #quantidade de palavras erradas que não estão presentes no conjunto de entidade nomeadas
        quant_pal_erradas = self.spell_check(tokens, self.ents_nom(essay))

        #quantidade de erros gramaticais
        quant_gramerros = self.grammar_check(essay)

        #Quantidade média de sentenças por parágrafos
        mean_sent_parag = len(sentences)/len(paragrafos)

        #Tamanho médio dos parágrafos
        mean_parag = statistics.mean([len(p) for p in paragrafos])
 
        # #quantidade de parágrafos com apenas uma sentença.
        # quant_parag_um_sent = 0
        # for p in paragrafos:
        #     if len(p.split('.')) == 1:
        #         quant_parag_um_sent += 1
      
        quant_conj_arg = 0
        cont_conj_ini_sent  = 0
        for conj in self.conjuncoes:
            #Proporção de conjunções em relação à quantidade de sentenças do texto
            if re.search("[^a-zA-Z0-9_]"+conj+"[^a-zA-Z0-9_]", essay):
                quant_conj_arg += 1
            #quantidade de sentenças iniciadas com conjunções
            for sent in sentences:
                if re.match("[^a-zA-Z0-9_]*"+conj+"[^a-zA-Z0-9_]", sent):
                    cont_conj_ini_sent += 1
        conj_sents_ratio = quant_conj_arg/len(sentences)

        
        cont_oracoes = 0
        cont_sent_simples = 0
        cont_sent_comp = 0
        cont_sent_ini_ger = 0
        cont_sent_sem_verb = 0
        cont_word_ant_verb_prin = 0
        cont_adv_ant_verb = 0
        for sentence in doc.sentences:
            cont_verb = 0
            for word in sentence.words:
                if word.upos == 'VERB':
                    cont_verb += 1 #quantidade de verbos na sentença
                    cont_oracoes += 1#quantidade de orações na redacao

            #quantidade de sentenças sem verbo
            if cont_verb == 0:
                cont_sent_sem_verb += 1

            #quantidade de sentenças com apenas uma oração
            if cont_verb == 1:
                cont_sent_simples += 1
            
            #quantidade de sentenças com mais de uma oração
            if cont_verb > 1:
                cont_sent_comp += 1
            
            #quantidade de sentenças iniciados com gerúndios
            if sentence.words[0].upos == 'VERB' or sentence.words[0].upos == 'AUX':
                if sentence.words[0].feats:
                    if 'VerbForm=Ger' in sentence.words[0].feats: 
                        cont_sent_ini_ger += 1 
            
            #quantidade de palavras antes do verbo principal
            for word in sentence.words:
                if word.head == 0:
                    break
                else:
                    if word.text.isalpha():
                        cont_word_ant_verb_prin += 1

            #Quantidade de orações com advérbio antes do verbo principal
            adv = False
            for word in sentence.words:
                if word.upos == 'ADV' and not adv:
                    adv = True
                    cont_adv_ant_verb += 1
                if word.upos == 'VERB':
                    adv = False



        #Quantidade de orações com advérbio antes do verbo principal em relação à quantidade de orações do texto
        adv_before_main_verb_ratio = cont_adv_ant_verb/cont_oracoes

        #média de orações por sentenças
        mean_ora_sent = cont_oracoes/len(sentences)

        #Proporção entre a quantidade de verbos e palavras
        prop_verb_pal = cont_oracoes/len(tokens)

        #Média de palavras antes dos verbos principais das orações principais das sentenças
        mean_word_ant_verb_pri = cont_word_ant_verb_prin/len(sentences)


        return [quant_pal_erradas,
                quant_gramerros,
                mean_sent_parag,
                mean_parag,
                cont_sent_simples,
                cont_sent_comp,
                cont_sent_ini_ger,
                cont_sent_sem_verb,
                mean_ora_sent,
                mean_word_ant_verb_pri,
                conj_sents_ratio,
                cont_conj_ini_sent,
                adv_before_main_verb_ratio,
                prop_verb_pal,] 



    def extractCorpus(self, corpus):
        featsets = []
        targetset = []
        i = 0

        for redacao in corpus.itertuples():
            i += 1
            self.contador(i)
            texto = '\n'.join(redacao.essay)  
            if texto != "":
                feats = self.extractText(texto) #obtem as features da redação
                featsets.append(feats) #adiciona no vetor de features
                targetset.append(redacao.c1) #adiciona o target no vetor de targets

        vecfeats = np.array(featsets) #converte o vetor de features para um array numpy

        # listFeatsTargets = []
        # for t in zip(featsets, targetset):
        #     listFeatsTargets.append(t)   

        return vecfeats, targetset



if __name__ == '__main__':

    train = pd.read_csv('essay-br.csv', converters={'essay': eval})

    ft = FeatsExtractor()

    X, y = ft.extractCorpus(train.loc[0:10])

    print(X, y)

 