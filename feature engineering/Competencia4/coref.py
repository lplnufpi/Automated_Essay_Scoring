
import stanza
from nltk import stem

class Refer_adj_sents:
    def get_sentences(self, sents):
        sentences = []
        for sentence in sents:
            sentences.append([token.text for token in sentence.words if token.upos=='NOUN' or token.upos=='PRON'])
        return sentences
    
    
    def word_pairs(self, s1, s2):
        for w1 in s1:
            for w2 in s2:
                yield w1.lower(), w2.lower()
    

    def sentence_pairs(self, sents):
        sentences = self.get_sentences(sents)
        for i in range(1, len(sentences)):
            yield sentences[i], sentences[i - 1]
                
    
    def value_for_text(self, sentences):
        if len(sentences) <= 1:
            return 0
        matches = 0
        pairs = 0
        for s1, s2 in self.sentence_pairs(sentences):
            for w1, w2 in self.word_pairs(s1, s2):
                if w1 == w2:
                    matches += 1
            pairs += 1

        return matches / pairs if pairs else 0




class Contword_adj_sents(Refer_adj_sents):
    def get_sentences(self, sents):
        sentences = []
        for sentence in sents:
            sentences.append([token.text for token in sentence.words if token.upos=='NOUN' or token.upos=='VERB' or token.upos=='ADJ' or token.upos=='ADV'])
        return sentences




class Stem_adj_sents(Refer_adj_sents):
    def __init__(self):
        self.stemmer = stem.RSLPStemmer()

    def get_sentences(self, sents):
        sentences = []
        for sentence in sents:
            sentences.append([self.stemmer.stem(token.text) for token in sentence.words if token.upos=='NOUN' or token.upos=='VERB' or token.upos=='ADJ' or token.upos=='ADV'])
        return sentences




class Refer_all_pair_sents(Refer_adj_sents):
    def sentence_pairs(self, sents):
        sentences = self.get_sentences(sents)
        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                yield sentences[i], sentences[j]




class Stem_all_pair_sents(Refer_adj_sents):
    def __init__(self):
        self.stemmer = stem.RSLPStemmer()

        
    def get_sentences(self, sents):
        sentences = []
        for sentence in sents:
            sentences.append([self.stemmer.stem(token.text) for token in sentence.words if token.upos=='NOUN' or token.upos=='VERB' or token.upos=='ADJ' or token.upos=='ADV'])
        return sentences
    

    def sentence_pairs(self, sents):
        sentences = self.get_sentences(sents)
        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                yield sentences[i], sentences[j]



if __name__ == '__main__':
    stz = stanza.Pipeline(lang='pt', processors='tokenize,mwt,pos')

    texto = "As crianças aprendem muito rápido. Pesquisas mostram que até os três anos de vida, o desenvolvimento do cérebro ocorre num ritmo bem acelerado. O que os pais fazem no dia-a-dia, como ler, cantar e demonstrar carinho, é crucial para o desenvolvimento saudável da criança. Mas de acordo com certo estudo, apenas cerca da metade dos pais com crianças entre dois e oito anos lê diariamente para elas. Você talvez se pergunte: ‘Será que ler para o meu filho realmente faz diferença?'"

    doc = stz(texto)

    coref = Stem_all_pair_sents()

    print(coref.value_for_text(doc.sentences))


