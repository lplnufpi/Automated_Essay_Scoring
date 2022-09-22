import stanza
import re

class Cand_refs_adj_sents:
    referents = {r'^elas$': 'fp',
                 r'^nelas$': 'fp',
                 r'^delas$': 'fp',
                 r'^.*-nas$': 'fp',
                 r'^.*-las$': 'fp',
                 r'^.*-as$': 'fp',
                 r'^eles$': 'mp',
                 r'^neles$': 'mp',
                 r'^deles$': 'mp',
                 r'^.*-nos$': 'mp',
                 r'^.*-los$': 'mp',
                 r'^.*-os$': 'mp',
                 r'^ela$': 'fs',
                 r'^nela$': 'fs',
                 r'^dela$': 'fs',
                 r'^.*-na$': 'fs',
                 r'^.*-la$': 'fs',
                 r'^.*-a$': 'fs',
                 r'^ele$': 'ms',
                 r'^nele$': 'ms',
                 r'^dele$': 'ms',
                 r'^.*-no$': 'ms',
                 r'^.*-lo$': 'ms',
                 r'^.*-o$': 'ms',
                 r'^lhes$': 'ap',
                 r'^lhe$': 'as',
                 }

    def __init__(self, nsentences=1):

        self.nsentences = nsentences

        self.compiled_referents = {}
        for regex, category in self.referents.items():
            self.compiled_referents[regex] = re.compile(regex)


    
    def find_candidates(self, sentences, indices, category):
        """Find nouns of a certain gender/number in a list of sentences.

        :sentences: the tagged sentences.
        :indices: the indices of the sentences to be searched.
        :category: the category of nouns to look for (ms, mp, fs, fp, as, ap).
        :rp: the resource pool to use.
        :returns: a list of nouns matching the category.
        """

        candidates = []
        for i in indices:
            if (i, category) not in self.computed_categories:
                sentence = sentences[i]
                curr_candidates = []

                for word in sentence.words:
                    if word.upos == 'NOUN':
                        if not word.feats:
                            continue
                        if category == 'ap':
                            if 'Number=Plur' in word.feats:
                                curr_candidates.append(word)
                        elif category == 'as':
                            if 'Number=Sing' in word.feats:
                                curr_candidates.append(word)
                        elif category == 'fp':
                            if 'Number=Plur' in word.feats and 'Gender=Fem' in word.feats:
                                curr_candidates.append(word)
                        elif category == 'fs':
                            if 'Number=Sing' in word.feats and 'Gender=Fem' in word.feats:
                                curr_candidates.append(word)
                        elif category == 'mp':
                            if 'Number=Plur' in word.feats and 'Gender=Masc' in word.feats:
                                curr_candidates.append(word)
                        elif category == 'ms':
                            if 'Number=Sing' in word.feats and 'Gender=Masc' in word.feats:
                                curr_candidates.append(word)

                self.computed_categories[(i, category)] = curr_candidates

            candidates.extend(self.computed_categories[(i, category)])

        return candidates



    def value_for_text(self, sentences):
        if len(sentences) <= 1:
            return 0

        ncandidates = 0
        self.computed_categories = {}
        for isent in range(1, len(sentences)):
            iprev_sents = range(max(isent - self.nsentences, 0), isent)

            for word in sentences[isent].words:
                for ref, category in self.referents.items():
                    if self.compiled_referents[ref].match(word.text.lower()):
                        candidates = self.find_candidates(sentences, iprev_sents,category)
                        ncandidates += len(candidates)
        return ncandidates / (len(sentences) - 1) if len(sentences) - 1 else 0




class Cand_refs_pronoum_adj_sents(Cand_refs_adj_sents):
    referents = {r'^elas$': 'fp',
                 r'^eles$': 'mp',
                 r'^ela$': 'fs',
                 r'^ele$': 'ms',
                 }

    def __init__(self):
        super(Cand_refs_pronoum_adj_sents, self).__init__(nsentences=1)



class Cand_refs_demo_pronoum_adj_sents(Cand_refs_adj_sents):
    referents = {r'^este$': 'ms',
                 r'^estes$': 'mp',
                 r'^esta$': 'fs',
                 r'^estas$': 'fp',
                 r'^isto$': 'ms',
                 r'^esse$': 'ms',
                 r'^esses$': 'mp',
                 r'^essa$': 'fs',
                 r'^essas$': 'fp',
                 r'^nesse$': 'ms',
                 r'^nesses$': 'mp',
                 r'^nessa$': 'fs',
                 r'^nessas$': 'fp',
                 r'^neste$': 'ms',
                 r'^nestes$': 'mp',
                 r'^nesta$': 'fs',
                 r'^nestas$': 'fp',
                 r'^isso$': 'ms',
                 r'^nisso$': 'ms',
                 r'^nisto$': 'ms',
                 r'^aquele$': 'ms',
                 r'^aqueles$': 'mp',
                 r'^aquela$': 'fs',
                 r'^aquelas$': 'fp',
                 r'^aquilo$': 'ms',
                 r'^naquele$': 'ms',
                 r'^naqueles$': 'mp',
                 r'^naquela$': 'fs',
                 r'^naquelas$': 'fp',
                 r'^naquilo$': 'sm',
                 r'^deste$': 'ms',
                 r'^destes$': 'mp',
                 r'^desta$': 'fs',
                 r'^destas$': 'fp',
                 r'^desse$': 'ms',
                 r'^desses$': 'mp',
                 r'^dessa$': 'fs',
                 r'^dessas$': 'fp',
                 r'^daquele$': 'ms',
                 r'^daqueles$': 'mp',
                 r'^daquela$': 'fs',
                 r'^daquelas$': 'fp',
                 r'^disto$': 'ms',
                 r'^disso$': 'ms',
                 r'^daquilo$': 'ms',
                 r'^o mesmo$': 'ms',
                 r'^a mesma$': 'fs',
                 r'^os mesmos$': 'mp',
                 r'^as mesmas$': 'fp',
                 r'^o próprio$': 'ms',
                 r'^a própria$': 'fs',
                 r'^os próprios$': 'mp',
                 r'^as próprias$': 'fp',
                 r'^tal$': 'sm',
                 r'^tais$': 'sp',
                 }

    def __init__(self):
        super(Cand_refs_demo_pronoum_adj_sents, self).__init__(nsentences=1)





if __name__ == "__main__":
    stz = stanza.Pipeline(lang='pt', processors='tokenize,mwt,pos')

    texto = "Ouvi dizer que estão tentando incluir orientação nutricional no currículo escolar. Sou totalmente defensor dessa proposta."

    doc = stz(texto)

    anaf = Cand_refs_demo_pronoum_adj_sents()

    print(anaf.value_for_text(doc.sentences))