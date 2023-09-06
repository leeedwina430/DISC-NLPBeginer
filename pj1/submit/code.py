#%%
import numpy as np
from tqdm import tqdm
import time
import re
from collections import Counter
import nltk
from nltk.corpus import reuters
from nltk.corpus import brown
import math as calc 
from copy import deepcopy

letters = "abcdefghijklmnopqrstuvwxyz-_'#"
offset = 0.1

class nGram():
    '''Create N-gram model with MLE, interpolation smoothing, dic form'''

    def __init__(self, uni=False, bi=False, tri=False, filename=None, lamb=None) -> None:
        '''build  n-gram models (n<=target)'''
        self.words = self.loadCorpus(filename)
        self.lamb = lamb
        if uni: self.unigram = self.createUnigram(self.words)
        if bi: self.bigram = self.createBigram(self.words)
        if tri: self.trigram = self.createTrigram(self.words)
        

    def loadCorpus(self, filename=None):
        '''load external file which contains raw corpus (.data files)
        corpus data is a long string contains start and end flag for each sent.'''
        # from current file
        if filename:
            print("Loading Corpus from data file")
            
            with open(filename) as f:
                file = f.read()
            words = [re.sub('[,.!“”‘’\—\–\'\"]','',word).lower() for sentence in nltk.sent_tokenize(file) for word in sentence.split()]
            
            return words 
 
        # from nltk corpus
        else:
            print('Load Corpus from nltk')
            # reuters   ACC:86%
            cate = reuters.categories()
            corpus_raw_text = reuters.sents(categories=cate)

            # # brown   ACC:82.2%
            # cate = brown.categories()
            # corpus_raw_text = brown.sents(categories=cate)

            words = []
            for sents in corpus_raw_text:
                sents = ['#'] + sents

                # remove punctuation
                for word in sents[::]: # to remove continuous ';' ';'
                    if (word in ['\'\'', '``', ',', '--', ';', ':', '(', ')', '&', '\'', '!', '?', '.']): sents.remove(word)
                words.extend(sents)
            return words

    def createUnigram(self, words):
        '''create unigram model for words loaded from corpus'''
        print("creating unigram model")
        
        return Counter(words)
    
    def createBigram(self, words):
        '''create Bigram model for words loaded from corpus'''
        print("creating bigram model")
        biwords = []
        for index, item in enumerate(words):
            if index == len(words) - 1: break
            biwords.append(item + ' ' + words[index+1])

        return Counter(biwords)
    
    def createTrigram(self, words):
        '''create Trigram model for words loaded from corpus'''
        print("creating trigram model")
        triwords = []
        for index, item in enumerate(words):
            if index == len(words) - 2: break
            triwords.append(item + ' ' + words[index+1] + ' ' + words[index+2])

        return Counter(triwords)
    
    def prob(self, word, words='', gram='uni', lamb=0.01):
        '''calculate the MLE of n-gram, in log form
            unigram with add one smoothing, bi & trigram with interpolation smoothing'''
        if gram == 'uni':
            return calc.log((self.unigram[word] + 1) / (len(self.words) + len(self.unigram)))
        if gram == 'bi':
            unigram = (self.unigram[word] + 1) / (len(self.words) + len(self.unigram))
            return calc.log(unigram * lamb + ( self.bigram[words] / unigram ) * (1-lamb))

            # add one smoothing
            # return calc.log((self.bigram[words]+1)/(self.unigram[word]+len(self.unigram)))
            
        if gram == 'tri':
            word1 = word.split(' ')[0]
            unigram = (self.unigram[word1] + 1) / (len(self.words) + len(self.unigram))
            bigram = unigram * lamb + ( self.bigram[word] / unigram ) * (1-lamb)
            return calc.log(unigram * lamb * offset + bigram * lamb * (1-offset) + ( self.trigram[words] / bigram ) * (1-lamb))
            
            # add one smoothing
            # return calc.log((self.trigram[words]+1)/(self.bigram[word]+len(self.unigram)))

    def sentProb(self, sent, gram='uni', form='antilog'):
        '''calculate cumulative n-gram MLE for a sentence/phrase'''
        words = [word.lower() for word in sent]
        P = 0
        if gram=='uni': 
            for index, item in enumerate(words): P += self.prob(item,lamb=self.lamb)
        if gram=='bi':
            for index, item in enumerate(words):
                if index == 0: P += self.prob(item, item+' '+words[index+1], 'bi', self.lamb)
                if index == len(words)-1: P += self.prob(item, words[index-1]+' '+item, 'bi', self.lamb)
                else:
                    words1 = words[index-1]+' '+item
                    words2 = item+' '+words[index+1]
                    P += self.prob(item, words1, 'bi', self.lamb) + self.prob(item, words2, 'bi', self.lamb)
        if gram=='tri':
            for index, item in enumerate(words):
                # three # 76.2%
                # if index == 0: P += self.prob(' '.join(words[:2]), ' '.join(words[:3]), 'tri', self.lamb)
                # elif index == len(words)-1: P += self.prob(' '.join(words[-3:-1]), ' '.join(words[-3:]), 'tri', self.lamb)
                # elif index == 1: P += self.prob(' '.join(words[:2]), ' '.join(words[:3]), 'tri', self.lamb) + self.prob(' '.join(words[1:3]), ' '.join(words[1:4]), 'tri', self.lamb)
                # elif index == len(words)-2: P += self.prob(' '.join(words[-3:-1]), ' '.join(words[-3:]), 'tri', self.lamb) + self.prob(' '.join(words[-4:-2]), ' '.join(words[-4:-1]), 'tri', self.lamb)
                # else:
                #     P += self.prob(' '.join(words[index-2:index]), ' '.join(words[index-2:index+1]), 'tri', self.lamb) + \
                #          self.prob(' '.join(words[index-1:index+1]), ' '.join(words[index-1:index+2]), 'tri', self.lamb) + \
                #          self.prob(' '.join(words[index:index+2]), ' '.join(words[index:index+3]), 'tri', self.lamb)
 
                # pre-text only # 75.3%
                if index in [0,1,2]: P += self.prob(' '.join(words[:2]), ' '.join(words[:3]), 'tri', self.lamb)
                elif index in [len(words)-1,len(words)-2,len(words)-3]: P += self.prob(' '.join(words[-3:-1]), ' '.join(words[-3:]), 'tri', self.lamb)
                else:
                    P += self.prob(' '.join(words[index-2:index]), ' '.join(words[index-2:index+1]), 'tri', self.lamb)
                        
        if form=='antilog': return calc.pow(calc.e, P)
        else: return P

# def testNGram(filename):
#     # filename = './data/corpus.data'
#     # myNgram = nGram(True,True,False,filename=filename)
#     myNgram = nGram(True,True,False)
#     print(myNgram.prob('handsome'))    # in log form
#     pp = myNgram.sentProb('you are handsome','bi')
#     pp = myNgram.sentProb('this looks strange','bi')
#     print(pp)s


#%%


class SpellCorrect():
    '''a class for spell correct'''

    def __init__(self,ngram=None,vocab=None,error_dict=None):
        self.ng = ngram
        self.vocab = set(vocab)
        self.vocabStr = ''.join(vocab)
        n = len(letters)
        self.conf_matrix = {"del": [[0 for _ in range(n)] for __ in range(n)], 
                            "ins": [[0 for _ in range(n)] for __ in range(n)], 
                            "sub": [[0 for _ in range(n)] for __ in range(n)], 
                            "trans": [[0 for _ in range(n)] for __ in range(n)]}
        self.fillCM(error_dict,self.conf_matrix)
    
    def known(self,words):
        'Return the subset of words that are actually in the dictionary.'
        return set(w for w in words if w in self.vocab)

    def edits1(self,word):
        "Return all strings that are one edit away from this word."
        splits       = [(word[:i], word[i:])    for i in range(len(word) + 1)]  # word[len(word)+1:] = ''
        
        deletes     = [L + R[1:]               for L, R in splits if R]
        transposes  = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces    = [L + c + R[1:]           for L, R in splits for c in letters]
        inserts     = [L + c + R               for L, R in splits for c in letters]
        
        return set(deletes + transposes + replaces + inserts)

    def edits2(self,word):
        "Return all strings that are two edits away from this word"
        return {e2 for e1 in self.edits1(word) for e2 in self.edits1(e1)}

    def editype(self, Word, Error, raw_conf_mtrix=None):
        '''judge the type of typo, maintain cases. Only consider 1 edit distance'''
        word, error = Word.lower(), Error.lower()
        if self.editDistance(word, error) > 1: return '','','farther'

        for i in range(len(word)):
            if word == error[1:]:
                if raw_conf_mtrix: raw_conf_mtrix['ins'][letters.index('#')][letters.index(error[0])] += 1
                return '#', error[0], 'ins'
            if word[1:] == error:
                if raw_conf_mtrix: raw_conf_mtrix['del'][letters.index(word[0])][letters.index('#')] += 1
                return word[0], '#', 'del'
            
            if i >= len(error):
                if raw_conf_mtrix: raw_conf_mtrix['del'][letters.index(error[i-1])][letters.index(word[i])] += 1
                return error[i-1],  word[i], 'del'
            
            elif word[i] != error[i]:
                if word in [error[:i] + k + error[i:] for k in letters]:
                    if raw_conf_mtrix: raw_conf_mtrix['del'][letters.index(error[i-1])][letters.index(word[i])] += 1
                    return error[i-1],  word[i], 'del'
                
                elif word in [error[:i] + k + error[i+1:] for k in letters]:
                    if raw_conf_mtrix: raw_conf_mtrix['sub'][letters.index(word[i])][letters.index(error[i])] += 1
                    return word[i], error[i], 'sub'
                
                elif word == error[:i] + error[i+1:] or word == error[:-1]: # consider the last case separately
                    if raw_conf_mtrix: raw_conf_mtrix['ins'][letters.index(word[i-1])][letters.index(error[i])] += 1
                    return word[i-1], error[i], 'ins'
                
                elif i+1 < len(error) and word[i] + word[i+1] == error[i+1] + error[i]: 
                    if raw_conf_mtrix: raw_conf_mtrix['trans'][letters.index(word[i])][letters.index(word[i+1])] += 1
                    return word[i], word[i+1], 'trans'
        
        if len(word) < len(error):
            if raw_conf_mtrix: raw_conf_mtrix['ins'][letters.index(word[-1])][letters.index(error[-1])] += 1
            return word[-1], error[-1], 'ins'

        return '','','correct'


    def fillCM(self, error_dict, conf_matrix):
        '''use the error corpus fill in the confusion matrix
            only consider 1 edit distance errors
            conf_matrix = {"del": {}, "ins": {}, "sub": {}, "trans": {}}
        '''
        print('generating confusion matrix')
        for word in tqdm(error_dict):
            misspells = error_dict[word]
            for misspell in misspells:
                self.editype(word, misspell, conf_matrix)


    def pEdit(self, candidate, word, conf_matrix):
        '''calculate the prob of this edit, in log form, with add one smoothing'''
        x,y,wrongType = self.editype(candidate, word, conf_matrix)
        if wrongType == 'farther': return 1 
        if wrongType == 'correct': return 1-W/V
        ix,iy = letters.index(x),letters.index(y)
        if wrongType == 'ins':
            if conf_matrix[wrongType][ix][iy] and self.vocabStr.count(' '+ y) and self.vocabStr.count(x):
                if x == '#': return (conf_matrix[wrongType][ix][iy] + 1) / self.vocabStr.count(' '+ y)
                else: return (conf_matrix[wrongType][ix][iy] + 1) / self.vocabStr.count(x)
            else: return 1/len(self.vocab)
        
        if wrongType in ['trans','del']:
            if conf_matrix[wrongType][ix][iy] and self.vocabStr.count(x+y): return (conf_matrix[wrongType][ix][iy] + 1) / self.vocabStr.count(x+ y)
            elif conf_matrix[wrongType][ix][iy]: return (conf_matrix[wrongType][ix][iy] + 1) / len(self.vocab)
            elif self.vocabStr.count(x+y): return 1/self.vocabStr.count(x+y)
            else: return 1/len(self.vocab)
        
        if wrongType == 'sub':
            if conf_matrix[wrongType][ix][iy] and self.vocabStr.count(y): return (conf_matrix[wrongType][ix][iy] + 1) / self.vocabStr.count(y)
            elif conf_matrix[wrongType][ix][iy]: return (conf_matrix[wrongType][ix][iy] + 1) / len(self.vocab)
            elif self.vocabStr.count(y): return 1/self.vocabStr.count(y)
            else: return 1/len(self.vocab)


    def editDistance(self, s1, s2):
        '''calculate the minimum edit distance getween s1& s2'''
        len1, len2 = len(s1), len(s2)
        matrix = [[i+j for j in range(len2 + 1)] for i in range(len1 + 1)]

        for row in range(len1):
            for col in range(len2):
                comp = [matrix[row+1][col]+1, matrix[row][col+1]+1]

                if s1[row] == s2[col]:
                    comp.append(matrix[row][col])
                else:
                    comp.append(matrix[row][col] + 1)

                if row > 0 and col > 0:
                    if s1[row] == s2[col-1] and s1[row-1] == s2[col]:
                        comp.append(matrix[row-1][col-1] + 1)
                
                matrix[row+1][col+1] = min(comp)
            
        return matrix[len1][len2]

    def nonWordCorrect(self, sentence):
        '''correct non word errors, return No. of errors'''
        wrong = 0
        for j in range(len(sentence)):
            word = sentence[j]
            # skip all the strange/difficult to handle words...
            if bool(re.search(r"[\d.,/'-]", word)) or word.lower() in self.vocab:
                continue

            lword = word.lower()
            candidates = self.known(self.edits1(lword))

            if len(candidates)==0:
                candidates = self.known(self.edits2(lword))
            
            maxp = -np.inf
            correct = word
            for candidate in candidates:

                # channel model + uni 86.1%     84.9%
                # p = self.ng.prob(candidate,gram='uni') + calc.log(self.pEdit(candidate, lword, self.conf_matrix)) 

                # # channel model + one-way bi 86.1%
                # words = sentence[j+1] if j==0 else sentence[j-1]
                # p = self.ng.prob(candidate,words,gram='bi') + calc.log(self.pEdit(candidate, lword, self.conf_matrix)) 

                # channel model + double-way bi 86.8%
                # words1 = sentence[j+1] if j==0 else sentence[j-1]
                # words2 = sentence[j-1] if j==len(sentence)-1 else sentence[j+1]
                # if words1==words2: p = self.ng.prob(candidate,words1,gram='bi') + calc.log(self.pEdit(candidate, lword, self.conf_matrix))
                # else: p = self.ng.prob(candidate,words1,gram='bi') + self.ng.prob(candidate,words2,gram='bi') + calc.log(self.pEdit(candidate, lword, self.conf_matrix))

                # channel model + double-way tri 76.2%
                # if j == 0: p = self.ng.prob(' '.join(sentence[:2]), ' '.join(sentence[:3]), 'tri', self.ng.lamb)
                # elif j == len(sentence)-1: p = self.ng.prob(' '.join(sentence[-3:-1]), ' '.join(sentence[-3:]), 'tri', self.ng.lamb)
                # elif j == 1: p = self.ng.prob(' '.join(sentence[:2]), ' '.join(sentence[:3]), 'tri', self.ng.lamb) + self.ng.prob(' '.join(sentence[1:3]), ' '.join(sentence[1:4]), 'tri', self.ng.lamb)
                # elif j == len(sentence)-2: p = self.ng.prob(' '.join(sentence[-3:-1]), ' '.join(sentence[-3:]), 'tri', self.ng.lamb) + self.ng.prob(' '.join(sentence[-4:-2]), ' '.join(sentence[-4:-1]), 'tri', self.ng.lamb)
                # else:
                #     p = self.ng.prob(' '.join(sentence[j-2:j]), ' '.join(sentence[j-2:j+1]), 'tri', self.ng.lamb) + \
                #          self.ng.prob(' '.join(sentence[j-1:j+1]), ' '.join(sentence[j-1:j+2]), 'tri', self.ng.lamb) + \
                #          self.ng.prob(' '.join(sentence[j:j+2]), ' '.join(sentence[j:j+3]), 'tri', self.ng.lamb)

                # p += calc.log(self.pEdit(candidate, lword, self.conf_matrix))

                # channel model + one-way tri 75.3%
                # if j in [0,1,2]: p = self.ng.prob(' '.join(sentence[:2]), ' '.join(sentence[:3]), 'tri', self.ng.lamb)
                # elif j in [len(sentence)-1,len(sentence)-2,len(sentence)-3]: p = self.ng.prob(' '.join(sentence[-3:-1]), ' '.join(sentence[-3:]), 'tri', self.ng.lamb)
                # else:
                #     p = self.ng.prob(' '.join(sentence[j-2:j]), ' '.join(sentence[j-2:j+1]), 'tri', self.ng.lamb) + calc.log(self.pEdit(candidate, lword, self.conf_matrix))
                         

                # language model uni   84%
                # words,gram = ('','uni') if j==0 else (sentence[j-1],'bi')
                # p = self.ng.prob(candidate,words, gram=gram)

                # language model bi    86.3%
                if j==0: p = self.ng.prob(sentence[j+1],candidate+''+sentence[j+1],gram='bi')
                if j==len(sentence)-1: p = self.ng.prob(candidate,sentence[j-1]+''+candidate,gram='bi')
                else: p = self.ng.prob(candidate,sentence[j-1]+''+candidate,gram='bi') + self.ng.prob(sentence[j+1],candidate+''+sentence[j+1],gram='bi')
                
                if p > maxp: maxp, correct = p, candidate
            
            if not word.islower():
                flag = 0
                for char in word: flag += int(char.isupper())
                if flag == 1: correct = correct[0].upper() + correct[1:]
                else: correct = correct.upper()
            
            sentence[j] = correct
            wrong += 1

        return wrong

    def realWordCorrect(self, sentence):
        '''correct real word errors'''
        # NOTE：尝试一个句子只有一个real word error
        # w_cand, p_cand = [''] * len(sentence), [0] * len(sentence)

        for j in range(len(sentence)):
            word = sentence[j]
            if bool(re.search(r"[\d.,/'-]", word)): continue

            lword = word.lower()
            candidates = self.known(self.edits1(lword))
            if len(candidates) == 0: candidates = self.known(self.edits2(lword))
            candidates.add(lword)

            maxp = -np.inf
            correct = lword
            for candidate in candidates:
                # TODO：调参大师，这里选择哪种是需要尝试的

                # channel model + uni 86.3%
                # p = self.ng.prob(candidate,gram='uni') + calc.log(self.pEdit(candidate, lword, self.conf_matrix)) 

                # channel model + sent prob + uni 85.1%
                # p = self.ng.sentProb(sentence, gram='uni', form='antilog') + calc.log(self.pEdit(candidate, lword, self.conf_matrix)) 

                # channel model + one-way bi 86.1%
                # words = sentence[j+1] if j==0 else sentence[j-1]
                # p = self.ng.prob(candidate,words,gram='bi') + calc.log(self.pEdit(candidate, lword, self.conf_matrix)) 

                # channel model + double-way bi 86.8%
                # words1 = sentence[j+1] if j==0 else sentence[j-1]
                # words2 = sentence[j-1] if j==len(sentence)-1 else sentence[j+1]
                # if words1==words2: p = self.ng.prob(candidate,words1,gram='bi') + calc.log(self.pEdit(candidate, lword, self.conf_matrix))
                # else: p = self.ng.prob(candidate,words1,gram='bi') + self.ng.prob(candidate,words2,gram='bi') + calc.log(self.pEdit(candidate, lword, self.conf_matrix)) 
                
                # channel model + sent prob + bi 85.1%
                # p = self.ng.sentProb(sentence, gram='bi', form='antilog') + calc.log(self.pEdit(candidate, lword, self.conf_matrix)) 
                
                # uni    84%
                # words,gram = ('','uni') if j==0 else (sentence[j-1],'bi')
                # p = self.ng.prob(candidate,words, gram=gram)

                # bi    %
                if j==0: p = self.ng.prob(sentence[j+1],candidate+''+sentence[j+1],gram='bi')
                if j==len(sentence)-1: p = self.ng.prob(candidate,sentence[j-1]+''+candidate,gram='bi')
                else: p = self.ng.prob(candidate,sentence[j-1]+''+candidate,gram='bi') + self.ng.prob(sentence[j+1],candidate+''+sentence[j+1],gram='bi')

                # bi sent   85.3%
                # p = self.ng.sentProb(sentence, gram='bi', form='antilog')

                if p > maxp: maxp, correct = p, candidate

            # NOTE：one real word error
            # p_cand[j] = -2000 if correct == lword else maxp

            if correct == lword: continue

            if word != lword:
                flag = 0
                for char in word: flag += int(char.isupper())
                if flag == 1: correct = correct[0].upper() + correct[1:]
                else: correct = correct.upper()
            sentence[j] = correct
            # if correct.lower() != lword: return None

        # NOTE：one real word error  85.1%
        #     w_cand[j] = correct
        # idx = np.argmax(p_cand)
        # sentence[idx] = w_cand[idx]
        
        
    def spellCorrect(self, osents, NosWrong):
        '''first check non-words, then real-words; No. of wrongs should be given in advance'''
        sentences = deepcopy(osents)
        for i in tqdm(range(len(sentences))):
            sentence = sentences[i]
            wrong = self.nonWordCorrect(sentence)
            if wrong < NosWrong[i]: self.realWordCorrect(sentence)
        return sentences


def genErrorDict(filename, filetype=0):
    '''generate error diction from misspelling data file
        './data/errors/spell-errors.txt'
    '''
    error_dict = dict()

    with open(filename) as fp:
        lines = ''.join(fp.readlines())

    instances = lines.split('\n')
    for instance in instances:
        toks = [el for el in instance.split(':') if el != '']
        error_dict[toks[0].strip()] = [el.split('*')[0].strip() for el in toks[1].split(',')]


    return error_dict

if __name__ == '__main__':
    start = time.time()

    # preprocessing
    print("preprocessing...")
    vocabpath = './dataset-SRILM/vocab.txt'
    with open(vocabpath,'r') as f:
        VOCAB = [re.sub('[0-9,.!“”‘’/\—\–\-\'\"]','', line.strip()) for line in f.readlines()]
    VOCAB = [word.lower() for word in VOCAB if word != '']
    
    testpath = './dataset-SRILM/testdata.txt'
    with open(testpath,'r') as f:
        sentences = f.read().split('\n')[:-1]   # to remove the last ''
    wrong_word_num = [int(row.split('\t')[1]) for row in sentences]
    sentences_orgin = [nltk.word_tokenize(row.split('\t')[2]) for row in sentences]
    # for the same word not be wrong
    V = sum([len(sentence) for sentence in sentences_orgin])
    W = sum(wrong_word_num)


    errorpath = './data/errors/spell-errors.txt'
    ed = genErrorDict(errorpath)

    # correct sentences
    lamb = 0.007
    # 0.002-86.5, 0.001-86.5, 0.005-86.9, 0.01-86.8, 0.007-86.7
    ng = nGram(True, True, False, filename='./dataset-SRILM/ans.txt', lamb=lamb)  # ACC:86.9%, Time:42s
    # ng = nGram(True, True, False, filename=None, lamb=lamb) # ACC:86.7%. Time:70s
    sc = SpellCorrect(ngram=ng,vocab=VOCAB, error_dict=ed)

    print("Start correcting...")
    sentences_correct = sc.spellCorrect(sentences_orgin, wrong_word_num)
    
    print("Time :" + str(time.time() - start))

    # write in result.txt
    filename = './dataset-SRILM/result.txt'
    with open(filename,'w') as f:
        for i, row in enumerate(sentences_correct):
            sentence = ' '.join(sentences_correct[i])
            f.write(str(i+1) + '\t' + sentence + '\n')






















