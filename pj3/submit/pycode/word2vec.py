#%%
import numpy as np
import random
import time

from gradcheck import gradcheck_naive
from gradcheck import sigmoid ### sigmoid function can be used in the following codes
# from sklearn.preprocessing import normalize

def softmax(x):
    """
    Compute the softmax function for each row of the input x.
    It is crucial that this function is optimized for speed because
    it will be used frequently in later code.
    You might find numpy functions np.exp, np.sum, np.reshape,
    np.max, and numpy broadcasting useful for this task. (numpy
    broadcasting documentation:
    http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
    """
    assert len(x.shape) <= 2
    # NOTE: 计算softmax：要先减去最大值，然后计算exponential，防止溢出
    y = np.exp(x - np.max(x, axis=len(x.shape) - 1, keepdims=True))
    normalization = np.sum(y, axis=len(x.shape) - 1, keepdims=True)
    return np.divide(y, normalization)


def normalizeRows(x):
    """ Row normalization function """
    # Implement a function that normalizes each row of a matrix to have unit length
    # 也就是让每一行的向量长度为1
    # NOTE: 为什么要有这个函数？or，为什么要使得每个单词的词向量长度为1？
    
    ### YOUR CODE HERE
    # method 1 use library function
    # x = normalize(x, norm='l2')

    # method 2 use numpy functions
    assert len(x.shape) <= 2
    x = x / np.sqrt(np.sum(np.power(x,2), axis=len(x.shape)-1)).reshape(-1,1) + 1e-30
    # NOTE：最后加上1e-30防止溢出
    ### END YOUR CODE
    
    return x

def test_normalize_rows():
    print("Testing normalizeRows...")
    x = normalizeRows(np.array([[3.0,4.0],[1, 2]])) 
    # the result should be [[0.6, 0.8], [0.4472, 0.8944]]
    print(x)
    assert (x.all() == np.array([[0.6, 0.8], [0.4472, 0.8944]]).all())
    print()


#%%
# # Testing
# # - predicted: (M,) numpy ndarray, center word vector 
# # - target: integer, the index of the outside word        
# # - outputVectors: (V,M) "output" vectors (as rows) for all tokens     
# # - dataset: needed for negative sampling        
# V,M,K = 20,4,10
# predicted = np.arange(M)
# target = 5
# outputVectors = np.arange(0,V * M).reshape((V,M))

# dataset = type('dummy', (), {})()   # NOTE: type('class name',father,attributes&functions) 会创建一个类
# def dummySampleTokenIdx():
#     return random.randint(0, V-1)
# dataset.sampleTokenIdx = dummySampleTokenIdx

# # cost, gradCenterVec, gradOutsideVecs = softmaxCostAndGradient(predicted, target, outputVectors, dataset)
# cost, gradCenterVec, gradOutsideVecs = negSamplingCostAndGradient(predicted, target, outputVectors, dataset, K=K)

# print("cost:\n",cost)
# print("gradCenterVec:\n",gradCenterVec)
# print("gradOutsideVecs:\n",gradOutsideVecs)

#%%
def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models """
    
    # Implement the cost and gradients for one predicted word vector  
    # and one target word vector as a building block for word2vec     
    # models, assuming the softmax prediction function and cross      
    # entropy loss.                                                   
    
    # Inputs:                                                         
    # - predicted: (1,M) numpy ndarray, predicted word vector (\hat{v} in 
    #   the written component or \hat{r} in an earlier version)
    # - target: integer, the index of the target word               
    # - outputVectors: (V,M) "output" vectors (as rows) for all tokens     
    # - dataset: needed for negative sampling, unused here.         
    
    # Outputs:                                                        
    # - cost: cross entropy cost for the softmax word prediction    
    # - gradPred: the gradient with respect to the predicted word   
    #        vector                                                
    # - grad: the gradient with respect to all the other word        
    #        vectors
    
    ### YOUR CODE HERE
    '''所以center word vecs只更新当前center, context word vecs会全部更新'''
    u_o = outputVectors[target, : ]     # the target outside word (1,M)
    v_c = predicted     # center word vector (1,M)
    u_w = outputVectors # all context/outside words vectors (V,M)

    p = softmax(np.dot(u_w,v_c))    # (V,M) dot (M,1) => (V,1)
    cost = -np.log(p[target])       # target is the true outside word we want
    gradCenterVec = -u_o + np.dot(p,u_w)    # (1,M) - (V,) dot (V,M) => (1,M)
    p[target] -= 1      # deal with the u_o
    gradOutsideVecs = np.outer(p, v_c)  # (V,) outer (1,M) => (V,M)
    ### END YOUR CODE
    
    return cost, gradCenterVec, gradOutsideVecs

def negSamplingCostAndGradient(predicted, target, outputVectors, dataset, 
    K=10):
    """ Negative sampling cost function for word2vec models """
    """ Samples K indexes which are not the outsideWordIdx """

    # Implement the cost and gradients for one predicted word vector  
    # and one target word vector as a building block for word2vec     
    # models, using the negative sampling technique. K is the sample  
    # size. You might want to use dataset.sampleTokenIdx() to sample  
    # a random word index. 
    # 
    # Note: See test_word2vec below for dataset's initialization.
    #                                       
    # Input/Output Specifications: same as softmaxCostAndGradient
    
    ### YOUR CODE HERE
    u_o = outputVectors[target, : ]     # the target outside word (1,M)
    v_c = predicted     # center word vector (M,)
    
    # TODO: 事实上不应该纯纯随机挑选欸；dataset的sample需要根据词语的频率作为挑选概率
    k_ind,_ = [],0
    while _ < K:
        new_ind = dataset.sampleTokenIdx()
        if new_ind != target:
            k_ind.append(new_ind)
            _ += 1
    

    k_ind = np.array(k_ind)
    u_k = outputVectors[k_ind]  # (K,M)

    sigmoid_o = sigmoid(np.dot(u_o, v_c))   # (1,M) dot (M,) => scaler
    sigmoid_k = sigmoid(-np.dot(u_k, v_c))  # (K,M) dot (M,) => (K,)
    cost = - np.log(sigmoid_o) - np.log(sigmoid_k).sum()

    sigmoid_o -= 1.0
    sigmoid_k = 1.0 - sigmoid_k 
    gradCenterVec = np.dot(sigmoid_o, u_o) + np.dot(sigmoid_k, u_k) # c * (1,M) + (1,K) dot (K,M) =? (1,M)

    gradOutsideVecs = np.zeros(outputVectors.shape, dtype=np.float32)
    gradOutsideVecs[target] += np.dot(sigmoid_o, v_c)    # scaler dot (M,)

    temp = np.outer(sigmoid_k, v_c)
    for i in range(len(k_ind)):
        gradOutsideVecs[k_ind[i]] += temp[i]

    # NOTE: ???为什么用下面这行会过不了gradient的check？？
    # gradOutsideVecs[k_ind] += np.outer(sigmoid_k, v_c)   # (1,K) outer (M,)
    ### END YOUR CODE
    
    return cost, gradCenterVec, gradOutsideVecs

    

#%%
def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors, 
    dataset, word2vecCostAndGradient = softmaxCostAndGradient):
    """ Skip-gram model in word2vec """

    # Implement the skip-gram model in this function.

    # Inputs:                                                         
    # - currrentWord: a string of the current center word           
    # - C: integer, context size                                    
    # - contextWords: list of no more than 2*C strings, the context words                                               
    # - tokens: a dictionary that maps words to their indices in    
    #      the word vector list     
    #                            
    # - inputVectors: "input" word vectors (as rows) for all tokens     AKA center word vecs     
    # - outputVectors: "output" word vectors (as rows) for all tokens   AKA context word vecs
    # - word2vecCostAndGradient: the cost and gradient function for 
    #      a prediction vector given the target word vectors,  
    #      could be one of the two cost functions you          
    #      implemented above

    # Outputs:                                                        
    # - cost: the cost function value for the skip-gram model       
    # - grad: the gradient with respect to the word vectors

    ### YOUR CODE HERE
    # TODO: 确保没错之后这里把链接改回去....
    centerVectors = inputVectors
    contextVectors = outputVectors

    cost, gradIn, gradOut = 0, np.zeros(centerVectors.shape), np.zeros(contextVectors.shape)
    for word in contextWords:
        cur_cost, gradCenterVec, gradOutsideVecs = word2vecCostAndGradient(centerVectors[tokens[currentWord]], tokens[word], contextVectors, dataset)
        cost += cur_cost
        gradIn[tokens[currentWord]] += gradCenterVec
        gradOut += gradOutsideVecs

    ### END YOUR CODE
    
    return cost, gradIn, gradOut

#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################
#%%
def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C, word2vecCostAndGradient = softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N//2,:]
    outputVectors = wordVectors[N//2:,:]
    for i in range(batchsize):
        C1 = random.randint(1,C)
        centerword, context = dataset.getRandomContext(C1)
        
        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1
        
        c, gin, gout = word2vecModel(centerword, C1, context, tokens, inputVectors, outputVectors, dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N//2, :] += gin / batchsize / denom
        grad[N//2:, :] += gout / batchsize / denom
        ### we use // to let N/2  be int
    return cost, grad

def test_word2vec():
    # Interface to the dataset for negative sampling
    dataset = type('dummy', (), {})()   # NOTE: type('class name',father,attributes&functions) 会创建一个类
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], [tokens[random.randint(0,4)] \
           for i in range(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
    print("==== Gradient check for skip-gram ====")
    # gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5), dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)
    #print("\n==== Gradient check for CBOW      ====")
    #gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5), dummy_vectors)
    #gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)

    print("\n=== Results ===")
    print(skipgram("c", 3, ["a", "b", "e", "d", "b", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset))
    print(skipgram("c", 1, ["a", "b"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset, negSamplingCostAndGradient))
    #print(cbow("a", 2, ["a", "b", "c", "a"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset))
    #print(cbow("a", 2, ["a", "b", "a", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset, negSamplingCostAndGradient))

#%%
if __name__ == "__main__":
    stime = time.time()
    # test_normalize_rows()
    test_word2vec()
    print("time: \t", time.time()-stime, "s")

#%%