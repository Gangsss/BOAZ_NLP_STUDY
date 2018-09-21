
# coding: utf-8

# In[4]:


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import nltk
import random
import numpy as np
from collections import Counter
from konlpy.tag import Kkma
import torch.utils.data as torchdata
from torchtext.data import Field
tagger = Kkma()
flatten = lambda l: [item for sublist in l for item in sublist] # ㅣist안의 list을 합치는데 이용
random.seed(1024)


# In[5]:


print(torch.__version__)
print(nltk.__version__)


# ### CPU ENV 

# In[23]:


FloatTensor = torch.FloatTensor # 차원수 만큼의 tensor
LongTensor = torch.LongTensor # 1열
dByteTensor = torch.ByteTensor # 


# ## Data loading and Preprocessing
# 
# ### Load corpus : Gutenberg corpus

# In[22]:


nltk.download('gutenberg')
nltk.corpus.gutenberg.fileids()


# In[35]:


#The sents() function divides the text up into its sentences, where each sentence is a list of words:
#nltk.corpus.gutenberg.sents('melville-moby_dick.txt')
corpus = list(nltk.corpus.gutenberg.sents('melville-moby_dick.txt'))[:100]
#corpus = list(nltk.corpus.gutenberg.sents('melville-moby_dick.txt'))[:100]
#print(corpus)

corpus = [[word.lower() for word in sent] for sent in corpus] # lower() 소문자 변환
print(corpus[:10])


# In[40]:


print(list(flatten(corpus))[:10])


# In[48]:


word_count=Counter(flatten(corpus))
print(len(flatten(corpus)))
print(len(word_count))


# In[72]:


word_count = Counter(flatten(corpus))
print(len(word_count))
border = int(len(word_count) * 0.01)
stopwords = word_count.most_common()[:border] + list(reversed(word_count.most_common()))[:border]
stopwords


# In[69]:


stopwords[0]


# In[79]:


print(list(reversed(word_count.most_common()))[:10]) # 가장 많이 안나온 것/ 1번 나온것 많음


# In[73]:


stopwords = [s[0] for s in stopwords] 
stopwords


# In[88]:


# CStopwords를 제거
vocab = list(set(flatten(corpus)) - set(stopwords))
vocab.append('<UNK>')

print(len(set(flatten(corpus))), len(vocab))


# In[100]:


word2index = {'<UNK>' : 0} 

for vo in vocab:
    if word2index.get(vo) is None:
        word2index[vo] = len(word2index)
# print(word2index) # word : index

index2word = {v:k for k, v in word2index.items()}
#print(index2word) # index : word


# In[104]:


WINDOW_SIZE = 3
windows = flatten([list(nltk.ngrams(['<DUMMY>'] * WINDOW_SIZE + c 
              + ['<DUMMY>'] * WINDOW_SIZE, WINDOW_SIZE * 2 + 1)) for c in corpus])


# In[107]:


windows[1],windows[-1]


# In[123]:


train_data = []

for window in windows:
    for i in range(WINDOW_SIZE * 2 + 1):
        if i == WINDOW_SIZE or window[i] == '<DUMMY>': #가운데단어 or dummy면 패스
            continue
        train_data.append((window[WINDOW_SIZE], window[i]))# 가운데 단어 : dummy제외한 모든 단어(한개의 ngram안에서)
train_data[:12]


# In[119]:


def getBatch(batch_size, train_data):
    random.shuffle(train_data)
    sindex = 0 # Start Index
    eindex = batch_size # End Index
    while eindex < len(train_data):
        batch = train_data[sindex:eindex]
        sindex  = eindex
        eindex = eindex + batch_size
        yield batch
        
    if eindex >= len(train_data):
        batch = train_data[sindex:]
        yield batch
def prepare_sequence(seq, word2index):
    idxs = list(map(lambda w: word2index[w] 
            if word2index.get(w) is not None else word2index["<UNK>"], seq))
    return Variable(LongTensor(idxs))

def prepare_word(word, word2index):
    return Variable(LongTensor([word2index[word]]) 
            if word2index.get(word) is not None else LongTensor([word2index["<UNK>"]]))


# In[124]:


X_p = []
y_p = []


# In[130]:


for tr in train_data:
    X_p.append(prepare_word(tr[0], word2index).view(1, -1))
    y_p.append(prepare_word(tr[1], word2index).view(1, -1))
    
train_data = list(zip(X_p, y_p))

print(X_p[0],y_p[0],train_data[0])


# In[132]:


class Skipgram(nn.Module):
    
    def __init__(self, vocab_size, projection_dim):
        super(Skipgram,self).__init__()
        self.embedding_v = nn.Embedding(vocab_size, projection_dim)
        self.embedding_u = nn.Embedding(vocab_size, projection_dim)

        self.embedding_v.weight.data.uniform_(-1, 1) # init
        self.embedding_u.weight.data.uniform_(0, 0) # init
        #self.out = nn.Linear(projection_dim,vocab_size)
        
    def forward(self, center_words,target_words, outer_words):
        center_embeds = self.embedding_v(center_words) # B x 1 x D
        target_embeds = self.embedding_u(target_words) # B x 1 x D
        outer_embeds = self.embedding_u(outer_words) # B x V x D
        
        scores = target_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2) # Bx1xD * BxDx1 => Bx1
        norm_scores = outer_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2) # BxVxD * BxDx1 => BxV
        
        nll = -torch.mean(torch.log(torch.exp(scores)/torch.sum(torch.exp(norm_scores), 1).unsqueeze(1))) # log-softmax
        
        return nll # negative log likelihood
    
    def prediction(self, inputs):
        embeds = self.embedding_v(inputs)
        
        return embeds


# ## Train

# In[133]:


EMBEDDING_SIZE = 30
BATCH_SIZE = 256
EPOCH = 100
LEARNING_RATE = 0.01

losses = []
model = Skipgram(len(word2index), EMBEDDING_SIZE)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# In[ ]:


for epoch in range(EPOCH):
    for i, batch in enumerate(getBatch(BATCH_SIZE, train_data)):
        inputs, targets = zip(*batch)
        
        inputs = torch.cat(inputs) # Bx1
        targets = torch.cat(targets) # Bx1
        vocabs = prepare_sequence(list(vocab), word2index).expand(inputs.size(0), len(vocab)) # BxV
        loss = model(inputs, targets, vocabs)
        
        # Backpropagation
        model.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.data[0])
        
    if epoch % 10 == 0:
        print("Epoch : %d, Mean_Loss : %.02f" % (epoch, np.mean(losses)))
        losses = []


# In[ ]:


def word_similarity(target, vocab):  
    target_V = model.prediction(prepare_word(target, word2index))
    similarities = []
    
    for i in range(len(vocab)):
        if vocab[i] == target:
            continue
        
        vector = model.prediction(prepare_word(list(vocab)[i], word2index))
        cosine_sim = F.cosine_similarity(target_V, vector).data.tolist()[0]
        similarities.append([vocab[i], cosine_sim])
    # Sort by similarity (상위 10개)
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:10]


# In[ ]:


test = random.choice(list(vocab))
test


# In[ ]:


word_similarity(test, vocab)

