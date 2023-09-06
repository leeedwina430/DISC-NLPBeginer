#%%
# import os
# os.chdir("../../../code/pretrain") 

#%%
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import gensim

from models_pretrain import CNN, RNN
from data_utils_pretrain import *

#%%
# read in 
filename = "../dataset/pj2/train.tsv"
dataset = tsvReader(filename)
dataset = add_clean(dataset)

# splitting the dataset into Training, Developing and Testing Data
train_data, test_data, dev_data = train_dev_test_spliter(dataset, train_propor=0.8, test=True)

# train word2vec from scratch
# word2ind, ind2word = get_vocab_token(train_data['clean_sentence'])


# # load pre-trained model
# stanford word2vec
word2vec_path = "../../pj3/data/GoogleNews-vectors-negative300.bin"
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True, limit=200000) 
word2ind, ind2word = word2vec_model.key_to_index, word2vec_model.index_to_key

# load glove pre-trained
# vocab, glove_embeddings = load_glove_model('/data/junhao/NLP/glove.6B.300d.txt')
# word2ind, ind2word = word2vec_model.key_to_index, word2vec_model.index_to_key


train_data = MyDataset(train_data, vocab_set=(word2ind, ind2word))
dev_data = MyDataset(dev_data, vocab_set=(train_data.word2ind, train_data.ind2word))
test_data = MyDataset(test_data, vocab_set=(train_data.word2ind, train_data.ind2word))

#%%

# # train word2vec from scratch
# VOCAB_SIZE = len(train_data.word2ind)
# STATIC = 0

# load pre-trained model
VOCAB_SIZE = word2vec_model.vectors
STATIC = 1

NUM_CHANNEL = 100
NUM_CLASSES = 5
EMBEDDING_DIM = 300

FILTER_SIZES = [3,4,5]
LR = 0.001
DROPOUT = 0.2
BATCH_SIZE = 1000
EPOCHS = 20

torch.manual_seed(42)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, drop_last=False)
dev_loader = DataLoader(dev_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, drop_last=False)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, drop_last=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CNN
model = CNN(NUM_CHANNEL, FILTER_SIZES, VOCAB_SIZE, EMBEDDING_DIM, NUM_CLASSES, DROPOUT, STATIC)

# RNN
# BIDIRECTIONAL = 2
# model = RNN(VOCAB_SIZE, EMBEDDING_DIM, NUM_CHANNEL, NUM_CLASSES, static=STATIC,
#              bidirectional=BIDIRECTIONAL, dropout=DROPOUT)


model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=LR)
lossfn = nn.CrossEntropyLoss().to(device)

#%%
# training
train_loss_logs, train_acc_logs = [], []
dev_loss_logs, dev_acc_logs = [], []

for epoch in range(EPOCHS):
    train_loss = []
    t_predictions, t_targets = [], []
    for x,y in train_loader:
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        cost = lossfn(outputs, y)

        t_predictions.extend(outputs.argmax(dim=1).tolist())
        t_targets.extend(y.tolist())

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        train_loss.append(cost.item())
    
    train_loss = np.mean(train_loss)   # average loss
    train_acc = accuracy_score(t_targets, t_predictions)
    # train_acc = np.mean(t_targets == t_predictions)

    dev_loss  = []
    predictions, targets = [], []
    for x,y in dev_loader:
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        cost = lossfn(outputs, y)
        dev_loss.append(cost.item())

        predictions.extend(outputs.argmax(dim=1).tolist())
        targets.extend(y.tolist())

    dev_loss = np.mean(dev_loss)
    dev_acc = accuracy_score(targets, predictions)
    # dev_acc = np.mean(targets==predictions)

    print('Epoch: {:>3}, Train Loss: {:>.3f}, Train Acc: {:>.3f}%, Dev Loss: {:>.3f}, Dev Acc: {:>.3f}%'.
            format(epoch + 1, train_loss, train_acc * 100, dev_loss, dev_acc * 100))

    train_loss_logs.append(train_loss)
    train_acc_logs.append(train_acc)
    dev_loss_logs.append(dev_loss)
    dev_acc_logs.append(dev_acc)


#%%
# # save logs
# with open('logs-CNN.txt', 'w') as f:
#     print(train_loss_logs)
#     f.write('train_loss: ' + ' '.join(map(str, train_loss_logs)))
#     print(train_acc_logs)
#     f.write('train_acc' + ' '.join(map(str, train_acc_logs)))
#     print(dev_loss_logs)
#     f.write('val_loss' + ' '.join(map(str, dev_loss_logs)))
#     print(dev_acc_logs)
#     f.write('val_acc' + ' '.join(map(str, dev_acc_logs)))


#%%
# testing
predictions, targets = [], []
for x,y in test_loader:
    x, y = x.to(device), y.to(device)
    outputs = model(x)
    predictions.extend(outputs.argmax(dim=1).tolist())
    targets.extend(y.tolist())

test_acc = accuracy_score(targets, predictions)
print('Test Acc: {:>.3f}%'.format(test_acc * 100))


