#%%
# import numpy as np
import torch
import torch.nn as nn
# import torch.optim as optim
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_channel=1, filter_sizes=[4,5,6], vocab_size=0, 
                  embedding_dim=300, num_classes=2, dropout=0.5, static=0):
        '''0-static, 1-non-static, 2-multichannel'''
        super(CNN, self).__init__()
        
        if isinstance(vocab_size, int):
            self.WV = nn.Embedding(vocab_size, embedding_dim)
        else:   # vocab_size is a pre-trained embedding matrix
            weights = torch.FloatTensor(vocab_size)
            self.static = static

            if static == 0:
                self.WV = nn.Embedding.from_pretrained(weights, freeze=True)
            elif static == 1:
                self.WV = nn.Embedding.from_pretrained(weights, freeze=False)
            elif static == 2:
                self.WV = nn.Embedding.from_pretrained(weights, freeze=False)
                self.WV_static = nn.Embedding.from_pretrained(weights, freeze=True)

        self.num_filters = num_channel * len(filter_sizes)
        # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.filter_list = nn.ModuleList([nn.Conv2d(1, num_channel, (size, 
                                          embedding_dim)) for size in filter_sizes])
        self.linear = nn.Linear(self.num_filters, num_classes)
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        ''''''
        w_embedding = self.WV(x)    # x=[batch_size, sequence_length, embedding_size]
        w_embedding = w_embedding.unsqueeze(1) # add channel(=1) 

        if hasattr(self, 'static') and self.static == 2:
            w_embedding += self.WV_static(x).unsqueeze(1)

        pooled_outputs = []
        for i, conv in enumerate(self.filter_list):
            h = F.tanh(conv(w_embedding))   # => [batch_size, num_channel, sequence_length - filter_sizes[i] + 1, 1]
            mp = nn.MaxPool2d((h.size(2), 1))    # => [batch_size, num_channel, 1, 1]
            pooled =mp(h).squeeze(3).squeeze(2) # => [batch_size, num_channel]
            pooled_outputs.append(pooled)

        h_pool = torch.cat(pooled_outputs, dim=1)
        h_drop = self.dropout(h_pool)
        out = self.linear(h_drop)   # [batch_size, num_filters]
        return self.softmax(out)

#%%
class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, n_class, num_layer=2, bidirectional=1, static=0, dropout=0.5):
        super(RNN, self).__init__()
        if isinstance(vocab_size, int):
            self.WV = nn.Embedding(vocab_size, embedding_size)
        else:   # vocab_size is a pre-trained embedding matrix
            weights = torch.FloatTensor(vocab_size)
            self.static = static

            if static == 0:
                self.WV = nn.Embedding.from_pretrained(weights, freeze=True)
            elif static == 1:
                self.WV = nn.Embedding.from_pretrained(weights, freeze=False)
            elif static == 2:
                self.WV = nn.Embedding.from_pretrained(weights, freeze=False)
                self.WV_static = nn.Embedding.from_pretrained(weights, freeze=True)


        self.rnn = nn.RNN(input_size=embedding_size, hidden_size=hidden_size,
                          num_layers=num_layer, dropout=dropout, batch_first=True,
                          bidirectional=(bidirectional==2))
        self.linear = nn.Linear(hidden_size * bidirectional, n_class, bias=True)
        self.softmax = nn.Softmax(dim=1)

        self.hidden_size = hidden_size


    def forward(self, x):
        ''''''
        w_embedding = self.WV(x)    # x=[batch_size, sequence_length, embedding_size]

        if hasattr(self, 'static') and self.static == 2:
            w_embedding += self.WV_static(x)

        # input: [batch_size, sequence_length, embedding_size]
        # output: [batch_size, num_directions(=1) * n_hidden]
        # hidden: [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        output, _ = self.rnn(w_embedding)  # hidden 用0初始化
        output = self.linear(output)
        output = self.softmax(output) 
        return torch.max(output, dim=1)[0]  # TODO: 或者也可以是mean

#%%
# ################## a toy example... ###################
# if __name__ == "__main__":
#     # set params
#     num_channel = 1
#     filter_sizes = [2,2,2]
#     embedding_dim = 2
#     num_classes = 2
#     dropout = 0.5

#     sentences = ["i love you", "he loves me", "she likes baseball", "i hate you", "sorry for that", "this is awful"]
#     labels = [1,1,1,0,0,0]
#     word_list = " ".join(sentences).split()
#     word_list = list(set(word_list))
#     word_dict = {w: i for i, w in enumerate(word_list)}
#     vocab_size = len(word_dict)

#     # model
#     model = CNN(num_channel, filter_sizes, vocab_size, embedding_dim, num_classes, dropout)

#     loss = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)

#     inputs = torch.LongTensor([np.asarray([word_dict[n] for n in sen.split()]) for sen in sentences])
#     labels = torch.LongTensor(labels)

#     # training
#     # for epoch in range(1000):
#     for epoch in range(10):
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         cost = loss(outputs, labels)
#         cost.backward()
#         optimizer.step()

#         if (epoch + 1) % 100 == 0:
#             print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(cost))


#     # test
#     test_text = 'sorry hate you'
#     tests = [np.asarray([word_dict[n] for n in test_text.split()])]
#     test_batch = torch.LongTensor(tests)

#     predict = model(test_batch).data.max(1, keepdim=True)[1]
#     if predict[0][0] == 0:
#         print(test_text,"is Bad Mean...")
#     else:
#         print(test_text,"is Good Mean!!")

