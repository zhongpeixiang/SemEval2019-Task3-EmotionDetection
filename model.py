
import torch
import torch.nn as nn
import torch.nn.functional as F


class RCNN(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size, vocab_size, n_layers=1, dropout=0, bidirection=True, \
                use_deepmoji=False, use_infersent=False, use_elmo=False, use_bert_word=False, additional_hidden_size=0, \
                recurrent_dropout=0.5, kmaxpooling=1):
        super(RCNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.bidirection = True
        self.dropout = dropout
        self.hidden = None
        self.additional_hidden_size = additional_hidden_size
        self.recurrent_dropout = recurrent_dropout
        self.kmaxpooling = kmaxpooling
        self.use_elmo = use_elmo
        self.use_bert_word = use_bert_word
        
        # layers
        self.embedding = nn.Embedding(vocab_size, embed_size)
        if use_elmo:
            embed_size = 1024
        if use_bert_word:
            embed_size = 768
        self.rnn = nn.LSTM(embed_size, hidden_size, batch_first=True, num_layers=n_layers, dropout=recurrent_dropout, \
                           bidirectional=self.bidirection)
        self.W2 = nn.Linear(2*hidden_size + embed_size, hidden_size)
        hidden_size *= self.kmaxpooling
        if use_deepmoji:
            hidden_size += 2304
        if use_infersent:
            hidden_size += 4096
        
        if self.additional_hidden_size != 0:
            self.hidden = nn.Linear(hidden_size, additional_hidden_size)
            self.output = nn.Linear(additional_hidden_size, output_size)
        else:
            self.output = nn.Linear(hidden_size, output_size)
        self.dropout_layer = nn.Dropout(dropout)

    def kmax_pooling(self, x, dim, k):
        index = x.topk(k, dim = dim)[1].sort(dim = dim)[0]
        return x.gather(dim, index)
        
    def forward(self, x, config, deepmoji=None, infersent=None, elmo=None, bert_word=None, bert=None):
        """
        deepmoji: (batch_size, 2304)
        """
        if self.use_elmo == False and self.use_bert_word == False:
            word_embedding = self.embedding(x)
        elif self.use_elmo:
            word_embedding = elmo
        elif self.use_bert_word:
            word_embedding = bert_word
        output, _ = self.rnn(word_embedding)
        final_encoding = torch.cat((output, word_embedding), dim=2) # (batch_size, seq_len, 2*hidden_size + embed_size)
        if self.shortcut:
            output = final_encoding
        else:
            output = self.W2(final_encoding) # (batch_size, seq_len, hidden_size)
        # output = F.max_pool1d(output.permute(0,2,1), output.shape[1]) # (batch_size, hidden_size, 1)
        # output = output.squeeze(2)
        output = self.kmax_pooling(output, dim=1, k=config["kmaxpooling"]).view(output.shape[0], -1) # (batch_size, hidden_size*k)
        output = self.dropout_layer(output) # (batch_size, hidden_size)
        
        # add additional sentence representation
        if deepmoji is not None:
            output = torch.cat([output, deepmoji], dim=1)
        if infersent is not None:
            output = torch.cat([output, infersent], dim=1)
        if self.additional_hidden_size != 0:
            output = F.relu(self.hidden(output))
            return self.output(self.dropout_layer(output))
        return self.output(output)
        

    def init_embedding(self, embedding, config):
        # set default valeus to unk, pod and eos
        embedding[0] = config["unk"]
        embedding[1] = config["pad"]
        embedding[2] = config["eos"]
        self.embedding.weight.data.copy_(embedding)



# for sentiment classification
class CNNClassifier(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size, vocab_size, n_kernels, kernel_sizes, stride=1, padding=0, dropout=0):
        super(CNNClassifier, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.n_kernels = n_kernels
        self.kernel_sizes = kernel_sizes
        self.stride = stride
        self.dropout = dropout
        self.hidden = None
        
        # layers
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.convs = nn.ModuleList([nn.Conv2d(1, n_kernels, (kernel, embed_size), stride=stride, padding=padding) for kernel in kernel_sizes])
        self.fc1 = nn.Linear(n_kernels * len(kernel_sizes), hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout_layer = nn.Dropout(dropout)
        
    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (batch_size, n_kernels, seq_len-#)
        x = F.max_pool1d(x, x.shape[2]).squeeze(2)
        return x
        
    def forward(self, x, device):
        seq_len = x.shape[1]
        if seq_len < max(self.kernel_sizes):
            diff = max(self.kernel_sizes) - seq_len
            x = torch.concat([x, torch.zeros((x.shape[0], diff)).long().to(device)], dim=1)
        x = self.embedding(x)  # x: (batch_size, seq_len, embedding_size)
        x.unsqueeze_(dim=1)
        outputs = [self.conv_and_pool(x, conv) for conv in self.convs] # [(batch_size, num_kernels),...]
        output = torch.cat(outputs, dim=1) # (batch_size, n_kernels * len(kernel_sizes))
        output = F.relu(self.fc1(self.dropout_layer(output))) # (batch_size, hidden_size)
        return self.fc2(self.dropout_layer(output))
    
    def init_embedding(self, embedding):
        self.embedding.weight.data.copy_(embedding)
        self.embedding.weight.requires_grad = False