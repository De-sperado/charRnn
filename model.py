import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

class CharRnn(nn.Module):
    #模型初始化函数，参数n_letters为字符的种类数，embedding_dim为词向量的位数，需要大于所有字符数，hidden_dim为隐藏层的维数,
    def __init__(self, n_letters, embedding_dim, hidden_dim):
        super(CharRnn, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(n_letters, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim)
        self.combined_output = nn.Linear(self.hidden_dim, n_letters)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax()
    #前向传播函数，view()函数作用是将一个多行的Tensor,拼接成一行，作为LSTM的输入。之后将LSTM的输出经过relu激活函数、dropout、softmax层输出到下一个时间步的CharRnn中。
    def forward(self, input, hidden):
        length = input.size()[0]
        embeds = self.embeddings(input).view((length, 1, -1))
        output, hidden = self.lstm(embeds, hidden)
        output = F.relu(self.combined_output(output.view(length, -1)))
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden
    #初始化隐藏层向量
    def initHidden(self, length=1):
        return (Variable(torch.zeros(length, 1, self.hidden_dim)),
                Variable(torch.zeros(length, 1, self.hidden_dim)))
