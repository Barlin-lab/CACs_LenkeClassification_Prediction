import torch.nn as nn
import torch.nn.functional as F
import torch

# class regression(nn.Module):
#   def __init__(self, input_dim=140):
#     super(regression, self).__init__()
#     self.liner1 = nn.Linear(input_dim, 300)  
#     self.drop1 = nn.Dropout(0.5)
#     self.liner2 = nn.Linear(300, 500)
#     self.drop2 = nn.Dropout(0.5)
#     self.liner3 = nn.Linear(500, 500)
#     self.drop3 = nn.Dropout(0.5)
#     self.out1 = nn.Linear(500, 34)
#     self.out2 = nn.Linear(500, 136)

#     nn.init.xavier_uniform_(self.liner1.weight)
#     nn.init.zeros_(self.liner1.bias)
#     nn.init.xavier_uniform_(self.liner2.weight)
#     nn.init.zeros_(self.liner2.bias)
#     nn.init.xavier_uniform_(self.liner3.weight)
#     nn.init.zeros_(self.liner3.bias)
#     nn.init.xavier_uniform_(self.out1.weight)
#     nn.init.xavier_uniform_(self.out2.weight)
#     nn.init.zeros_(self.out1.bias)
#     nn.init.zeros_(self.out2.bias)
    
    

#   def forward(self, x, hm):
#     bs = x.shape[0]
#     x = x.view(bs, -1)
#     hm = hm.view(bs, -1)
#     x = torch.cat((x,hm), 1)
#     z = F.relu(self.liner1(x))
#     z = self.drop1(z)
#     z = F.relu(self.liner2(z))
#     z = self.drop2(z)
#     z = F.relu(self.liner3(z))
#     feat = self.drop3(z)
#     c = self.out1(feat)  
#     c = F.sigmoid(c)
#     hm = self.out2(feat)
#     hm = F.tanh(hm)

#     return c, hm

class regression(nn.Module):
  def __init__(self, input_dim=140):
    super(regression, self).__init__()
    self.liner1 = nn.Linear(input_dim, 300)  
    self.drop1 = nn.Dropout(0.5)
    self.liner2 = nn.Linear(300, 500)
    self.drop2 = nn.Dropout(0.5)
    self.liner3 = nn.Linear(500, 500)
    self.drop3 = nn.Dropout(0.5)
    self.out1 = nn.Linear(500, 34)
    # self.out2 = nn.Linear(500, 136)

    nn.init.xavier_uniform_(self.liner1.weight)
    nn.init.zeros_(self.liner1.bias)
    nn.init.xavier_uniform_(self.liner2.weight)
    nn.init.zeros_(self.liner2.bias)
    nn.init.xavier_uniform_(self.liner3.weight)
    nn.init.zeros_(self.liner3.bias)
    nn.init.xavier_uniform_(self.out1.weight)
    # nn.init.xavier_uniform_(self.out2.weight)
    nn.init.zeros_(self.out1.bias)
    # nn.init.zeros_(self.out2.bias)
    
    

  def forward(self, x):
    bs = x.shape[0]
    x = x.view(bs, -1)
    # hm = hm.view(bs, -1)
    # x = torch.cat((x,hm), 1)
    z = F.relu(self.liner1(x))
    z = self.drop1(z)
    z = F.relu(self.liner2(z))
    z = self.drop2(z)
    z = F.relu(self.liner3(z))
    feat = self.drop3(z)
    c = self.out1(feat)  
    c = F.sigmoid(c)
    # hm = self.out2(feat)
    # hm = F.tanh(hm)

    return c