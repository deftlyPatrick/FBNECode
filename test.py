import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


class MF(nn.Module):
    def __init__(self, num_users, num_items, emb_size=100):
        super(MF, self).__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.item_emb = nn.Embedding(num_items, emb_size)
        # initializing our matrices with a positive number generally will yield better results
        self.user_emb.weight.data.uniform_(0, 0.5)
        self.item_emb.weight.data.uniform_(0, 0.5)
    def forward(self, u, v):
        u = self.user_emb(u)
        v = self.item_emb(v)
        return (u*v).sum(1)  # taking the dot product

def train_epocs(model, epochs=10, lr=0.01, wd=0.0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    model.train()
    for i in range(epochs):
        user_ids = torch.LongTensor(train_df.user_id.values)
        position_ids = torch.LongTensor(train_df.position_id.values)
        ratings = torch.FloatTensor(train_df.rating.values)
        y_hat = model(user_ids, position_ids)
        loss = F.mse_loss(y_hat, ratings)
        optimizer.zero_grad()  # reset gradient
        loss.backward()
        optimizer.step()
        print(loss.item())
    test(model)

def test(model):
    model.eval()
    user_ids = torch.LongTensor(test_df.user_id.values)
    position_ids = torch.LongTensor(test_df.position_id.values)
    ratings = torch.FloatTensor(test_df.rating.values)
    y_hat = model(user_ids, position_ids)
    loss = F.mse_loss(y_hat, ratings)
    print("test loss %.3f " % loss.item())


df = pd.read_csv('data.csv')

train_df, valid_df = train_test_split(df, test_size=0.2)
train_df = train_df.reset_index(drop=True)
test_df = valid_df.reset_index(drop=True)
#

model = MF(3200, 20741, emb_size=100)
train_epocs(model, epochs=20, lr=0.01)

user_id = torch.tensor([1804])
position_ids = torch.tensor(df['position_id'].tolist())
predictions = model(user_id, position_ids).tolist()

predictions = np.array(predictions)
print(predictions)

normalized_predictions = [i/max(predictions)*10 for i in predictions]
print(normalized_predictions)

normalized_predictions = np.array(normalized_predictions)

sortedIndices = normalized_predictions.argsort()

recommendations = df['position_id'][sortedIndices][:5]  # taking top 30
print(recommendations)

# num_users = []
# num_items = []
# rating = []
#
# for i in df.iterrows():
#     print(i[1][2])
#     num_users.append(i[0])
#     for position_id in i[1][1]:
#         num_items.append(position_id)
#     rating.append(i[1][2])