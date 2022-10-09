import torch
import torch.nn as nn
import numpy as np

import statistics
import time
import graph
from collections import defaultdict
from UV_Encoders import UV_Encoder
from UV_Aggregators import UV_Aggregator
from Folded_Encoders import Folded_Encoder
import torch.nn.functional as F
import torch.utils.data
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import argparse

import random
import matplotlib.pyplot as plt

import os
import networkx as nx
from networkx.algorithms import bipartite as bi
import itertools

"""
GraphRec: Graph Neural Networks for Social Recommendation. 
Wenqi Fan, Yao Ma, Qing Li, Yuan He, Eric Zhao, Jiliang Tang, and Dawei Yin. 
In Proceedings of the 28th International Conference on World Wide Web (WWW), 2019. Preprint[https://arxiv.org/abs/1902.07243]
If you use this code, please cite our paper:
```
@inproceedings{fan2019graph,
  title={Graph Neural Networks for Social Recommendation},
  author={Fan, Wenqi and Ma, Yao and Li, Qing and He, Yuan and Zhao, Eric and Tang, Jiliang and Yin, Dawei},
  booktitle={WWW},
  year={2019}
}
```
"""


class GraphRec(nn.Module):

    def __init__(self, enc_u, enc_v_history, r2e):
        super(GraphRec, self).__init__()
        self.enc_u = enc_u
        self.enc_v_history = enc_v_history
        self.embed_dim = enc_u.embed_dim

        self.w_ur1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_ur2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_vr1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_vr2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_uv1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.w_uv2 = nn.Linear(self.embed_dim, 16)
        self.w_uv3 = nn.Linear(16, 1)
        self.r2e = r2e
        self.bn1 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn2 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn3 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn4 = nn.BatchNorm1d(16, momentum=0.5)
        self.criterion = nn.MSELoss()

    def forward(self, nodes_u, nodes_v):
        embeds_u = self.enc_u(nodes_u)
        embeds_v = self.enc_v_history(nodes_v)

        x_u = F.relu(self.bn1(self.w_ur1(embeds_u)))
        x_u = F.dropout(x_u, training=self.training)
        x_u = self.w_ur2(x_u)
        x_v = F.relu(self.bn2(self.w_vr1(embeds_v)))
        x_v = F.dropout(x_v, training=self.training)
        x_v = self.w_vr2(x_v)

        x_uv = torch.cat((x_u, x_v), 1)
        x = F.relu(self.bn3(self.w_uv1(x_uv)))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.bn4(self.w_uv2(x)))
        x = F.dropout(x, training=self.training)
        scores = self.w_uv3(x)
        return scores.squeeze()

    def loss(self, nodes_u, nodes_v, labels_list):
        scores = self.forward(nodes_u, nodes_v)
        return self.criterion(scores, labels_list)


def train(model, device, train_loader, optimizer, epoch, best_rmse, best_mae):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        batch_nodes_u, batch_nodes_v, labels_list = data
        batch_nodes_u = batch_nodes_u.to(device)
        batch_nodes_v = batch_nodes_v.to(device)
        labels_list = labels_list.to(device)

        optimizer.zero_grad()
        loss = model.loss(batch_nodes_u, batch_nodes_v, labels_list)
        loss.backward(retain_graph=True)
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 0:
            print('Training: [%d, %5d] loss: %.3f, The best rmse/mae: %.6f / %.6f' % (
                epoch, i, running_loss / 100, best_rmse, best_mae))
            running_loss = 0.0
    return 0


def test(model, device, test_loader, total_accuracy, total_prediction, epoch):
    model.eval()
    tmp_pred = []
    target = []
    with torch.no_grad():
        counter = 0
        for test_u, test_v, tmp_target in test_loader:
            test_u, test_v, tmp_target = test_u.to(device), test_v.to(device), tmp_target.to(device)

            # val_output = model.forward(test_u, test_v)
            # val_output = torch.clamp(val_output, min=0, max=1)
            #
            #
            #
            # val_output = val_output.tolist()
            # tmp_pred.append(val_output)
            #
            # total_accuracy[test_u.tolist()[0]].append(val_output)
            # total_prediction[test_u.tolist()[0]].append(test_v.tolist()[0])
            # print("\nprediction for user: ", test_u.tolist()[0], " | ", val_output)
            # print("job predicted: ", test_v.tolist()[0])
            #
            # tmp_target = tmp_target.tolist()
            # target.append(tmp_target[0])
            # print("target: ", tmp_target)
            ###############################################################################################
            for i in range(len(test_u)):
                temp_test_u = torch.ones(len(test_v), dtype=torch.int64) * test_u[i]
                val_output = model.forward(temp_test_u, test_v)

                loss = F.mse_loss(val_output, tmp_target)

                val_output = torch.clamp(val_output, min=0, max=1)
                val_output, indices = torch.sort(val_output, descending=True)

                tmp_pred.append(val_output.tolist())

                predicted_values = test_v.tolist()
                indices = indices.tolist()

                total_accuracy[(temp_test_u.numpy()[i], loss.tolist(), epoch)].append(val_output.tolist())
                total_prediction[(temp_test_u.numpy()[i], loss.tolist(), epoch)].append(test_v.tolist())
                # print("\nloss: ", loss.tolist())
                print("\nprediction for user: ", temp_test_u.numpy()[i], " | ", tmp_pred[counter][0:10])
                print("predicted jobs: ", [predicted_values[i] for i in indices][0:10])

                target.append(list(tmp_target.data.cpu().numpy()))
                target[counter].sort(reverse=True)
                # print("target: ", target[counter][0:10], "\n")
                counter += 1
            ###############################################################################################
            # temp_test_v = None
            # previous_test_v = None
            # for i in range(len(test_u)):
            #     temp_test_u = torch.ones(len(test_v), dtype=torch.int64) * test_u[i]
            #     for j in range(len(test_v)):
            #
            #         # if counter % 2 == 0:
            #         temp_test_v = torch.ones(len(test_v), dtype=torch.int64) * test_v[j]
            #
            #         previous_test_v = test_v[j]
            #
            #         val_output = model.forward(temp_test_u, temp_test_v)
            #         val_output = torch.clamp(val_output, min=0, max=1)
            #
            #         val_output, indices = torch.sort(val_output, descending=True)
            #
            #         tmp_pred.append(val_output.tolist())
            #
            #         predicted_values = temp_test_v.tolist()
            #         indices = indices.tolist()
            #
            #         total_accuracy[temp_test_u.numpy()[i]].append(tmp_pred[counter][0])
            #         total_prediction[temp_test_u.numpy()[i]].append(temp_test_v.numpy()[j])
            #         print("\nprediction for user: ", temp_test_u.numpy()[i], " | ", tmp_pred[counter][0:10])
            #         print("predicted jobs: ", [predicted_values[i] for i in indices][0:10])
            #
            #         target.append(list(tmp_target.data.cpu().numpy()))
            #         target[counter].sort(reverse=True)
            #         print("target: ", target[counter][0:10], "\n")
            #         counter += 1
            # else:
            #     size = int(len(temp_test_v)/2)
            #     temp_test_v_a = temp_test_v[0:size] * previous_test_v
            #     temp_test_v_b = temp_test_v[size:] * test_v[j]
            #
            #     previous_test_v = test_v[j]
            #
            #     temp_test_v = temp_test_v_a.tolist() + temp_test_v_b.tolist()
            #
            #     temp_test_v = torch.tensor(temp_test_v, dtype=torch.int64)
            #
            #     val_output = model.forward(temp_test_u, temp_test_v)
            #     val_output = torch.clamp(val_output, min=0, max=1)
            #
            #     val_output, indices = torch.sort(val_output, descending=True)
            #
            #     tmp_pred.append(val_output.tolist())
            #
            #     predicted_values = temp_test_v.tolist()
            #     indices = indices.tolist()
            #
            #     print("\nprediction for user: ", temp_test_u.numpy()[i], " | ", tmp_pred[counter][0:10])
            #     print("predicted jobs: ", [predicted_values[i] for i in indices][0:10])
            #
            #     target.append(list(tmp_target.data.cpu().numpy()))
            #     target[counter].sort(reverse=True)
            #     print("target: ", target[counter][0:10], "\n")
            #     counter += 1

    tmp_pred = np.array(sum(tmp_pred, []))
    # print("\ntmp_pred: ", tmp_pred)
    target = np.array(sum(target, []))
    # print("target: ", target)
    expected_rmse = sqrt(mean_squared_error(tmp_pred, target))
    mae = mean_absolute_error(tmp_pred, target)

    # expected_rmse = sqrt(mean_squared_error([sum(tmp_pred)], [sum(target)]))
    # mae = mean_absolute_error([sum(tmp_pred)], [sum(target)])
    return expected_rmse, mae, total_accuracy, total_prediction


def save_homogenous_graph_to_file(A, datafile, index_row, index_item):
    (M, N) = A.shape  # 8371 * 8371
    csr_dict = A.__dict__
    data = csr_dict.get("data")
    indptr = csr_dict.get("indptr")
    indices = csr_dict.get("indices")
    col_index = 0
    with open(datafile, 'w') as fw:
        for row in range(M):
            for col in range(indptr[row], indptr[row + 1]):
                r = row
                c = indices[col]
                fw.write(str(index_row.get(r)) + "\t" + str(index_item.get(c)) + "\t" + str(data[col_index]) + "\n")
                col_index += 1


def calculate_centrality(G, uSet, bSet, mode='hits'):
    authority_u = {}
    authority_v = {}
    if mode == 'degree_centrality':
        a = nx.degree_centrality(G)
    else:
        # h, a = nx.hits(G)
        a = nx.pagerank(G)

    max_a_u, min_a_u, max_a_v, min_a_v = 0, 100000, 0, 100000

    for node in G.nodes():
        if node in uSet:
            if max_a_u < a[node]:
                max_a_u = a[node]
            if min_a_u > a[node]:
                min_a_u = a[node]
        if node in bSet:
            if max_a_v < a[node]:
                max_a_v = a[node]
            if min_a_v > a[node]:
                min_a_v = a[node]

    for node in G.nodes():
        if node in uSet:
            if max_a_u - min_a_u != 0:
                authority_u[node] = (float(a[node]) - min_a_u) / (max_a_u - min_a_u)
            else:
                authority_u[node] = 0
        if node in bSet:
            if max_a_v - min_a_v != 0:
                authority_v[node] = (float(a[node]) - min_a_v) / (max_a_v - min_a_v)
            else:
                authority_v[node] = 0
    return authority_u, authority_v


def get_random_walks_restart(datafile, hits_dict, percentage, maxT, minT):
    G = graph.load_edgelist(datafile, undirected=True)
    print("Folded HIN ==> number of nodes: {}".format(len(G.nodes())))
    print("walking...")
    # walks = graph.build_deepwalk_corpus_random(G, hits_dict, percentage=percentage, maxT = maxT, minT = minT, alpha=0)
    walks = graph.build_deepwalk_corpus(G, None, 5, alpha=0, rand=random.Random())
    print("walking...ok")
    return G, walks


def generate_bipartite_folded_walks(path, history_u_lists, history_v_lists, edge_list_uv, edge_list_vu):
    BiG = nx.Graph()
    node_u = history_u_lists.keys()
    node_v = history_v_lists.keys()
    node_u = sorted(node_u)
    node_v = sorted(node_v)

    BiG.add_nodes_from(node_u, bipartite=0)
    BiG.add_nodes_from(node_v, bipartite=1)
    BiG.add_weighted_edges_from(edge_list_uv + edge_list_vu)
    A = bi.biadjacency_matrix(BiG, node_u, node_v, dtype=np.float, weight='weight', format='csr')

    row_index = dict(zip(node_u, itertools.count()))  # node_u_id_original : index_new
    col_index = dict(zip(node_v, itertools.count()))  # node_v_id_original : index_new

    index_row = dict(zip(row_index.values(), row_index.keys()))  # index_new : node_u_id_original
    index_item = dict(zip(col_index.values(), col_index.keys()))

    AT = A.transpose()
    fw_u = os.path.join(path, "homogeneous_u.dat")
    fw_v = os.path.join(path, "homogeneous_v.dat")
    save_homogenous_graph_to_file(A.dot(AT), fw_u, index_row, index_row)
    save_homogenous_graph_to_file(AT.dot(A), fw_v, index_item, index_item)

    authority_u, authority_v = calculate_centrality(BiG, node_u, node_v)  # todo task

    G_u, walks_u = get_random_walks_restart(fw_u, authority_u, percentage=0.30, maxT=32, minT=1)
    G_v, walks_v = get_random_walks_restart(fw_v, authority_v, percentage=0.30, maxT=32, minT=1)

    return G_u, walks_u, G_v, walks_v


def model_prediction(model, users=5, total_users=None, data=None, user=None):

    while users > 0:
        user = random.choice(total_users)
        to_predict = user

        # total_loss = []
        #
        # tmp_pred = []
        # target = []
        #
        # best_jobs = []
        # jobs_pred = []
        # counter = 0

        data = sorted(data)
        job_output = {}
        counter = 0
        for i in data:
            job_output[counter] = i
            counter += 1

        # linkedin_reindex_job_exp = open("linkedin_reindex_exp_id.csv", "w", encoding="utf-8")
        # for k, v in job_output.items():
        #     id = str(k)
        #     job_exp = str(v)
        #     linkedin_reindex_job_exp.write(id + "," + job_exp + "\n")
        #
        # linkedin_reindex_job_exp.close()

        # linkedin_reindex_job_exp = open("linkedin_reindex_company_id.csv", "w", encoding="utf-8")
        # for k, v in job_output.items():
        #     id = str(k)
        #     job_exp = str(v)
        #     linkedin_reindex_job_exp.write(id + "," + job_exp + "\n")
        #
        # linkedin_reindex_job_exp.close()

        total_positions = torch.tensor(list(job_output.keys()))

        to_predict = torch.ones(len(total_positions), dtype=torch.int64) * to_predict

        val_output = model.forward(to_predict, total_positions)
        val_output = torch.clamp(val_output, min=0, max=1)

        val_output, indices = torch.sort(val_output, descending=True)

        print("\nprediction for user: ", to_predict.numpy()[0], " | ", val_output.tolist()[0:10])

        predicted_jobs = [total_positions.tolist()[i] for i in indices][0:10]

        output_jobs = []

        for i in predicted_jobs:
            output_jobs.append(job_output[i])

        print("predicted jobs: ", output_jobs)

        users -= 1

    # return expected_rmse, mae, total_loss, job_pred


def load(path):
    """
    history_u_lists, history_ur_lists:  user's purchased history (item set in training set), and his/her rating score (dict)
    history_v_lists, history_vr_lists:  user set (in training set) who have interacted with the item, and rating score (dict)
    train_u, train_v, train_r: training_set (user, item, rating)
    test_u, test_v, test_r: testing set (user, item, rating)
    """
    uSet_u2u = set()
    uSet_u2b = set()
    bSet_u2b = set()

    social_adj_lists = defaultdict(set)
    history_u_lists = defaultdict(list)
    history_v_lists = defaultdict(list)

    history_ur_lists = defaultdict(list)
    history_vr_lists = defaultdict(list)

    neg_neighbors_u2u = defaultdict()
    neg_neighbors_u2b = defaultdict()

    total_users = set()
    total_positions = set()

    G = nx.Graph()
    G.name = path

    for net_type in ['linkedin_data_u2s', 'linkedin_data_u2exp']:
        with open(path + net_type + ".net") as fp:
            for line in fp:
                info = line.strip().split("`t")
                if net_type == 'linkedin_data_u2s':
                    # # user_id
                    # node1 = info[1]
                    #
                    # total_users.add(int(node1))
                    #
                    # # job_history
                    # # node2 = info[2]
                    # # node2 = node2.replace("[", "")
                    # # node2 = node2.replace("]", "")
                    # # job_history = node2.split(",")
                    # # for job in job_history:
                    # #     node2 = int(job)
                    # #     G.add_edge(node1, node2, type='u2u')
                    # #     uSet_u2u.add(node1)
                    # #     uSet_u2u.add(node2)
                    #
                    # # skills
                    # node2 = info[2]
                    # node2 = node2.replace("[", "")
                    # node2 = node2.replace("]", "")
                    # skills_list = node2.split(",")
                    # if len(skills_list) > 1:
                    #     for skill in skills_list:
                    #         node2 = int(skill)
                    #         G.add_edge(node1, node2, type='u2u')
                    #         uSet_u2u.add(node1)
                    #         uSet_u2u.add(node2)
                    # else:
                    #     break
                    pass

                else:
                    # user_id
                    node1 = int(info[1])

                    total_users.add(int(node1))

                    node2 = info[2]
                    node2 = node2.replace("[", "")
                    node2 = node2.replace("]", "")
                    company_history = node2.split(",")
                    if len(company_history) > 1:
                        for company in company_history:
                            node2 = int(company)
                            rating = int(info[3])
                            total_positions.add(int(node2))
                            G.add_edge(node1, node2, type='u2b', rating=rating)
                            uSet_u2b.add(node1)
                            bSet_u2b.add(node2)
                    else:
                        break


                    # # position_id
                    # node2 = int(info[2])

                    # total_positions.add(int(node2))
                    #
                    # rating = int(info[3])
                    # G.add_edge(node1, node2, type='u2b', rating=rating)
                    # uSet_u2b.add(node1)
                    # bSet_u2b.add(node2)

                    # #all exp_ids total
                    # node2 = info[2]
                    # node2 = node2.replace("[", "")
                    # node2 = node2.replace("]", "")
                    # position_history = node2.split(",")
                    # if len(position_history) > 1:
                    #     for position in position_history:
                    #         node2 = int(position)
                    #         rating = int(info[3])
                    #         total_positions.add(int(node2))
                    #         G.add_edge(node1, node2, type='u2b', rating=rating)
                    #         uSet_u2b.add(node1)
                    #         bSet_u2b.add(node2)
                    # else:
                    #     break

    print(nx.info(G))
    print("uSet of u2u, size: " + str(len(uSet_u2u)))
    print("uSet of u2b, size: " + str(len(uSet_u2b)))
    print("bSet of u2b, size: " + str(len(bSet_u2b)))

    G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='sorted', label_attribute="name")

    node_names = nx.get_node_attributes(G, 'name')  # key-value dict {'id':'name'}
    inv_map = {v: k for k, v in node_names.items()}

    # uSet_u2u = set([inv_map.get(name) for name in uSet_u2u])
    uSet_u2b = set([inv_map.get(name) for name in uSet_u2b])
    bSet_u2b = set([inv_map.get(name) for name in bSet_u2b])

    edge_list_uv = []
    edge_list_vu = []

    for node in G:
        for edge in G[node]:
            if G[node][edge]['type'] == 'u2b':
                r = G[node][edge]['rating']
                if node in uSet_u2b and edge in bSet_u2b:
                    # print("node inner: ", node, "nbr: ", nbr)
                    history_u_lists[node].append(edge)
                    history_v_lists[edge].append(node)
                    history_ur_lists[node].append(r)
                    history_vr_lists[edge].append(r)
                    edge_list_uv.append((node, edge, r))
                    edge_list_vu.append((edge, node, r))
                if edge in uSet_u2b and node in bSet_u2b:
                    # print("node outer: ", node, "nbr: ", nbr)
                    history_u_lists[edge].append(node)
                    history_v_lists[node].append(edge)
                    history_ur_lists[edge].append(r)
                    history_vr_lists[node].append(r)
                    edge_list_uv.append((edge, node, r))
                    edge_list_vu.append((node, edge, r))

    # print "Len: social_adj_lists: ", len(social_adj_lists)
    print("history_u_lists: ", len(history_u_lists))
    print("history_v_lists: ", len(history_v_lists))

    # create bipartite graph for random walk
    G_u, walks_u, G_v, walks_v = generate_bipartite_folded_walks(path, history_u_lists, history_v_lists, edge_list_uv,
                                                                 edge_list_vu)

    # data split
    data = []
    for (u, v) in sorted(G.edges()):
        # print(u, v)
        if G[u][v]['type'] == 'u2b':
            # r = G[u][v]['rating'] - 1
            r = G[u][v]['rating']
            if u in uSet_u2b:
                # print("u inner: ", u, v, r)
                data.append((u, v, r))
            else:
                # print("u outer: ", u, v, r)
                data.append((v, u, r))

    size = len(data)
    # train_data = data[:int(0.8 * size)]  # 35704
    # test_data = data[int(0.8 * size):]  # 8927

    random.shuffle(data)
    train_data = data[:int(0.75 * size)]  #
    test_data = data[int(0.75 * size):]  #

    train_data = sorted(train_data)
    test_data = sorted(test_data)

    train_u, train_v, train_r, test_u, test_v, test_r = [], [], [], [], [], []
    for u, v, r in train_data:
        train_u.append(u)
        train_v.append(v)
        train_r.append(r)

    for u, v, r in test_data:
        test_u.append(u)
        test_v.append(v)
        test_r.append(r)

    ratings_list = [0, 1]
    return history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, walks_u, walks_v, train_u, train_v, train_r, test_u, test_v, test_r, social_adj_lists, ratings_list, total_users, total_positions

    ## ------------------------------reindexed users and items respectively------------------------
    ##reindex
    _social_adj_lists = defaultdict(set)
    _history_u_lists = defaultdict(list)
    _history_v_lists = defaultdict(list)

    _history_ur_lists = defaultdict(list)
    _history_vr_lists = defaultdict(list)
    _train_u, _train_v, _train_r, _test_u, _test_v, _test_r = [], [], [], [], [], []

    user_id_dic = {v: k for k, v in dict(enumerate(history_u_lists.keys())).items()}
    item_id_dic = {v: k for k, v in dict(enumerate(history_v_lists.keys())).items()}

    for u in history_u_lists:
        _history_u_lists[user_id_dic[u]] = [item_id_dic[v] for v in history_u_lists[u]]

    for v in history_v_lists:
        _history_v_lists[item_id_dic[v]] = [user_id_dic[u] for u in history_v_lists[v]]

    for u in history_ur_lists:
        _history_ur_lists[user_id_dic[u]] = history_ur_lists[u]

    for v in history_vr_lists:
        _history_vr_lists[item_id_dic[v]] = history_vr_lists[v]

    tempList = []

    for u in social_adj_lists:

        for us in social_adj_lists[u]:
            if us in user_id_dic:
                tempList.append(user_id_dic[us])

        if u in user_id_dic:
            _social_adj_lists[user_id_dic[u]] = tempList

    for u, v, r in train_data:
        # print(u, v, r)
        _train_u.append(user_id_dic[u])
        _train_v.append(item_id_dic[v])
        _train_r.append(r)

    for u, v, r in test_data:
        # print(u, v, r)
        _test_u.append(user_id_dic[u])
        _test_v.append(item_id_dic[v])
        _test_r.append(r)

    # re-index walks_u and walks_v
    _walks_u = defaultdict(list)
    _walks_v = defaultdict(list)
    for u in walks_u:
        _walks_u[user_id_dic[u]] = [user_id_dic[us] for us in walks_u[u]]
    for v in walks_v:
        _walks_v[item_id_dic[v]] = [item_id_dic[vs] for vs in walks_v[v]]

    return _history_u_lists, _history_ur_lists, _history_v_lists, _history_vr_lists, _walks_u, _walks_v, _train_u, \
           _train_v, _train_r, _test_u, _test_v, _test_r, _social_adj_lists, ratings_list, total_users, total_positions


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Social Recommendation: GraphRec model')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size for training')
    parser.add_argument('--embed_dim', type=int, default=64, metavar='N', help='embedding size')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N', help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    use_cuda = False
    if torch.cuda.is_available():
        # use_cuda = False
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    embed_dim = args.embed_dim

    # path = '../SageIndRec/data/EDH/'
    path = os.getcwd() + "\\"

    history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, walks_u, walks_v, train_u, train_v, train_r, test_u, test_v, test_r, social_adj_lists, ratings_list, total_users, total_positions = load(
        path)

    total_u = train_u + test_u
    total_v = train_v + test_v
    total_r = train_r + test_r

    trainset = torch.utils.data.TensorDataset(torch.LongTensor(train_u), torch.LongTensor(train_v),
                                              torch.FloatTensor(train_r))
    testset = torch.utils.data.TensorDataset(torch.LongTensor(test_u), torch.LongTensor(test_v),
                                             torch.FloatTensor(test_r))
    totalset = torch.utils.data.TensorDataset(torch.LongTensor(total_u), torch.LongTensor(total_v),
                                              torch.FloatTensor(total_r))

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True)
    total_loader = torch.utils.data.DataLoader(totalset, batch_size=100, shuffle=True)
    # train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    # test_loader = torch.utils.data.DataLoader(testset, shuffle=True)

    num_users = 3171
    num_items = 21291
    num_ratings = 3171
    print("number of users, items, ratings: ", (num_users, num_items, num_ratings))
    u2e = nn.Embedding(num_users, embed_dim).to(device)
    v2e = nn.Embedding(num_items, embed_dim).to(device)
    r2e = nn.Embedding(num_ratings, embed_dim).to(device)
    # user feature

    # self, v2e, r2e, u2e, embed_dim, cuda="cpu", uv=True)
    agg_u_history = UV_Aggregator(v2e, r2e, u2e, embed_dim, cuda=device, uv=True)
    enc_u_history = UV_Encoder(u2e, embed_dim, history_u_lists, history_ur_lists, agg_u_history, cuda=device, uv=True)
    enc_u = Folded_Encoder(lambda nodes: enc_u_history(nodes).t(), u2e, embed_dim, 5, walks_u, base_model=enc_u_history,
                           cuda=device)

    # item feature: user * rating
    agg_v_history = UV_Aggregator(v2e, r2e, u2e, embed_dim, cuda=device, uv=False)
    enc_v_history = UV_Encoder(v2e, embed_dim, history_v_lists, history_vr_lists, agg_v_history, cuda=device, uv=False)

    enc_v = Folded_Encoder(lambda nodes: enc_v_history(nodes).t(), v2e, embed_dim, 5, walks_v, base_model=enc_v_history,
                           cuda=device)

    # model
    graphrec = GraphRec(enc_u, enc_v, r2e).to(device)
    optimizer = torch.optim.RMSprop(graphrec.parameters(), lr=args.lr, alpha=0.9)

    best_rmse = 9999.0
    best_mae = 9999.0
    endure_count = 0

    total_accuracy = defaultdict(list)
    total_predictions = defaultdict(list)

    total_loader = torch.utils.data.DataLoader(totalset, batch_size=100, shuffle=True)

    start = time.time()

    epoch_rmse = defaultdict(list)
    epoch_mae = defaultdict(list)

    for epoch in range(1, args.epochs + 1):

        train(graphrec, device, train_loader, optimizer, epoch, best_rmse, best_mae)
        expected_rmse, mae, total_accuracy, total_predictions = test(graphrec, device, test_loader, total_accuracy,
                                                                     total_predictions, epoch)
        # please add the validation set to tune the hyper-parameters based on your datasets.

        epoch_rmse[epoch].append(expected_rmse)

        epoch_mae[epoch].append(mae)

        # early stopping (no validation set in toy dataset)
        if best_rmse > expected_rmse:
            best_rmse = expected_rmse
            best_mae = mae
            endure_count = 0
        else:
            endure_count += 1
        print("rmse: %.4f, mae:%.4f " % (expected_rmse, mae))

        if endure_count > 10:
            break

    rmse_list = sorted(epoch_rmse.items())
    X_rmse, y_rmse = zip(*rmse_list)

    plt.title("RMSE vs Epoch")
    plt.plot(list(map(int, X_rmse)), y_rmse, marker='o', markerfacecolor='r')
    plt.show()

    mae_list = sorted(epoch_mae.items())
    X_mae, y_mae = zip(*mae_list)

    plt.title("MAE vs Epoch")
    plt.plot(list(map(int, X_mae)), y_mae, marker='o', markerfacecolor='r')
    plt.show()

    end = time.time()
    print("Time Elapsed: ", (end - start) / 60)

    epoch_prediction_avg = defaultdict(list)
    epoch_loss_avg = defaultdict(list)

    for key, value in total_accuracy.items():
        epoch_prediction_avg[key[2]].append(value[0])
        epoch_loss_avg[key[2]].append(key[1])

    for k, v in epoch_prediction_avg.items():
        epoch_prediction_avg[k] = statistics.mean(v[0])

    for k, v in epoch_loss_avg.items():
        epoch_loss_avg[k] = statistics.mean(v)

    prediction_list = sorted(epoch_prediction_avg.items())
    X_pred, y_pred = zip(*prediction_list)

    plt.title("Accuracy vs Epoch")
    plt.plot(list(map(int, X_pred)), y_pred, marker='o', markerfacecolor='r')
    plt.show()

    loss_list = sorted(epoch_loss_avg.items())
    X_loss, y_loss = zip(*loss_list)

    plt.title("Loss vs Epoch")
    plt.plot(list(map(int, X_loss)), y_loss, marker='o', markerfacecolor='r')
    plt.show()

    total_users = list(total_users)
    # expected_rmse, mae, total_loss, job_pred = model_prediction(graphrec, device=device, total_users=total_users, data=total_loader, user=None)
    model_prediction(graphrec, users=5, total_users=total_users, data=total_positions, user=None)

    # print(expected_rmse)
    # print(mae)
    # print(total_loss)
    # print(job_pred)

if __name__ == "__main__":
    main()