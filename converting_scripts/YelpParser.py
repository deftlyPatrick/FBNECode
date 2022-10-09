import json
import pickle
from random import random
import numpy as np
import ijson as ijson
import time
from collections import defaultdict
from datetime import datetime
import csv
import regex as re
from pathlib import Path
import os
import pandas as pd
import random
import math
from tqdm import tqdm

class YelpParser():

        def __init__(self):

            self.file_name = ""

            self.businesses_rating_dict = defaultdict()
            self.businesses_category_dict = defaultdict()
            self.businesses_dict = defaultdict()
            self.categories_dict = defaultdict()
            self.user_dict = defaultdict()
            self.user_friends_dict = {}

            self.user_dict_rating = defaultdict()
            self.total_review_dict = defaultdict()

        def preprocess(self, files=None):
            print("\n")

            start = time.time()

            click_list = []
            trust_list = []

            u_items_list = []
            u_users_list = []
            u_users_items_list = []
            i_users_list = []

            df_total = pd.read_csv(files[0],
                                   names=['id', 'user_id', 'business_id', 'category_id', 'stars', 'helpfulness', 'review_id'])

            df_user_business = pd.read_csv(files[1],
                                           names=['user_id', 'business_id'])

            df_business = pd.read_csv(files[2], names=['id', 'business_id'])

            df_business_rating = pd.read_csv(files[3], names=['business_id', 'rating'])

            df_user = pd.read_csv(files[4], names=['id', 'user_id'])

            df_user_dict = dict(zip(list(df_user.id), list(df_user.user_id)))
            del df_user_dict['id']

            df_user_dict = {str(key): int(value) for key, value in df_user_dict.items()}

            # for key in df_user_dict.items():
            #     key = str(key)
            #
            # temp = list(df_user_dict.items())[:262146]
            # #
            # print(temp)


            user_count = int(df_user.iloc[-1]['id'])

            item_count = int(df_business.iloc[-1]['id'])

            df_business_rating = df_business_rating.sort_values('rating')

            rate_count = float(df_business_rating.iloc[-2]['rating'])

            for i, row in tqdm(df_total.iterrows()):
                if not math.isnan(i):
                    uid = int(row['id'])
                    iid = int(row['business_id'])
                    label = int(float(row['stars']))

                    click_list.append([uid, iid, label])


            pos_list = []
            for i in range(len(click_list)):
                pos_list.append((click_list[i][0], click_list[i][1], click_list[i][2]))

            # remove duplicate items in pos_list because there are some cases where a user may have different rate scores on the same item.
            pos_list = list(set(pos_list))

            # train, valid and test data split
            random.shuffle(pos_list)
            num_test = int(len(pos_list) * 0.1)
            test_set = pos_list[:num_test]
            valid_set = pos_list[num_test:2 * num_test]
            train_set = pos_list[2 * num_test:]
            print('Train samples: {}, Valid samples: {}, Test samples: {}'.format(len(train_set), len(valid_set),
                                                                                  len(test_set)))

            # with open(workdir + args.dataset + '/dataset.pkl', 'wb') as f:
            #     pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
            #     pickle.dump(valid_set, f, pickle.HIGHEST_PROTOCOL)
            #     pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)

            train_df = pd.DataFrame(train_set, columns=['uid', 'iid', 'label'])
            valid_df = pd.DataFrame(valid_set, columns=['uid', 'iid', 'label'])
            test_df = pd.DataFrame(test_set, columns=['uid', 'iid', 'label'])

            with open('dataset.pkl', 'wb') as f:
                pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(valid_set, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)

            print(train_df)
            print(valid_df)
            print(test_df)

            click_df = pd.DataFrame(click_list, columns=['uid', 'iid', 'label'])
            train_df = train_df.sort_values(axis=0, ascending=True, by='uid')


            print("inside u_items_list")
            """
            u_items_list: 存储每个用户交互过的物品iid和对应的评分，没有则为[(0, 0)]
            """

            print("user_count: ",user_count)

            key_id = list(df_user_dict.keys())
            val_user_id = list(df_user_dict.values())



            for u in tqdm(range(int(user_count) + 1)):
                # position = key_id.index(str(u))
                hist = train_df[train_df['uid'] == u]
                u_items = hist['iid'].tolist()
                u_ratings = hist['label'].tolist()
                if u_items == []:
                    u_items_list.append([(0, 0)])
                else:
                    u_items_list.append([(iid, rating) for iid, rating in zip(u_items, u_ratings)])

            train_df = train_df.sort_values(axis=0, ascending=True, by='iid')

            # print(train_df)
            print("inside i_users_list")
            """
            i_users_list: 存储与每个物品相关联的用户及其评分，没有则为[(0, 0)]
            """
            for u in tqdm(range(item_count + 1)):
                # position = key_id.index(str(u))
                hist = train_df[train_df['uid'] == u]
                i_users = hist['uid'].tolist()
                i_ratings = hist['label'].tolist()
                if i_users == []:
                    i_users_list.append([(0, 0)])
                else:
                    i_users_list.append([(uid, rating) for uid, rating in zip(i_users, i_ratings)])

            for i, row in df_user_business.iterrows():
                if not math.isnan(i):
                    uid = int(row['user_id'])
                    fid = int(row['business_id'])

                    trust_list.append([uid, fid])

            trust_df = pd.DataFrame(trust_list, columns=['uid', 'fid'])
            trust_df = trust_df.sort_values(axis=0, ascending=True, by='uid')

            print("user_count+1")
            """
            u_users_list: 存储每个用户互动过的用户uid；
            u_users_items_list: 存储用户每个朋友的物品iid列表
            """
            for u in tqdm(range(user_count + 1)):
                hist = trust_df[trust_df['uid'] == u]
                u_users = hist['fid'].unique().tolist()
                if u_users == []:
                    u_users_list.append([0])
                    u_users_items_list.append([[(0, 0)]])
                else:
                    u_users_list.append(u_users)
                    uu_items = []
                    for uid in u_users:
                        position = val_user_id.index(uid)
                        uu_items.append(u_items_list[position])
                    u_users_items_list.append(uu_items)

            print(len(u_items_list))
            print(u_items_list, "\n")

            print(len(u_users_list))
            print(u_users_list, "\n")

            print(len(u_users_items_list))
            print(u_users_items_list, "\n")

            print(len(i_users_list))
            print(i_users_list, "\n")

            print(user_count)

            with open('list.pkl', 'wb') as f:
                pickle.dump(u_items_list, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(u_users_list, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(u_users_items_list, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(i_users_list, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump((user_count, item_count, rate_count), f, pickle.HIGHEST_PROTOCOL)

            end = time.time()
            print("Time Elapsed: ", (end - start) / 60)


        def parseReviewToCSV(self, file_name):
            # print("\n")

            start = time.time()

            counter_2 = 0

            file = file_name + ".json"

            #convert user_id to number
            #convert business_id to number
            #check if user_id exists in yelp_academic_dataset_user.csv
            #check if business_id exists in yelp_academic_dataset_business_rating.csv

            self.user_dict = defaultdict()
            self.businesses_dict = defaultdict()

            with open(file, 'r', encoding='utf-8') as reviews:
                counter = 0
                for line in tqdm(reviews):
                    if counter_2 != 1000:
                        data = json.loads(line)

                        tempReviewID = self.convertLettersToNumbers(data["review_id"])
                        tempUserID = self.convertLettersToNumbers(data["user_id"])
                        tempBusinessID = self.convertLettersToNumbers(data["business_id"])

                        if tempUserID in self.user_dict_rating.keys() and tempUserID in self.user_friends_dict.keys():
                            if tempBusinessID in self.businesses_rating_dict.keys():
                                helpfulness = (self.businesses_rating_dict[tempBusinessID] + data["stars"])/2
                                self.total_review_dict[counter] = [counter, tempUserID, tempBusinessID, self.businesses_category_dict[tempBusinessID], data["stars"], helpfulness, self.user_friends_dict[tempUserID], tempReviewID]
                                self.user_dict[counter] = tempUserID
                                self.businesses_dict[counter] = tempBusinessID
                                counter += 1

                        else:
                            counter_2 += 1
                    else:
                        pass

            # print(self.total_review_dict)

            self.output_to_CSV(self.total_review_dict, name=self.file_name+"_total_dataset.csv", specialColumns=True, labels=["id", "user_id", "business_id", "category_id", "stars", "helpfulness", "friends", "review_id"])

            #updates business_total
            self.output_to_CSV(self.businesses_dict, "yelp_academic_dataset_business_total.csv", labels=["id", "business_id"])

            #updates user_total
            self.output_to_CSV(self.user_dict, "yelp_academic_dataset_user_total.csv", labels=["id", "user_id"])

            file = self.file_name+"_total_dataset.csv"

            df = pd.read_csv(file)
            df_user_business = pd.read_csv(file)

            # print(df)
            df.sort_values(["user_id"], axis=0, ascending=True, inplace=True)
            df = df.iloc[:, 0:]

            df_user_business.sort_values(["user_id"], axis=0, ascending=True, inplace=True)
            df_user_business = df.iloc[:, :2]

            # print(df)

            tempPath_overall = Path(file)
            tempPath_user_business = Path(self.file_name + "_user_business.csv")

            if tempPath_overall.is_file():
                os.remove(file)
                print("Deleted: ", file)

            if tempPath_user_business.is_file():
                os.remove(self.file_name + "_user_business.csv")
                print("Deleted: ", self.file_name + "_user_business.csv")

            df.to_csv(file)
            df_user_business.to_csv(self.file_name + "_user_business.csv")
            print("Created: ", file)
            print("Created: ", self.file_name + "_user_business.csv")


            end = time.time()
            print("Time Elapsed: ", (end - start) / 60)


        #converts business id to numbers and grab rating to be pushed to CSV
        def parseBusinessToCSV(self, file_name):
            print("\n")

            start = time.time()

            self.file_name = file_name

            counter = 0

            file = file_name + ".json"

            categories = set()
            businesses = set()
            category_counter = 0
            business_counter = 0

            with open(file, 'r', encoding='utf-8') as f:
                for line in tqdm(f):
                    data = json.loads(line)

                    if int(data["review_count"]) > 5 and data["is_open"] != 0 and data["categories"] is not None:
                        if counter != 5000:
                            tempBusinessID = self.convertLettersToNumbers(data["business_id"])

                            tempCategories = data["categories"].split(",")
                            tempCategories = [category.strip() for category in tempCategories]


                            if len(tempCategories) > 1:
                                for category in tempCategories:
                                    if category not in categories:
                                        categories.add(category)
                                        category_counter += 1
                                        self.categories_dict[category_counter] = category

                                if tempBusinessID not in businesses:
                                    businesses.add(tempBusinessID)
                                    business_counter += 1
                                    self.businesses_dict[business_counter] = tempBusinessID

                                self.businesses_category_dict[tempBusinessID] = tempCategories
                                self.businesses_rating_dict[tempBusinessID] = data["stars"]
                            counter += 1
                        else:
                            pass


            #converts categories names to their generated id numbers

            for k, v in self.businesses_category_dict.items():
                category_string_to_int = []
                if len(v) > 1:
                    for category in v:
                        if category in self.categories_dict.values():
                            num = list(self.categories_dict.keys())[list(self.categories_dict.values()).index(category)]
                            category_string_to_int.append(num)
                else:
                    num = list(self.categories_dict.keys())[list(self.categories_dict.values()).index(v)]
                    print(num)
                v = category_string_to_int
                self.businesses_category_dict[k] = v
                # print("k: ", k, "v:", v)


            # print(businesses_rating_dict)
            # print(self.businesses_category_dict)

            # with open(file_name + "_rating.csv", 'w') as f:
            #     writer = csv.writer(f, lineterminator='\n')
            #     for key, value in self.businesses_rating_dict.items():
            #         writer.writerow([key, value])
            #
            # with open(file_name + "_category.csv", 'w') as f:
            #     writer = csv.writer(f, lineterminator='\n')
            #     for key, value in self.businesses_category_dict.items():
            #         writer.writerow([key, value])
            #
            # with open(file_name + '_categories_index.csv', 'w') as f:
            #     writer = csv.writer(f, lineterminator='\n')
            #     for key, value in self.categories_dict.items():
            #         writer.writerow([key, value])

            self.output_to_CSV(self.businesses_rating_dict, labels=["business_id", "rating"])
            self.output_to_CSV(self.businesses_dict, name=self.file_name+"_total.csv", labels=["id", "business_id"])
            self.output_to_CSV(self.businesses_category_dict, name=self.file_name + "_category.csv", labels=["business_id", "category_ids"])
            self.output_to_CSV(self.categories_dict, name=self.file_name+"_categories_index.csv", labels=["category_id", "category_name"])

            end = time.time()
            print("Time Elapsed: ", (end - start) / 60)

        # user_id
        # yelping_since
        #converts user id to numbers and grab user first created date to be pushed to CSV
        def parseUserToCSV(self, file_name):
            print("\n")
            start = time.time()

            self.file_name = file_name

            counter = 0

            file = file_name + ".json"

            # #1900-01-01 01:01:01
            latest_date = datetime(1900, 1, 1, 1, 1, 1)

            #1048575 = 524287

            users = set()
            user_counter = 0

            with open(file, 'r', encoding='utf-8') as f:
                for line in tqdm(f):
                    data = json.loads(line)
                    if int(data["review_count"]) > 10 and len(data["name"]) > 1:
                        if counter != 5000:

                            tempUserID = self.convertLettersToNumbers(data["user_id"])

                            if tempUserID not in users:
                                self.user_dict[user_counter] = int(tempUserID)
                                user_counter += 1

                            self.user_dict_rating[tempUserID] = data["yelping_since"]

                            # tempDict[data["user_id"]] = data["yelping_since"]

                            yelp_user_date = datetime.fromisoformat(data["yelping_since"])

                            if latest_date < yelp_user_date:
                                latest_date = yelp_user_date

                            if len(data["friends"]) > 10:

                                friendList = data["friends"].split()

                                friendList = [friend.replace(',','') for friend in friendList]

                                tempFriendList = []

                                for friend_id in friendList:

                                    tempFriend = self.convertLettersToNumbers(friend_id)

                                    tempFriendList.append(int(tempFriend))

                                self.user_friends_dict[tempUserID] = list(map(int, tempFriendList))

                            counter += 1
                        else:
                            pass

            usersToBeDeleted = []
            #
            for key, val in tqdm(self.user_friends_dict.items()):
                non_matches = list(set(val) - set(self.user_dict.values()))
                existing_friends = list(set(val)-set(non_matches))


                # for friend in range(len(val)):
                #     if val[friend] in self.user_dict.values():
                #         friendsToKeep.append(val[friend])
                    # else:
                        # print("removed: ", val[friend])
                print(len(existing_friends))
                if len(existing_friends) > 4:
                    self.user_friends_dict[key] = existing_friends
                else:
                    # print("to be deleted: ", key)
                    usersToBeDeleted.append(key)

            for key in tqdm(self.user_friends_dict.copy()):
                if key in usersToBeDeleted:
                    del self.user_friends_dict[key]


            self.output_to_CSV(self.user_dict_rating, labels=["user_id", "yelping_since"])
            self.output_to_CSV(self.user_dict, name=self.file_name+"_total.csv", labels=["id", "user_id"])
            self.output_to_CSV(self.user_friends_dict, name=self.file_name+"_friends_total.csv", labels=["user_id", "friend_list"])

            end = time.time()
            print("Time Elapsed: ", (end - start) / 60)

        def convertLettersToNumbers(self, dataValue):

            strippedID = re.sub(r"[^0-9a-zA-Z]+", "", dataValue)

            ID = ""

            for value in strippedID.lower():
                if not value.isnumeric():
                    outputVal = ord(value) - 96
                    ID += str(outputVal)
                else:
                    ID += str(value)

            return ID

        def output_to_CSV(self, data, name=None, specialColumns=False, labels=None):

            if labels is None:
                labels = []
            if name is None:
                output = self.file_name + "_rating.csv"
            else:
                output = name

            tempPath = Path(output)
            if tempPath.is_file():
                os.remove(output)
                print("\nDeleted: ", output)

            with open(output, 'w') as f:
                writer = csv.writer(f, lineterminator='\n')
                if labels:
                    writer.writerow([label for label in labels])

                if not specialColumns:
                    for key, value in data.items():
                        writer.writerow([key, value])
                else:
                    for key, values in data.items():
                        counter = 0
                        for i in range(len(values)):
                            if counter == 0:
                                writer.writerow([value for value in values])
                                counter += 1
                            else:
                                continue

            print("Created: ", output)

#######################################################################################################################

a = YelpParser()
a.parseUserToCSV("yelp_academic_dataset_user")
a.parseBusinessToCSV("yelp_academic_dataset_business")
a.parseReviewToCSV("yelp_academic_dataset_review")
# a.preprocess(files=["yelp_academic_dataset_business_total_dataset.csv", "yelp_academic_dataset_business_user_business.csv", "yelp_academic_dataset_business_total.csv", "yelp_academic_dataset_business_rating.csv", "yelp_academic_dataset_user_total.csv"])

# parseUserToCSV(file_name_user)
# parseBusinessToCSV(file_name_business)
