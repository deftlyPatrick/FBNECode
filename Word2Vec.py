import gensim.downloader as api
import pandas as pd
import numpy as np
import nltk
import ast
import numpy as np

nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
import re
import string
import random
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
from matplotlib import pyplot
from gensim.models import KeyedVectors
import re
import unicodedata
from gensim.models import word2vec


class Word2Vec:

    def __init__(self, df):
        self.df = df
        self.line = 0
        self.inputs = []
        self.model = None
        self.tfidf_vectors = []
        self.tfidf = None
        self.tfidf_list = None
        self.tfidf_list = None
        self.models = {}
        self.modelCounter = 0

    def model_diff(self, model1, model2):

        assert self.models[model1]["predict_val"] == self.models[model2]["predict_val"], "Model 1: predict_val (%s) " \
                                                                                         "does not equal to Model 2: " \
                                                                                         "predict_val (%s)" % \
                                                                                         (self.models[model1]
                                                                                          ["predict_val"],
                                                                                          self.models[model2]
                                                                                          ["predict_val"])

        keys = self.models[model1]["predict_val"],  self.models[model2]["predict_val"]

        df_all = pd.concat([self.models[model1]["recommendation"].set_index('id'), self.models[model2]
                            ["recommendation"].set_index('id')], axis='columns', keys=keys)

        print(df_all)

    def create_basic_model(self, modelName: str, vector_size=500, min_count=2, window=5, sg=1, epochs=5):
        self.converting_data()
        self.model = word2vec.Word2Vec(self.inputs, vector_size=vector_size, min_count=min_count, window=window, sg=sg) # train with CBOW algorithm
        self.model.train(self.inputs, total_examples=self.model.corpus_count, epochs=epochs)
        model_name = 'job_word_embeddings.model'
        self.model.save(model_name)
        print("Saved: %s" % model_name)

        if modelName not in self.models:
            self.models[modelName] = {
                "model_name": model_name,
                "type": "w2v",
                "vector_size": vector_size,
                "min_count": min_count,
                "window": window,
                "sg": sg,
                "epochs": epochs,
                "predict_val": None,
                "recommendation": None
            }
        else:
            for k in self.models.keys():
                if k == modelName:
                    self.models[modelName] = {
                        "model_name": model_name,
                        "type": "w2v",
                        "vector_size": vector_size,
                        "min_count": min_count,
                        "window": window,
                        "sg": sg,
                        "epochs": epochs,
                        "predict_val": None,
                        "recommendation": None
                    }

    def create_tfidf_model(self, modelName: str, ngram_range=(1,3), min_df=5):
        self.tfidf = TfidfVectorizer(analyzer='word', ngram_range=ngram_range, min_df=min_df, stop_words='english')
        self.tfidf.fit(df['current_job'])

        if modelName not in self.models:
            self.models[modelName] = {
                "type": "tfidf",
                "ngram_range": ngram_range,
                "mid_df": min_df,
                "predict_val": None,
                "recommendation": None
            }
        else:
            for k in self.models.keys():
                if k == modelName:
                    self.models[modelName] = {
                        "type": "tfidf",
                        "ngram_range": ngram_range,
                        "mid_df": min_df,
                        "predict_val": None,
                        "recommendation": None
                    }

    # Remove brackets from text
    def normalize_job_title(self, title):
        title = unicodedata.normalize('NFKC', title)
        title = re.sub(r'【.*】', '', title)
        title = re.sub(r'\[.*\]', '', title)
        title = re.sub(r'「.*」', '', title)
        title = re.sub(r'\(.*\)', '', title)
        title = re.sub(r'\<.*\>', '', title)
        title = re.sub(r'[※@◎].*$', '', title)
        title = re.sub(r"\b(?=[MDCLXVIΙ])M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})([IΙ]X|[IΙ]V|V?[IΙ]{0,3})\b\.?", '',
                       title)
        return title.lower().rstrip()

    def convert_job_posting(self, job):
        converted = []

        title_and_requirements = [self.normalize_job_title(job['current_job'])]
        skills = ast.literal_eval(job['skills'])
        for skill in skills:
            title_and_requirements.append(skill.lower())

        converted.append(title_and_requirements)
        return converted

    # convert data to inputs of Gensim's word2vec API

    def converting_data(self):
        for _, p in self.df.iterrows():
            #     print(convert_job_posting(p), "\n")
            self.inputs += w2v.convert_job_posting(p)

    def similar_words(self, title):
        return self.model.wv.most_similar(title.lower())

    # Generate the average word2vec for the each job description

    def vectors(self):
        # Creating a list for storing the vectors (description into vectors)
        global word_embeddings
        word_embeddings = []
        not_append = []
        counter = 0

        # Reading the each job description
        for line in self.inputs:
            avgword2vec = None
            count = 0
            for attrib in line:
                if attrib in self.model.wv:
                    count += 1
                    if avgword2vec is None:
                        avgword2vec = self.model.wv[attrib]
                    else:
                        avgword2vec = avgword2vec + self.model.wv[attrib]

            if avgword2vec is not None:
                avgword2vec = avgword2vec / count
                word_embeddings.append(avgword2vec)
            else:
                not_append.append(counter)
            counter += 1
        return word_embeddings

    def recommendations(self, job, modelName, printResult=False):
        array_embeddings = self.vectors()
        cosine_similarities = cosine_similarity(array_embeddings, array_embeddings)
        jobs = df[['id', 'current_job', 'skills']]
        # Reverse mapping of the index
        indices = pd.Series(df.index, index=df['current_job']).drop_duplicates()
        idx = indices[job]
        sim_scores = list(enumerate(cosine_similarities[idx]))
        for i in range(len(sim_scores)):
            sim_scores[i] = (sim_scores[i][0], np.mean(sim_scores[i][1]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:6]
        job_indices = [i[0] for i in sim_scores]
        recommend = jobs.iloc[job_indices]
        if printResult:
            print("\n\nRecommendations with basic W2V for %s" % job)
            print(recommend)
        self.models[modelName]["predict_val"] = job
        self.models[modelName]["recommendation"] = recommend

    def vectors_tf_idf(self):
        self.tfidf_list = dict(zip(self.tfidf.get_feature_names(), list(self.tfidf.idf_)))
        self.tfidf_feature = self.tfidf.get_feature_names()  # tfidf words/col-names
        try:
            for desc in self.inputs:
                # Word vectors are of zero length (Used 300 dimensions)
                sent_vec = np.zeros(500)
                # num of words with a valid vector in the book description
                weight_sum = 0
                # for each word in the book description
                for word in desc:
                    if word in self.model.wv and word in self.tfidf_feature:
                        vec = self.model.wv[word]
                        tf_idf = self.tfidf_list[word] * (desc.count(word) / len(desc))
                        #             print("vec: ", len(vec))
                        #             print("tf_idf: ", tf_idf)
                        sent_vec += (vec * tf_idf)
                        weight_sum += tf_idf
                if weight_sum != 0:
                    sent_vec /= weight_sum
                self.tfidf_vectors.append(sent_vec)
                self.line += 1
        except ValueError:
            print("Inputs is empty")

    def recommendations_tf_idf(self, job, modelName, printResult=False):
        # finding cosine similarity for the vectors

        self.vectors_tf_idf()
        cosine_similarities = cosine_similarity(self.tfidf_vectors, self.tfidf_vectors)

        # taking the title and book image link and store in new data frame called books
        jobs = df[['id', 'current_job', 'skills']]
        # Reverse mapping of the    index
        indices = pd.Series(df.index, index=df['current_job']).drop_duplicates()

        idx = indices[job]
        sim_scores = list(enumerate(cosine_similarities[idx]))
        for i in range(len(sim_scores)):
            sim_scores[i] = (sim_scores[i][0], np.mean(sim_scores[i][1]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:6]
        job_indices = [i[0] for i in sim_scores]
        recommend = jobs.iloc[job_indices]
        if printResult:
            print("\n\nRecommendations with tf-idf for %s" % job)
            print(recommend)
        self.models[modelName]["predict_val"] = job
        self.models[modelName]["recommendation"] = recommend

    def print_all_models(self):
        for k, v in self.models.items():
            print("\n")
            print("Model Name: ", k)
            print("Result: ")
            print("predict_val: ", v["predict_val"])
            print("recommendation: \n", v["recommendation"].iloc[:, 1:])


df = pd.read_csv('./data/updated_fixed_linkedin.csv')
df = df.drop([537, 1620, 1774, 1787, 1821, 1894, 2007, 2232, 2831]).reset_index(drop=True)
w2v = Word2Vec(df=df)
job_titles = df['current_job']
df['current_job'].head(20).apply(w2v.normalize_job_title)

modelOne = "w2vModel"
modelTwo = "tfidfModel"

w2v.create_basic_model(modelName=modelOne)
w2v.similar_words('software engineer')  # software engineer
w2v.recommendations("Software Engineer", modelName=modelOne)

w2v.create_tfidf_model(modelName=modelTwo)
w2v.recommendations_tf_idf("Software Engineer", modelName=modelTwo)

w2v.print_all_models()
# w2v.model_diff(model1=modelOne, model2=modelTwo)