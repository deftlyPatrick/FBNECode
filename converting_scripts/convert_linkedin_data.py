import pandas as pd
import ast
from collections import defaultdict
import csv

df = pd.read_csv('./data/linkedin_data.csv')

# print(df)

tempDict = {}

for i, row in df.iterrows():
    tempDict[i] = row

counter = 0

final = {}

for k, v in tempDict.items():
    final[v[0]] = ast.literal_eval(v[1])

# print(final[1])

info = ['id', 'name', 'years of experience', 'current_job', 'current_job_company_id', 'current_job_id',
        'current_company', 'current_company_id', 'education_degree', 'education_degree_type', 'education_ids', 'skills',
        'skills_ids', 'industries', 'total_jobs_history', 'total_companies_history', 'past_jobs', 'past_jobs_ids',
        'past_companies', 'past_education_degree', 'past_education_type']

# with open('test6.csv', 'w') as f:
#     for key in final.keys():
#         for k, v in final:
#             f.write("%s, %s\n" % (key, final[key]))

for k, v in final.items():
    print("\nk : ", k)
    print("v : ", v)
    print(v.values())

with open('./data/updated_fixed_linkedin.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(info)
#
    for k, v in final.items():
       writer.writerow(v.values())