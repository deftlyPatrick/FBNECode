import csv
import pandas as pd
import json
import ast

# Open CSV file to read CSV, note: reading and write file should be under "with"

# file = "yelp_academic_dataset_business_total_dataset_small"
file = "linkedin_data"

df = pd.read_csv("../../PycharmProjects/FBNE/linkedin_data.csv")

# print(json.dumps(df, indent=4))

tempDict = {}

for i, row in df.iterrows():
    tempDict[i] = row

counter = 0

final = {}

for k, v in tempDict.items():
    final[v[0]] = ast.literal_eval(v[1])

# for k, v in final.items():
#     print(v)


# user-skills - u2s
# user-experience - u2exp
# user-education - u2edu

# 0 = id

user_to_exp = {}

counter = 0

for k, v in final.items():
    # if counter != 100:
    user_id = v["id"]
    username = str(v["name"])
    user_position_id = str(v["current_job_id"])
    user_current_company = str(v["current_company_id"])
    user_current_job_id = str(v["current_job_company_id"])
    user_jobs = str(v["total_jobs_history"])
    user_skills = str(list(v["skills_ids"]))
    user_years_of_experience = str(v["years of experience"])
    user_education = str(v["education_ids"])
    total_jobs_history = (list(v["total_jobs_history"]))
    past_position_ids = str(list(v["past_jobs_ids"]))
    past_companies = str(list(v["total_companies_history"]))

    # if len(total_jobs_history) > 1:
    #     user_to_exp[user_id] = (total_jobs_history, 1)
    # else:
    #     user_to_exp[user_id] = (total_jobs_history, 0)

    if len(total_jobs_history) > 1:
        user_to_exp[user_id] = (user_position_id, user_current_company, 1)
    else:
        user_to_exp[user_id] = (user_position_id, user_current_company, 0)


# counter += 1


df = pd.DataFrame.from_dict(user_to_exp, orient="index")

df.to_csv("data.csv")
