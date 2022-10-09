import csv
import pandas as pd
import json
import ast
import os

# Open CSV file to read CSV, note: reading and write file should be under "with"

# file = "yelp_academic_dataset_business_total_dataset_small"
file = "linkedin_data"

df = pd.read_csv("../data/updated_fixed_linkedin.csv")

print(os.path.abspath(os.getcwd()))

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


u2s = open(file + "_u2s.net", "w", encoding="utf-8")
u2pos = open(file + "_u2pos.net", "w", encoding="utf-8")
u2exp = open(file + "_u2exp.net", "w", encoding="utf-8")
u2edu = open(file + "_u2edu.net", "w", encoding="utf-8")
u2pos_history = open(file + "_u2posHistory.net", "w", encoding="utf-8")
u2company = open(file + "_u2company.net", "w", encoding="utf-8")

counter = 0

for k, v in final.items():
    # if counter != 100:
    user_id = str(v["id"])
    username = str(v["name"])
    user_position_id = str(v["current_job_id"])
    user_current_company = str(v["current_company_id"])
    user_current_job_id = str(v["current_job_company_id"])
    user_jobs = str(v["total_jobs_history"])
    user_skills = str(list(v["skills_ids"]))
    user_years_of_experience = str(v["years of experience"])
    user_education = str(v["education_ids"])
    total_jobs_history = str(list(v["total_jobs_history"]))
    past_position_ids = str(list(v["past_jobs_ids"]))
    past_companies = str(list(v["total_companies_history"]))

    if len(total_jobs_history) > 1:
        u2pos.write(str(k) + "`t" + user_id + "`t" + user_position_id + "`t" + str(1) + "\n")
        u2exp.write(str(k) + "`t" + user_id + "`t" + total_jobs_history + "`t" + str(1) + "\n")
        u2s.write(str(k) + "`t" + user_id + "`t" + user_skills + "`t" + "\n")
        u2edu.write(str(k) + "`t" + user_id + "`t" + user_education + "`t" + "\n")
        u2pos_history.write(str(k) + "`t" + user_id + "`t" + past_position_ids + "`t" + str(1) + "\n")
        u2company.write(str(k) + "`t" + user_id + "`t" + past_companies + "`t" + str(1) + "\n")
    else:
        print(user_id)
        u2pos.write(str(k) + "`t" + user_id + "`t" + user_position_id + "`t" + str(0) + "\n")
        u2exp.write(str(k) + "`t" + user_id + "`t" + total_jobs_history + "`t" + str(0) + "\n")
        u2s.write(str(k) + "`t" + user_id + "`t" + user_skills + "`t" + "\n")
        u2edu.write(str(k) + "`t" + user_id + "`t" + user_education + "`t" + "\n")
        u2pos_history.write(str(k) + "`t" + user_id + "`t" + past_position_ids + "`t" + str(0) + "\n")
        u2company.write(str(k) + "`t" + user_id + "`t" + past_companies + "`t" + str(0) + "\n")
        # counter += 1

u2exp.close()
u2s.close()
u2edu.close()