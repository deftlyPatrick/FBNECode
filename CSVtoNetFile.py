import csv
import pandas as pd
import json
import ast
#Open CSV file to read CSV, note: reading and write file should be under "with"

# file = "yelp_academic_dataset_business_total_dataset_small"
file = "linkedin_data"

df = pd.read_csv("linkedin_data.csv")

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


#user-skills - u2s
#user-experience - u2exp
#user-education - u2edu

#0 = id


u2s = open(file + "_u2s.net", "w", encoding="utf-8")
u2exp = open(file + "_u2exp.net", "w", encoding="utf-8")
u2edu = open(file + "_u2edu.net", "w", encoding="utf-8")

counter = 0

for k, v in final.items():
    # if counter != 1000:
    user_id = str(v["id"])
    user_current_job = str(v["current_job_id"])
    user_current_company = str(v["current_company_id"])
    user_jobs = str(v["total_jobs_history"])
    user_skills = str(v["skills_ids"])
    user_years_of_experience = str(v["years of experience"])
    user_education = str(v["education_ids"])

    # print("\n\n")
    # print(user_id)
    # print(user_current_job)
    # print(user_jobs)
    # print(user_skills)
    u2exp.write(user_id + "`t" + user_current_job + "`t" + user_years_of_experience + "`t" + user_current_company + "\n")
    u2s.write(user_id + "`t" + user_skills + "`t" + "\n")
    u2edu.write(user_id + "`t" + user_education + "`t" + "\n")
        # counter += 1

u2exp.close()
u2s.close()
u2edu.close()
#
#     #Read CSV
#     counter = 0
#     readCsv = csv.reader(csvFile)
#     for row in readCsv:
#         print(row)
#         #Get Values and manupilate in the file.write
#         if counter != 0:
#             id = row[0]
#             user_id = row[2]
#             business_id = row[3]
#             rating = row[5]
#             friends = row[7]
#
#             #Write CSV you need format it to string if the value is an int
#             u2b.write(id + "`t" + user_id + "`t" + business_id + "`t" + rating + "\n")
#             u2u.write(id + "`t" + user_id + "`t" + friends + "\n")
#         else:
#             counter += 1
#
# u2b.close()
# u2u.close()