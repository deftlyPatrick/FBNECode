from collections import defaultdict
import datetime

import mysql.connector
from mysql.connector import Error
import json
import time
import pandas as pd
from extra_functions import output_to_CSV
import pprint

connection = mysql.connector.connect(host='10.33.113.250',
                                     database='linkedin',
                                     user='jazevedo',
                                     password='tpKMBgPMgC4V',
                                     port='3308'
                                     )
cursor = None

if connection.is_connected:
    db_Info = connection.get_server_info()
    print("Connected to MySQL Server version ", db_Info)
    cursor = connection.cursor()
    cursor.execute("select database();")
    record = cursor.fetchone()
    print("You're connected to database: ", record)

# 0 - id
# 1 - user_url
# 2 - name
# 3 - location
# 4 - header
employee_query = "SELECT * FROM employees;"

# 0 - emp_id
# 1 - edu_id
# 2 - start_date
# 3 - end_date
# 4 - GPA
# 5 - activities
# 6 - description
employee_education_query = "SELECT * FROM employee_education"

# 0 - emp_id
# 1 - exp_id
# 2 - start_date
# 3 - end_date
# 4 - location
# 5 - description
# 6 - employment_type
employee_experience_query = "SELECT * FROM employee_experience"

# 0 - emp_id
# 1 - skill_id
employee_skill_query = "SELECT * FROM employee_skill"

# 0 - id
# 1 - position
# 2 - company_name
experiences_query = "SELECT * FROM experiences ORDER BY id"

# 0 - id
# 1 - skill
# 2 - category
skills_query = "SELECT * FROM skills"

# 0 - id
# 1 - institution
# 2 - degree
# 3 - degree_type
educations_query = "SELECT * FROM educations"

# employee_query
cursor.execute(employee_query)
employee_query_records = cursor.fetchall()

# employee_education_query
cursor.execute(employee_education_query)
employee_education_records = cursor.fetchall()

# employee_skill_query
cursor.execute(employee_skill_query)
employee_skill_records = cursor.fetchall()

# employee_experience_query
cursor.execute(employee_experience_query)
employee_experience_records = cursor.fetchall()

# experiences_query
cursor.execute(experiences_query)
experiences_records = cursor.fetchall()

# skills_query
cursor.execute(skills_query)
skills_records = cursor.fetchall()

# educations_query
cursor.execute(educations_query)
educations_records = cursor.fetchall()

total_experiences_record = {}

for row in experiences_records:
    total_experiences_record[row[0]] = [row[1], row[2]]

output_to_CSV(data=total_experiences_record, name="linkedin_total_experience", specialColumns=True,
              labels=["id", "position", "company"])

# id - name - current job - education - skills

employees = {}

start = time.time()

for row in employee_query_records:
    employees[row[0]] = {
        "id": row[0],
        "name": row[2],
        "years of experience": 0,
        "current_job": "",
        "current_job_company_id": "",
        "current_job_id": "",
        "current_company": "",
        "current_company_id": "",
        "education_degree": "",
        "education_degree_type": "",
        "education_ids": [],
        "skills": [],
        "skills_ids": [],
        "industries": set(),
        "total_jobs_history": set(),
        "total_companies_history": set(),
        "past_jobs": set(),
        "past_jobs_ids": set(),
        "past_companies": set(),
        "past_education_degree": set(),
        "past_education_type": set(),
    }

total_jobs = 0
current_id = 1

total_jobs_set = set()
total_employer_set = set()
total_recommended_companies = defaultdict(list)
total_recommended_employers = defaultdict(list)

overall_job_id = {}
overall_company_id = {}
job_counter = 1
company_counter = 1

for row in experiences_records:

    position = row[1].lower()
    company = row[2]

    # if position not in overall_job_id.values():
    #     overall_job_id[job_counter] = row[1]
    #     job_counter += 1
    #
    # if company not in overall_company_id.values():
    #     overall_company_id[company_counter] = company
    #     company_counter += 1

    # if position not in total_jobs_set:
    #     total_jobs_set.add(position)
    if row[1] not in total_jobs_set:
        overall_job_id[job_counter] = row[1]
        job_counter += 1
    if row[2] not in total_employer_set:
        total_employer_set.add(row[2])
        overall_company_id[company_counter] = company
        company_counter += 1
    total_recommended_companies[row[1]].append(row[2])
    total_recommended_employers[row[2]].append(row[1])

output_to_CSV(data=overall_company_id, name="linkedin_company_id", labels=["id", "company"])
output_to_CSV(data=overall_job_id, name="linkedin_job_id", labels=["id", "position"])

company_list_keys = list(overall_company_id.keys())
company_list_values = list(overall_company_id.values())

job_list_keys = list(overall_job_id.keys())
job_list_values = list(overall_job_id.values())
#

for row in employee_experience_records:
    # print(row)
    if current_id != row[0]:
        total_jobs = 0
        current_id = row[0]

    current_date = datetime.date.today()

    if row[0] in employees.keys():
        for subrow in experiences_records:
            if row[1] == subrow[0]:
                if total_jobs == 0:
                    employees[row[0]]["current_job_company_id"] = subrow[0]
                    employees[row[0]]["current_job"] = subrow[1]
                    employees[row[0]]["current_company"] = subrow[2]
                    employees[row[0]]["total_jobs_history"].add(subrow[0])

                    job_idx = job_list_values.index(subrow[1])
                    employees[row[0]]["current_job_id"] = job_list_keys[job_idx]

                    company_idx = company_list_values.index(subrow[2])
                    employees[row[0]]["current_company_id"] = company_list_keys[company_idx]
                    employees[row[0]]["total_companies_history"].add(company_list_keys[company_idx])

                    employees[row[0]]["years of experience"] += (current_date - row[2]).days // 365

                    total_jobs += 1
                else:
                    employees[row[0]]["total_jobs_history"].add(subrow[0])
                    employees[row[0]]["past_jobs_ids"].add(subrow[0])
                    employees[row[0]]["past_jobs"].add(subrow[1])
                    employees[row[0]]["past_companies"].add(subrow[2])

                    company_idx = company_list_values.index(subrow[2])
                    employees[row[0]]["total_companies_history"].add(company_list_keys[company_idx])

                    if row[3] == None:
                        employees[row[0]]["years of experience"] += (current_date - row[2]).days // 365
                    else:
                        employees[row[0]]["years of experience"] += (row[3] - row[2]).days // 365

                    total_jobs += 1

total_educations = 0
current_id = 1
for row in employee_education_records:
    if current_id != row[0]:
        total_educations = 0
        current_id = row[0]

    if row[0] in employees.keys():
        for subrow in educations_records:
            if row[1] == subrow[0]:
                if total_educations == 0:
                    employees[row[0]]["education_degree"] = subrow[2]
                    employees[row[0]]["education_degree_type"] = subrow[3]
                    employees[row[0]]["education_ids"].append(row[1])
                    total_educations += 1
                else:
                    employees[row[0]]["past_education_degree"].add(subrow[2])
                    employees[row[0]]["past_education_type"].add(subrow[2])
                    employees[row[0]]["education_ids"].append(row[1])

for row in employee_skill_records:
    if row[0] in employees.keys():
        for subrow in skills_records:
            if row[1] == subrow[0]:
                employees[row[0]]["skills"].append(subrow[1])
                employees[row[0]]["skills_ids"].append(int(subrow[0]))
                if subrow[2] not in employees[row[0]]["industries"]:
                    employees[row[0]]["industries"].add(subrow[2])

final_employees = {}

counter = 0
for k, v in employees.items():
    v['id'] = counter
    final_employees[counter] = v
    counter += 1


pp = pprint.PrettyPrinter(depth=4)
pp.pprint(final_employees)
# pp.pprint(employees)

# print(json.dumps(employees, indent=4))
# print(employees)

df = pd.DataFrame(final_employees.items())
# df = pd.DataFrame(employees.items())
df.to_csv('linkedin_data.csv', index=False)

end = time.time()
print("Time Elapsed: ", (end - start) / 60)
