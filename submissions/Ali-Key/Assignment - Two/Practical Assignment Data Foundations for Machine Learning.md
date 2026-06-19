# Practical Assignment: Data Foundations for Machine Learning

## Somali Graduate Employability Dataset

**Student Submission — Assignment 2**
**Course:** DS-ML Bootcamp
**Due:** Saturday, June 20, 2026 — 12:00 PM (EAT)
**Author:** Ali Omar Abdi — Alikey

---

## Table of Contents

1. [Title and Collection Method](#1-title-and-collection-method)
2. [Features and Labels](#2-features-and-labels)
3. [Dataset Structure](#3-dataset-structure)
4. [Data Quality Issues](#4-data-quality-issues)
5. [Learning Type](#5-learning-type)
6. [Use Case and Data Science Lifecycle](#6-use-case-and-data-science-lifecycle)
7. [Conclusion](#7-conclusion)

---

## 1. Title and Collection Method

**Title:** Predicting Employment Outcomes Among Somali University Graduates

### What Is This Dataset About?

In Somalia, many university graduates finish their degree but cannot find a job. Others get hired quickly. The difference between them is not always obvious.

This dataset tries to answer a simple question: **what actually helps a graduate get a job?**

Is it a high GPA? Having done an internship? Speaking English well? Having digital skills?

To find out, we collected information from 110 graduates and recorded whether each person ended up employed or unemployed. Once this data is clean, we can train a machine learning model to learn the pattern and make predictions.

### How Was the Data Collected?

A paper and online questionnaire was given to recent graduates and final-year students at several universities in Mogadishu.

Each person answered questions about:

- Age and gender
- Which university they attended and what they studied
- Their GPA (grade point average)
- How well they speak English
- Whether they did an internship
- Their level of digital skills
- How many professional certificates they have
- Whether they are currently employed or unemployed

**Total rows collected: 110**

This dataset was built from scratch for this assignment. It was not downloaded from Kaggle, UCI, or any other online source.

---

## 2. Features and Labels

Every machine learning dataset is split into two things:

- **Features (X)** — the information we give to the model as input
- **Label (y)** — the answer we want the model to learn to predict

A good way to think about it: imagine you are a detective. The features are your clues — GPA, internship experience, digital skills. The label is the answer you are trying to figure out — did this person get a job or not?

### Features (X) — What We Put Into the Model

| Feature | Data Type | What It Means |
|---|---|---|
| Age | Number | How old the graduate is |
| Gender | Category | Male or Female |
| University | Category | Which university they attended |
| Degree_Program | Category | What subject they studied |
| GPA | Number | Their academic grade (out of 4.0) |
| English_Proficiency | Category | Low, Medium, or High |
| Internship_Experience | Category | Did they do an internship? Yes or No |
| Digital_Skills_Level | Category | Beginner, Intermediate, or Advanced |
| Certifications_Count | Number | How many professional certificates they earned |

### Label (y) — What We Want the Model to Predict

| Label | Data Type | Possible Values |
|---|---|---|
| Employment_Status | Category | Employed or Unemployed |

Every row in this dataset already has an Employment_Status value. This is important — it means the model has real answers to learn from.

---

## 3. Dataset Structure

| Property | Value |
|---|---|
| Total Rows (Samples) | 110 |
| Total Columns | 10 |
| Features | 9 |
| Label | 1 |

### Sample Table — First 10 Rows

| Age | Gender | University | Degree_Program | GPA | English_Proficiency | Internship_Experience | Digital_Skills_Level | Certifications_Count | Employment_Status |
|---|---|---|---|---|---|---|---|---|---|
| 23 | Male | SIMAD University | Computer Science | 3.5 | High | Yes | Advanced | 4 | Employed |
| 24 | Female | Mogadishu University | Nursing | 3.7 | Medium | Yes | Intermediate | 2 | Employed |
| 22 | Male | University of Somalia | Business Administration | 2.9 | Low | No | Beginner | 0 | Unemployed |
| 25 | Female | SIMAD University | Accounting | 3.4 | High | Yes | Intermediate | 3 | Employed |
| 23 | Male | Jamhuriya University | Information Technology | 3.1 | Medium | No | Intermediate | 1 | Unemployed |
| 26 | Female | East Africa University | Economics | 3.6 | High | Yes | Advanced | 5 | Employed |
| 21 | Male | Benadir University | Medicine | 3.8 | Medium | No | Beginner | 1 | Unemployed |
| 27 | Female | SIMAD University | Software Engineering | 3.9 | High | Yes | Advanced | 6 | Employed |
| 24 | Male | Mogadishu University | Business Administration | 2.7 | Low | No | Beginner | 0 | Unemployed |
| 22 | Female | University of Somalia | Nursing | 3.3 | Medium | Yes | Intermediate | 2 | Employed |

Already, a pattern starts to show. Graduates with high English proficiency, internship experience, and advanced digital skills tend to be employed. Graduates with low English, no internship, and beginner digital skills tend to be unemployed.

---

## 4. Data Quality Issues

Raw data collected from surveys is almost never perfect. This dataset has five types of problems that need to be fixed before we can train a model.

### Issue 1: Missing Values

Some participants skipped questions. For example, a person might have left their GPA blank or not answered the internship question. These empty cells are called missing values. A machine learning model cannot work with empty cells — it needs a number or a category in every cell.

**Fix:** For number columns like GPA, replace the empty cell with the average of all other GPA values. For category columns like English Proficiency, replace it with the most common value in that column.

### Issue 2: Typos and Inconsistent Text

People typed university names in different ways. For example:

- "SIMAD" vs "SIMAD University" vs "Simad Univ"

All three mean the same university, but the computer sees them as three completely different categories.

**Fix:** During preprocessing, clean and standardize all text. Choose one correct format and apply it to every row.

### Issue 3: Duplicate Rows

If someone submitted the survey twice, they appear in the dataset twice. Training the model on duplicate rows teaches it the same example twice, which can cause it to overfit.

**Fix:** Check for rows where all values are identical and remove the duplicates.

### Issue 4: Class Imbalance

Suppose 85 out of 110 rows say "Employed" and only 25 say "Unemployed." This is imbalanced. A model trained on this data will quickly learn that saying "Employed" is almost always right — so it stops trying to actually learn the pattern.

**Fix:** Use a technique called SMOTE (Synthetic Minority Oversampling Technique) to create extra examples of the minority class (Unemployed), or reduce the number of majority class rows.

### Issue 5: Self-Reporting Bias

Every participant filled in their own information. Some people may have written a higher GPA than they actually had, or claimed to have advanced digital skills when they are really a beginner.

**Fix:** This is the hardest problem to fix with code alone. The best solution is to verify information during collection. During modeling, cross-validation helps detect if the data is producing unreliable results.

---

## 5. Learning Type

This is a **Supervised Learning** problem, specifically a **Classification** task.

### Why Supervised Learning?

Supervised learning is used when every row in the dataset already has a label — a known correct answer. In this dataset, every graduate already has an Employment_Status of either "Employed" or "Unemployed."

This means the model has something to learn from. It looks at a row, studies the features, and checks the label. Over time, it figures out which combinations of features tend to lead to employment.

As covered in Lesson 2, supervised learning works like a student studying with an answer key. The model sees the question (features) and the correct answer (label) together, thousands of times, until it learns the pattern.

### Why Classification and Not Regression?

The label here is a **category**, not a number. Employed or Unemployed — two options.

Regression predicts numbers. For example, if we were trying to predict a graduate's monthly salary in dollars, we would use regression. But since we are predicting which group a person falls into, we use classification.

---

## 6. Use Case and Data Science Lifecycle

### Who Can Use This Model?

Once trained, a classification model built on this dataset could be used by:

- **Universities** — to spot students who may struggle to find work after graduation and offer them extra support
- **Government agencies** — to understand which skills are missing in the workforce and design training programs
- **Students** — to enter their own information and get an early estimate of their job prospects
- **NGOs and scholarship providers** — to direct funding and support toward graduates who need it most

### How This Dataset Fits the Data Science Lifecycle

Lesson 1 introduced the eight stages of the Data Science lifecycle. Here is exactly where this dataset fits:

| Stage | What Happens |
|---|---|
| 1. Problem Definition | We ask: what factors predict whether a Somali graduate gets a job? |
| 2. Data Collection | We survey 110 graduates and record their features and employment status |
| 3. Data Cleaning | We fix missing values, typos, duplicates, and imbalanced classes |
| 4. Exploratory Data Analysis | We look for patterns — does GPA matter? Does internship experience help? |
| 5. Feature Engineering | We convert text categories to numbers and scale numerical features |
| 6. Model Building | We train a classification model, such as Logistic Regression or Random Forest |
| 7. Model Evaluation | We test accuracy, precision, and recall on data the model has never seen |
| 8. Deployment | We build a tool that universities or students can use to get predictions |

---

## 7. Conclusion

This assignment produced a 110-row dataset about employment outcomes among Somali university graduates. The dataset has 9 features — covering age, gender, university, degree program, GPA, English proficiency, internship experience, digital skills, and certifications — and one label: Employment_Status.

Because every row has a known label, this is a Supervised Learning problem. Because the label is a category (Employed or Unemployed), it is a Classification task.

Before training any model, the dataset needs preprocessing: filling in missing values, fixing text inconsistencies, removing duplicate rows, and balancing the two classes. These steps are covered in Lesson 4.

A clean version of this dataset can train a model that helps predict graduate employment outcomes — and give universities, governments, and students a data-driven tool to make better decisions.

---

*Submitted for DS-ML Bootcamp — Assignment 2*
*Due: Saturday, June 20, 2026*