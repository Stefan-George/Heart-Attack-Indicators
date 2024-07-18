
## HAI - Heart Attack Indicators


=========================

### Executive Summary

This project aims to identify key indicators of heart attacks using survey data from the CDC's Behavioral Risk Factor Surveillance System (BRFSS). By leveraging various data science techniques, including data loading, exploratory data analysis, pre-processing, and machine learning modeling, we strive to uncover meaningful insights that can inform public health strategies for cardiovascular disease prevention.
 
**Key Risk Factors in Heart Disease**

*Heart Attack Indicators*

Heart attacks or myocardial infarctions are a leading cause of morbidity and mortality worldwide. Understanding the indicators of a heart attack and risks associated with those factors is needed for early diagnosis, prevention and treatment. 

*Who is affected?*

There were 55.4 million deaths recorded in 2019 globally. 17.9 million died in 2019 from cardiovascular diseases 32% of global deaths and the number 1 reason for mortality in 2019. Since then, the instance of cardiovascular diseases has raised year on year by approximately 1.5 - 2 million on average from the recorded global data available. Which is due to several key indicators that have been researched and are known to increase the instance of cardiovascular disease. 

*Demographic Factors*

- Age: Higher ages are a risk factor, with higher incidence rates observed in older sexes. 

- Sex: Males generally have a higher risk of heart attacks compared to females, though post-menopausal women also face increased risk. 

*Medical History*

- Previous Cardiovascular Events: History of heart attack, angina, or stroke significantly increases the risk of subsequent heart attacks. 

- Chronic Conditions: Presence of conditions such as hypertension, diabetes, hyperlipidemia, and chronic kidney disease are strong indicators. 

- Family History: A family history of heart disease can increase an individual's risk. 

*Lifestyle Factors* 

- Smoking: Current and former smokers are at a higher risk compared to non-smokers. 

- Exercise: Lack of regular physical activity (minimum 1 hour daily) is associated with higher risk. 

- Diet: Poor dietary habits, including high intake of saturated fats and cholesterol, contribute to poor health at an increased rate of heart attacks. 

- Alcohol: Excessive alcohol consumption is a risk factor. 

*Psychological and Social Factors* 

- Stress and Depression: Chronic stress and depressive disorders are linked to increased risk of heart attacks. 

- Social: Factors such as socio-economic status, education, and access to healthcare can influence risk, as different areas have different lifestyle factors which contribute to a longer lifespan. 

*Biometric Indicators*

- BMI (Body Mass Index): Higher BMI is associated with increased risk of heart attacks. Though BMI has been disputed as an overall health measurement recently with research focused more on DEXA scans and waist to height measurements seeing more success. 

- Blood Pressure: Elevated blood pressure (hypertension) is a risk factor. 

- Cholesterol Levels: High levels of low-density lipoproteins (LDL) cholesterol and low levels of high-density lipoproteins (HDL) cholesterol are indicators of heart attacks. 

- Blood Sugar Levels: Elevated blood glucose levels are linked to increased risk, more so with patients with diabetes. 

 
**Mortality Preventative Measures**

*Cardiovascular Managment* 

Exercise, diet and social activity are but a few key areas to address if the instance of cardiovascular disease is a potential risk. Low cost and effective way of combating diseases. 

*Medicine*

Aspirin, beta blockers and ACE inhibitors are a few medications taken to combat heart disease though if you are taking them, you most likely already have a cardiovascular disease. In some countries medicine can be affordable but for most of the world, even the simplest of medicine remains out of reach.  This is almost always due to cost or country mismanagement. 

*Surgery*

If cardiovascular disease is advanced surgery may be the only option. Bypasses, ballons and transplants are few of the limited options of open-heart surgery. This is always a high risk, high cost and time consuming solution. It is good that the option for surgery is present but preferably not getting to this point would be the ideal situation. 


**Solution**

*Early Diagnosis*

Using data obtained by the CDC in America, creating a model or machine learning application that takes in a specific amount of known cardiovascular disease parameters and can predict if the user is at a high risk of a heart attack.  

*Impact of the solution*

Defining if someone is at a higher risk of cardiovascular disease / heart attacks is a data based, low cost, early diagnosis measure that can be implemented anywhere where there is an internet connection. It reduces the burden on healthcare institutions everywhere once implemented correctly. Knowing that if a patient comes into the clinic with this early diagnosis obtained through past data, the doctors will flag the patient and could then conduct further medical image analysis as soon as possible. For anyone wanting to know if they are at an increased risk, they do not have to book a doctor's appointment if they do not want to. Early diagnosis is the best way of preventing most diseases and in this instance, it is no different.  

 
**Dataset Description**

The Dataset (heart_attack_with_nans.csv) was obtained through Kaggle (https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease/data) but was compiled by the Centers for Disease Control and Prevention (CDC) in America, in 2022 (https://www.cdc.gov/brfss/annual_data/annual_2022.html). The CDC helps control the induction and spread of diseases. They also gain data on different diseases affecting the American population annually.  

As the CDC is a governmental institution the data obtained have gone through rigorous check before publication and even though mistakes can be made in the collection process, they are usually dealt with on a severity scale. The range is if the mistake is a minor one the CDC will use established data science techniques to rectify the problem or if the mistakes are severe the data will not be published. The dataset was collected though 17 minute telephone consultations in all 50 states. 40 Questions were asked, which represents the 40 columns present. This survey is conducted annually and in 2022 (the year of the dataset being used) there were 40 questions (40 columns) asked with 445,132 responses.  

 

- Each response from an individual corresponds with a single row of data. 

 

**Data Dictionary:**

`State:` The state in which the respondent resides. 

`Sex:` The gender of the respondent. 

`GeneralHealth:` Self-assessed health status of the respondent. 

`PhysicalHealthDays:` Number of days in the past month the respondent experienced poor physical health. 

`MentalHealthDays:` Number of days in the past month the respondent experienced poor mental health. 

`LastCheckupTime:` Time since the respondent's last routine checkup with a doctor. 

`PhysicalActivities:` Indicates whether the respondent engages in physical activities or exercises. 

`SleepHours:` Average number of hours of sleep the respondent gets in a 24-hour period. 

`RemovedTeeth:` Indicates if the respondent has had any teeth removed due to decay or gum disease. 

`HadHeartAttack:` Indicates if the respondent has ever had a heart attack. 

`HadAngina:` Indicates if the respondent has ever had angina or coronary heart disease. 

`HadStroke:` Indicates if the respondent has ever had a stroke. 

`HadAsthma:` Indicates if the respondent has ever been diagnosed with asthma. 

`HadSkinCancer:` Indicates if the respondent has ever been diagnosed with skin cancer. 

`HadCOPD:` Indicates if the respondent has ever been diagnosed with chronic obstructive pulmonary disease (COPD). 

`HadDepressiveDisorder:` Indicates if the respondent has ever been diagnosed with a depressive disorder. 

`HadKidneyDisease:` Indicates if the respondent has ever been diagnosed with kidney disease. 

`HadArthritis:` Indicates if the respondent has ever been diagnosed with arthritis. 

`HadDiabetes:` Indicates if the respondent has ever been diagnosed with diabetes. 

`DeafOrHardOfHearing:` Indicates if the respondent is deaf or has serious difficulty hearing. 

`BlindOrVisionDifficulty:` Indicates if the respondent is blind or has serious difficulty seeing, even when wearing glasses. 

`DifficultyConcentrating:` Indicates if the respondent has difficulty concentrating, remembering, or making decisions. 

`DifficultyWalking:` Indicates if the respondent has serious difficulty walking or climbing stairs. 

`DifficultyDressingBathing:` Indicates if the respondent has difficulty dressing or bathing. 

`DifficultyErrands:` Indicates if the respondent has difficulty doing errands alone such as visiting a doctor’s office or shopping. 

`SmokerStatus:` Indicates the respondent's smoking status. 

`ECigaretteUsage:` Indicates the respondent's e-cigarette usage. 

`ChestScan:` Indicates if the respondent has had a chest scan in the past year. 

`RaceEthnicityCategory:` The race or ethnicity category the respondent identifies with. 

`AgeCategory:` The age category the respondent belongs to. 

`HeightInMeters:` The height of the respondent in meters. 

`WeightInKilograms:` The weight of the respondent in kilograms. 

`BMI:` The body mass index (BMI) of the respondent, calculated from height and weight. 

`AlcoholDrinkers:` Indicates if the respondent consumes alcoholic beverages. 

`HIVTesting:` Indicates if the respondent has ever been tested for HIV. 

`FluVaxLast12:` Indicates if the respondent received a flu vaccine in the past 12 months. 

`PneumoVaxEver:` Indicates if the respondent has ever received a pneumococcal vaccine. 

`TetanusLast10Tdap:` Indicates if the respondent received a tetanus shot in the last 10 years and if it included the Whooping Cough vaccine(pertussis vaccine Tdap). 

`HighRiskLastYear:` Indicates if the respondent was at high risk for complications from the flu in the past year. 

`CovidPos:` Indicates if the respondent has ever tested positive for COVID-19. 



### Demo (Streamlit Demo TBD)

... Show your work:
...     Data visualisations
...     Interactive demo (e.g., stremlit app)
...     Short video of users trying out the solution


### Methodology

**Project Structure**

1. Data Loading
Objective: To import and examine the dataset from the BRFSS, ensuring it is ready for subsequent analysis.

Tasks:

Import the heart_attack_enc_m.csv dataset.
Conduct a preliminary inspection to understand the structure and types of variables.

2. Exploratory Data Analysis (EDA)
Objective: To explore and understand the data's underlying patterns, distributions, and relationships.

Tasks:

Data inspection to identify missing values and anomalies.
Descriptive statistics to summarize the central tendencies and dispersions.
Univariate analysis to examine individual variable distributions.
Bivariate analysis to explore relationships between pairs of variables using correlation matrices, cross-tabulations, and visualizations.

3. Pre-processing
Objective: To transform the dataset into a format suitable for machine learning modeling.

Tasks:

Apply binary encoding for binary categorical variables (No as 0.0, Yes as 1.0).
Apply ordinal encoding for ordinal variables, reflecting the order of severity.
Normalize and scale numerical variables.
Handle any remaining missing values using suitable imputation methods.

4. Modelling
Objective: To develop and evaluate various machine learning models to predict heart attack risk.

Tasks:

Implement and compare multiple machine learning algorithms: Logistic Regression, Decision Trees, Random Forests, k-Nearest Neighbors (k-NN), XGBoost, SMOTE Logistic Regression, and One Feature Removed Logistic Regression.
Train and validate models using training and validation datasets.
Optimize hyperparameters to enhance model performance.
Evaluate models using metrics such as accuracy, precision, recall, F1-score, and AUC-ROC.
Interpret model results to identify significant predictors of heart attacks.

4. Findings
Objective: To summarize the insights gained from the modeling phase and identify the best-performing models.

Tasks:

Compile the performance metrics of each model.
Highlight key indicators of heart attacks identified by the models.
Discuss the implications of these findings for public health strategies.
Appendix
Objective: To document any additional analyses or information that supports the main narrative of the project.

Tasks:

Include supplementary analyses that provide deeper insights or validate the main findings.
Document any additional exploratory work or feature engineering steps that were considered.


#### Dataset

[... Google Drive links to datasets and pickeled models](https://drive.google.com/drive/folders/1oelthiRJcGF24SZ-952hYRCMBk832qkX?usp=drive_link)

### Credits & References



------------------------------------------------------------------------------