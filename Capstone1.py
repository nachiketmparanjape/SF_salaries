import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB

""" Data Cleaning and Descriptive Analysis"""

#Load the Data
salaries_df = pd.read_csv("Salaries.csv",low_memory=False)

#Remove strings 'Not Provided' by replace them by np.nan in a pd.Series
salaries_df = salaries_df.replace(to_replace='Not Provided',value=np.nan)

#Set the right data types
salaries_df['BasePay'] = salaries_df['BasePay'].astype('float')
salaries_df['OvertimePay'] = salaries_df['OvertimePay'].astype('float')
salaries_df['OtherPay'] = salaries_df['OtherPay'].astype('float')
salaries_df['Benefits'] = salaries_df['Benefits'].astype('float')
salaries_df['Agency'] = salaries_df['Agency'].astype('category')

#Cleaning up the data
del salaries_df['Notes']
del salaries_df['EmployeeName']
salaries_df['Benefits'].fillna(0, inplace=True)
salaries_df['Status'].fillna('Unknown', inplace=True)
salaries_df['Status'] = salaries_df['Status'].astype('category')
#drop the columns where BasePay is not available
salaries_df.dropna(inplace=True)
#salaries_df.isnull().sum()

#comparing the BasePay, OvertimePay and OtherPay
g = sns.pairplot(salaries_df[['BasePay','OvertimePay','OtherPay']], palette="Set2", diag_kind="hist", size=6)

def uppercase(series):
    #Convert a series/list of strings in to uppercase
    CAT = []
    for job in series:
        CAT.append(job.upper())
    return CAT

#Define a new subset-datarame for data analysis
df = salaries_df[['Id','BasePay','OvertimePay','OtherPay','Benefits','Year','Status','TotalPayBenefits']]
#Define Job Titles by converting them all in uppercase
df['JobTitle'] = uppercase(salaries_df['JobTitle'])

#most popular job titles
top_ten = df['JobTitle'].value_counts()[:10]
top_ten.plot.bar()

#Richest jobs - 300k+
rich = df[df['TotalPayBenefits'] >= 300000]
rich['JobTitle'] = rich['JobTitle'].astype('category')
#group the jobs by job title
obj = rich.groupby('JobTitle')
#Top 10 most paying jobs
richest = obj.mean().dropna().sort_values(by='TotalPayBenefits',ascending=False)
richest[:10].plot(y = ['BasePay','OvertimePay','Benefits','OtherPay'],kind='bar')


""" Naive Bayes with cross validation """

def effect_of_k_on_scores(klist):
    """Function to test GaussianNB on a different 'k's for cross-validation
    Takes an input of integers (k) and returns a DataFrame of correspoding scores"""
    #DataFrame to store k values are corresponding scores
    kdf = pd.DataFrame(columns=['k','avg_score'])
    kdf['k'] = klist
    for i in klist:
        #K fold cross-validation
        folds = cross_validation.KFold(len(df),n_folds=i)
        
        #Naive Bayes on each fold
        scores = []
        for train, test in folds:
            train_df = df.iloc[train]
            test_df  = df.iloc[test]

            X = np.array(train_df[['BasePay','OvertimePay','OtherPay']])
            y = np.array(train_df['JobTitle'])
            clf = GaussianNB()
            clf.fit(X, y)
            Xtest = np.array(test_df[['BasePay','OvertimePay','OtherPay']])
            ytest = np.array(test_df['JobTitle'])
            scores.append(clf.score(Xtest,ytest))
        print scores
        kdf['avg_score'] = sum(scores)/len(scores)
    return kdf
    
effect_of_k_on_scores([2,6,10])