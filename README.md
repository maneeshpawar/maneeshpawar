Exploratory Data Analysis on the Titanic Dataset
This is my first kernal and the objective of this kernal is to conduct exploratory data analysis (EDA) and statistical modeling on the Titanic Dataset in order to gather insights and evenutally predicting survior(0 = Not Survived, 1 = Survived). Out of the 891 passengers that went on board the titanic, approximately 38% of them got surived where as majority 62% did not survive the disaster.

I have outlined below the process i followed in conducting the aforementioned procedure

1.Import the relevant python libraies for the analysis 2.Load the train and test dataset and set the index if applicable 3.Visually inspect the head of the dataset,Examine the train dataset to understand in particular if the data is tidy, shape of the dataset,examine datatypes, examine missing values, unique counts and build a data dictictionary dataframe 4.Run discriptive statistics of object and numerical datatypes, and finally transform datatypes accordingly 5.Carry-out univariate,bivariate and multivariate analysis using graphical and non graphical(some numbers represting the data) mediums 6.Feature Engineering : Extract title from name, Extract new features from name, age, fare, sibsp, parch and cabin 7.Preprocessing and Prepare data for statistical modeling 8.Statistical Modelling
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns,set()
titanic_data=pd.read_csv("/content/train.csv")
titanic_test=pd.read_csv("/content/test.csv")
titanic_test.head()
titanic_test.isnull
sns.heatmap(titanic_test.isnull(),yticklabels=False);sns.heatmap(titanic_test.isnull(),yticklabels=False);
titanic_test.drop('Cabin',axis=1,inplace=True)
titanic_test.head()
titanic_test['Pclass'].unique()
titanic_test.shape
mean_age = titanic_test.groupby('Pclass').mean()['Age']
mean_age
def import_age(cols):
    Age = cols[0]
    Pclass = cols[1]

    if pd.isnull(Age):
       if Pclass == 1:
            return 41
       elif Pclass ==2:
            return 29
       else: 
            return 24
    else:
      return Age
      titanic_test['Age'] = titanic_test[['Age','Pclass']].apply(import_age,axis=1)
      sns.heatmap(titanic_test.isnull(),yticklabels=False);
      titanic_test['Fare'] = titanic_test['Fare'].fillna(titanic_test['Fare'].mean())
      sns.heatmap(titanic_test.isnull(),yticklabels=False);
      titanic_test.dropna(inplace=False);
      titanic_test.isnull().sum()
      titanic_test.info()
      titanic_test.head()
Converting Age and Embarked colum categorical date intoc numerical
Sex = pd.get_dummies(titanic_test['Sex'])
Sex.head()
Sex = pd.get_dummies(titanic_test['Sex'],drop_first = True)
Sex.head()
Embark = pd.get_dummies(titanic_test['Embarked'],drop_first = True)
Embark.head()
titanic_test.head()
titanic_test.drop(['Sex','Embarked','Name','PassengerId','Ticket'],axis=1,inplace=True)
titanic_test.head()
titanic = pd.concat([titanic_test,Sex,Embark],axis=1)
titanic.head()
titanic_test.info()
plt.figure(figsize=(8,4))
plt.xlabel('Age')
titanic['Age'].plot.hist(edgecolor='k').autoscale(enable=True,axis='both',tight=True);
titanic_test.head()
figure = plt.figure(figsize=(10,5));
plt.title('Titanic');
sns.countplot(x='SibSp',hue='Pclass',data=titanic);
plt.xlabel('SibSp',fontsize=14);
plt.ylabel('count',fontsize=14);
plt.yticks(fontsize=14);
plt.figure(figsize=(12,7))
sns.boxplot(x='Pclass',y='Age',data = titanic_test);
titanic_data.head()
titanic_data.isnull
sns.heatmap(titanic_data.isnull(),yticklabels=False);
titanic_data.drop('Cabin',axis=1,inplace=True)
titanic_data.head()



      
