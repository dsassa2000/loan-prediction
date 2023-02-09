from multiprocessing.reduction import duplicate
from sqlalchemy.engine import create_engine
# Data analysis and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import seaborn as sns

# classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree

# metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB 
from sklearn.naive_bayes import BernoulliNB

DIALECT = 'oracle'
SQL_DRIVER = 'cx_oracle'
USERNAME = 'system' #enter your username
PASSWORD = '123' #enter your password
HOST = 'localhost' #enter the oracle db host url
PORT = 1521 # enter the oracle port number
SERVICE = 'xe' # enter the oracle db service name
ENGINE_PATH_WIN_AUTH = DIALECT + '+' + SQL_DRIVER + '://' + USERNAME + ':' + PASSWORD +'@' + HOST + ':' + str(PORT) + '/?service_name=' + SERVICE

engine = create_engine(ENGINE_PATH_WIN_AUTH)


#test query

test_df = pd.read_sql_query('SELECT * FROM LOAN_DATASET', engine)
#sort = test_df.head()
#print(sort)
data_with_NaN = test_df.isna().sum() 
print(data_with_NaN )
values = {"gender": "Male","married": "No", "dependents": 0, "self_employed": "Yes", "loanamount": test_df['loanamount'].median(), "loan_amount_term":test_df['loan_amount_term'].median(), "credit_history": 0}
clean_data = test_df.fillna(value=values)
# replacing by 0 and 1
clean_data.loan_status = clean_data.loan_status.replace({"Y": 1, "N" : 0})
clean_data.gender = clean_data.gender.replace({"Male": 1, "Female" : 0})
clean_data.married = clean_data.married.replace({"Yes": 1, "No" : 0})
clean_data.self_employed = clean_data.self_employed.replace({"Yes": 1, "No" : 0})
clean_data.education = clean_data.education.replace({"Graduate": 1, "Not Graduate" : 0})
clean_data.dependents= clean_data.dependents.replace({'3+':4})
clean_data.property_area = clean_data.property_area.replace({"Urban": 2, "Rural" : 0, "Semiurban":1})

print(clean_data)
display_data = clean_data.isna().sum()
print(display_data)
#test if we have duplicated rows in my dataframe if not we don't need to use function drop_duplicate()
duplicate_data = clean_data.duplicated()
print(duplicate_data)
fig, ax = plt.subplots(3, 2, figsize=(16, 18))

clean_data.groupby(['gender'])[['gender']].count().plot.bar(
    color=plt.cm.Paired(np.arange(len(clean_data))), ax=ax[0,0])
clean_data.groupby(['married'])[['married']].count().plot.bar(
    color=plt.cm.Paired(np.arange(len(clean_data))), ax=ax[0,1])
clean_data.groupby(['education'])[['education']].count().plot.bar(
    color=plt.cm.Paired(np.arange(len(clean_data))), ax=ax[1,0])
clean_data.groupby(['self_employed'])[['self_employed']].count().plot.bar(
    color=plt.cm.Paired(np.arange(len(clean_data))), ax=ax[1,1])

clean_data.groupby(['loan_status'])[['loan_status']].count().plot.bar(
    color=plt.cm.Paired(np.arange(len(clean_data))),ax=ax[2,0])
clean_data.groupby(['property_area'])[['loan_status']].count().plot.bar(
    color=plt.cm.Paired(np.arange(len(clean_data))),ax=ax[2,1])

plt.show()

# Here, I pass all categorical columns into a list

categorical_columns = clean_data.select_dtypes('object').columns.to_list()
# Then, I filter he list to remove Loan_ID column which is not relevant to the analysis
print(categorical_columns[1:])
# This code loops through the list, and creates a chart for each
#ax = sns.barplot(x = "gender", y = "loan_status", data = clean_data)
#ax.set(xlabel="X Label", ylabel = "Y Label")

fig, ax = plt.subplots(3, 2, figsize=(16, 18))
sns.countplot(x='gender' ,hue='loan_status', data=clean_data, palette='ocean',ax=ax[0,0])
sns.countplot(x='married' ,hue='loan_status', data=clean_data, palette='ocean',ax=ax[0,1])
sns.countplot(x='dependents' ,hue='loan_status', data=clean_data, palette='ocean',ax=ax[1,0])
sns.countplot(x='education' ,hue='loan_status', data=clean_data, palette='ocean',ax=ax[1,1])
sns.countplot(x='self_employed' ,hue='loan_status', data=clean_data, palette='ocean',ax=ax[2,0])
sns.countplot(x='credit_history' ,hue='loan_status', data=clean_data, palette='ocean',ax=ax[2,1])
#sns.countplot(x='property_area' ,hue='loan_status', data=clean_data, palette='ocean',ax=ax[2,2])
plt.show()
############################################# Logistic regression ##################################################
logistic_model = LogisticRegression()
logistic_features = ['self_employed','gender','married','credit_history','property_area']

x_logistic = clean_data[logistic_features].values
y_logistic = clean_data['loan_status'].values
logistic_model.fit(x_logistic, y_logistic)
predicted = logistic_model.predict_proba(x_logistic)
print("predict : ",predicted)
print('Coefficient of model :', logistic_model.coef_)
#confusio  matrix
cm_log = confusion_matrix(y_logistic, logistic_model.predict(x_logistic))
print(cm_log)
# check the intercept of the model
print('Intercept of model',logistic_model.intercept_)
# Accuray Score on train dataset
score = logistic_model.score(x_logistic, y_logistic)
print('accuracy_score of logistic regression :', score)
#100% 
print('accuracy_score percent :', round(score*100,2))


fig, ax = plt.subplots(figsize=(9, 7))
correlations = clean_data.corr()
  
# plotting correlation heatmap
dataplot = sns.heatmap(correlations, cmap="YlGnBu", annot=True)
  
# displaying heatmap
plt.show()

##################################### Naive Bays #################################################################
############## Gaussian for continious variables ######################
continious_features = ['applicantincome','coapplicantincome','loanamount','loan_amount_term'] 
x_Gauss = clean_data[continious_features].values
y_Gauss = clean_data['loan_status'].values
classifier_Gauss = GaussianNB()  
classifier_Gauss.fit(x_Gauss, y_Gauss)

# Predicting the Test set results  

y_pred = classifier_Gauss.predict(x_Gauss)
y_predict = classifier_Gauss.predict_proba(x_Gauss)
print(y_predict) 
score_Gauss = accuracy_score(y_Gauss, y_pred)  
print('accuracy_score of gaussian  :',score_Gauss)
# confusion matrix 
cm_Gauss = confusion_matrix(y_Gauss, y_pred)  

print(cm_Gauss) 
##################### bernoulli for discrete variable ######################
discrete_features = ['gender','married','education','self_employed','credit_history','property_area']
x_B = clean_data[discrete_features].values
y_B = clean_data['loan_status'].values
classifier_Bern = BernoulliNB()
classifier_Bern.fit(x_B, y_B)

# Prediction
y_proba = classifier_Bern.predict(x_B)
print(y_pred) 
score_Bern = accuracy_score(y_B, y_proba)  
print("bernoulli's accuracy score :",score_Bern)
#confusion matrix
cm_bern = confusion_matrix(y_B,y_proba)
print(cm_bern)
######################################### decision for small dataframe #############################
small_data = clean_data.head();
print(small_data);
tree_features_cat = ['gender','married','education','self_employed','credit_history','property_area']
X_tree = small_data[tree_features_cat].values
Y_tree = small_data['loan_status'].values
model = DecisionTreeClassifier()
model.fit(X_tree,Y_tree)
predict = model.predict(X_tree)
print(classification_report(Y_tree, predict))
print("Accuracy:", accuracy_score(predict, Y_tree))
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)

tree.plot_tree(model)
plt.show()





############################# decision tree ####################################
tree_features = ['gender','married','education','self_employed','credit_history','property_area']
x_tree = clean_data[tree_features].values
y_tree = clean_data['loan_status'].values
df_model = DecisionTreeClassifier()
df_model.fit(x_tree, y_tree)
predict_y = df_model.predict(x_tree)
print(classification_report(y_tree, predict_y))
print("Accuracy of decision tree:", accuracy_score(predict_y, y_tree))
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(df_model)
plt.show()
