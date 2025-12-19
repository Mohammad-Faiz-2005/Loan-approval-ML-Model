import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder , OneHotEncoder
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB,BernoulliNB,MultinomialNB
from sklearn.metrics import accuracy_score , precision_score , recall_score , f1_score , classification_report , confusion_matrix
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV , RandomizedSearchCV
import streamlit as st

st.title("Loan Approval Model Using Machine Learning")

dataset = pd.read_csv("loan.csv")
df = pd.DataFrame(dataset)

df.drop(columns = ["Loan_ID"] , inplace = True)
#df

#df.isna().sum()

for i in df.select_dtypes(["int64","float64"]):
    df[i].fillna(df[i].mean() , inplace = True)

for i in df.select_dtypes("object"):
    df[i].fillna(df[i].mode()[0] , inplace = True)

df.isna().sum()
print(df["Dependents"].value_counts())

df["Dependents"] = df["Dependents"].replace("3+","3")
df["Dependants"] = df["Dependents"].astype("int64")
print(df["Dependents"].value_counts())

#df

df.isna().sum()
#df
li = ["Gender" , "Married" , "Education" , "Self_Employed" , "Property_Area" , "Loan_Status"]
for i in li:
    ore = OrdinalEncoder()
    df[i] = ore.fit_transform(df[[i]])

#df

df.describe()
li2 = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]
for j in li2:
    plt.figure(figsize=(8,4))
    sns.boxplot(x=j,data = df)
    plt.show()

li3 = ["LoanAmount", "CoapplicantIncome", "ApplicantIncome"]

for k in li3:
    q1 = df[k].quantile(0.25)
    q3 = df[k].quantile(0.75)
    IQR = q3 - q1

    max_range = q3 + 1.5 * IQR
    min_range = q1 - 1.5 * IQR

    df = df[(df[k] >= min_range) & (df[k] <= max_range)]

#df


x = df.iloc[:,:-2]
y = df["Loan_Status"]

x_train , x_test , y_train ,y_test = train_test_split(x,y,test_size=0.25,random_state=42)

#x_train["ApplicantIncome"]

#y_train.value_counts()

ros = RandomOverSampler()
x_train_ros , y_train_ros = ros.fit_resample(x_train , y_train)
#y_train_ros.value_counts() , y_train.value_counts()


li4 =["ApplicantIncome" , "CoapplicantIncome" , "LoanAmount"]
x_train_ros_scalled = x_train_ros.copy()
x_test_scalled = x_test.copy()

ss = StandardScaler()
x_train_ros_scalled[li4] = ss.fit_transform(x_train_ros[li4])
x_test_scalled[li4] = ss.transform(x_test[li4])

#x_train
#x_train_ros_scalled


#y_train_ros.value_counts() , y_train.value_counts()

#df.shape , x_train.shape  , x_train_ros.shape , x_train_ros_scalled.shape

#y_train.shape , y_train_ros.shape   # no need of --> y_train_ros_scalled.shape

#x_train_ros_scalled.shape , y_train_ros.shape

#x_test_scalled.shape , y_test.shape

#y_test.value_counts()  # no need of ros

lr = LogisticRegression()
print(lr.fit(x_train_ros_scalled , y_train_ros))
print(lr.score(x_test_scalled , y_test)*100)
print(lr.score(x_train_ros_scalled , y_train_ros)*100)

svc = SVC()
print(svc.fit(x_train_ros_scalled , y_train_ros))
print(svc.score(x_test_scalled , y_test)*100)
print(svc.score(x_train_ros_scalled , y_train_ros)*100)

dtc = DecisionTreeClassifier()
print(dtc.fit(x_train_ros_scalled , y_train_ros))
print(dtc.score(x_test_scalled , y_test)*100)
print(dtc.score(x_train_ros_scalled , y_train_ros)*100)

knn = KNeighborsClassifier()
print(knn.fit(x_train_ros_scalled , y_train_ros))
print(knn.score(x_test_scalled , y_test)*100)
print(knn.score(x_train_ros_scalled , y_train_ros)*100)

gnb = GaussianNB()
print(gnb.fit(x_train_ros_scalled , y_train_ros))
print(gnb.score(x_test_scalled , y_test)*100)
print(gnb.score(x_train_ros_scalled , y_train_ros)*100)

bnb = BernoulliNB()
print(bnb.fit(x_train_ros_scalled , y_train_ros))
print(bnb.score(x_test_scalled , y_test)*100)
print(bnb.score(x_train_ros_scalled , y_train_ros)*100)

rfc = RandomForestClassifier(n_estimators=50)
print(rfc.fit(x_train_ros_scalled , y_train_ros))
print(rfc.score(x_test_scalled , y_test)*100)
print(rfc.score(x_train_ros_scalled , y_train_ros)*100)

#y_train_ros.value_counts() , y_train.value_counts()

dtc = DecisionTreeClassifier()
print(dtc.fit(x_train_ros , y_train_ros))
print(dtc.score(x_test , y_test)*100)
print(dtc.score(x_train_ros , y_train_ros)*100)

rfc1 = RandomForestClassifier(n_estimators=50)
print(rfc1.fit(x_train_ros , y_train_ros))
print(rfc1.score(x_test , y_test)*100)
print(rfc1.score(x_train_ros , y_train_ros)*100)

y_pred = rfc.predict(x_test)
cm = confusion_matrix(y_test , y_pred)
#cm

dt = DecisionTreeClassifier(criterion = "gini" , max_depth = 17 , splitter = "best")
print(dt.fit(x_train_ros , y_train_ros))
print(dt.score(x_test , y_test)*100)
print(dt.score(x_train_ros , y_train_ros)*100)

rfc2 = RandomForestClassifier(n_estimators=100 , criterion = "gini" , max_depth = 15)
print(rfc2.fit(x_train_ros , y_train_ros))
print(rfc2.score(x_test , y_test)*100)
print(rfc2.score(x_train_ros , y_train_ros)*100)

#['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed','ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
#       'Loan_Amount_Term', 'Credit_History', 'Property_Area']

Gender = st.radio("Gender : ",["Male" , "Female"])
if Gender == "Male":
    Gender = 1
else:
    Gender = 0

Married = st.radio("Married Status :" ,["Married" , "Unmarried"])
if Married == "Married":
    Married = 1
else:
    Married = 0

Dependents = st.radio("Dependents :" ,["0" , "1" , "2" , "3+"])
if Dependents == "0":
    Dependents = 0

elif Dependents == "1":
    Dependents = 1

elif Dependents == "2":
    Dependents = 2

else:
    Dependents = 3


Education = st.radio("Education :",["Graduate","Not Graduate"])
if Education == "Graduate":
    Education = 0
else:
    Education = 1


Self_Employed = st.radio("Self Employed :" , ["Yes","No"])
if Self_Employed == "Yes":
    Self_Employed = 1
else:
    Self_Employed = 0

ApplicantIncome = st.number_input("Enter ApplicantIncome :" ,step=1)

CoapplicantIncome = st.number_input("Enter CoapplicantIncome :" ,step=1)

LoanAmount = st.number_input("Enter LoanAmount :" ,step=1)

Loan_Amount_Term = st.number_input("Enter Loan_Amount_Term :" ,step=1)

Credit_History = st.radio("Credit_History :" , ["Yes","No"])
if Credit_History == "Yes":
    Credit_History = 1
else:
    Credit_History = 0

Property_Area = st.radio("Property_Area :" , ["Rural","Urban","Semiurban"])
if Property_Area == "Rural":
    Property_Area = 0

elif Property_Area == "Urban":
    Property_Area = 2
else:
    Property_Area = 1



res = [[Gender,Married ,Dependents, Education, Self_Employed,ApplicantIncome, CoapplicantIncome, LoanAmount,
      Loan_Amount_Term, Credit_History, Property_Area]]

y_pred = rfc2.predict(res)
print(f"test accuracy: {rfc2.score(x_test , y_test)*100}")
print(f"train accuracy: {rfc2.score(x_train_ros , y_train_ros)*100}")
y_pred = rfc2.predict(res)
print(y_pred)

st.text(f"test accuracy: {rfc2.score(x_test , y_test)*100}")
st.text(f"train accuracy: {rfc2.score(x_train_ros , y_train_ros)*100}")

if y_pred == 0:
    st.info("Loan is Rejected")
else:
    st.success("Loan is approved")
