df = pd.read_csv("credit1.zip")   #loading data
df.info()
df.isnull().sum() 
df.describe() #checking statistical info
#since 'Credit_Score' is our target column, let's see the unique values there and what percentage each of them contributes
df['Credit_Score'].value_counts(normalize=True) * 100
df_prep = df.copy()
df_prep.head(3)
# remove CustomerID to see duplicate rows
df_prep.drop('ID', axis=1, inplace=True)
df_prep.drop('Customer_ID', axis=1, inplace=True)
df_prep.drop('Month', axis=1, inplace=True)
df_prep.drop('Name', axis=1, inplace=True)
df_prep.drop('Age', axis=1, inplace=True)
df_prep.drop('SSN', axis=1, inplace=True)
df_prep.drop('Occupation', axis=1, inplace=True)
df_prep.drop('Payment_Behaviour', axis=1, inplace=True)
df_prep.drop('Credit_Utilization_Ratio', axis=1, inplace=True)
df_prep.drop('Type_of_Loan', axis=1, inplace=True)

df_prep.drop('Changed_Credit_Limit', axis=1, inplace=True)
df_prep.drop('Num_Credit_Inquiries', axis=1, inplace=True)

df_prep.drop('Payment_of_Min_Amount', axis=1, inplace=True)
df_prep.drop('Total_EMI_per_month', axis=1, inplace=True)
df_prep.drop('Amount_invested_monthly', axis=1, inplace=True)

#Transforming the column in 0, 1, or 2.
df_prep['Credit_Mix'] = df['Credit_Mix'].map({'Good': 2, 'Standard': 1, 'Bad': 0})
df_prep['Credit_Score'] = df['Credit_Score'].map({'Good': 2, 'Standard': 1, 'Poor': 0})

# Importing train_test_split and splitting the data in X (features) and y(target)
from sklearn.model_selection import train_test_split

X = df_prep[["Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate", "Num_of_Loan",
        "Delay_from_due_date", "Num_of_Delayed_Payment", "Credit_Mix", "Outstanding_Debt", "Credit_History_Age",
        "Monthly_Balance"]].values
y = df_prep['Credit_Score'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)
print(f'The amount of training data consists of {len(X_train)} rows')
print(f'Number of class 0 : {sum(y_train==0)}')
print(f'Number of class 1 : {sum(y_train==1)}\n')
print(f'Number of class 2 : {sum(y_train==2)}\n')

print(f'The amount of test data consists of {len(X_test)} rows')
print(f'Number of class 0 : {sum(y_test==0)}')
print(f'Number of class 1 : {sum(y_test==1)}\n')
print(f'Number of class 2 : {sum(y_test==2)}\n')

#Normalize Data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
col=["Annual_Income", "Monthly_Inhand_Salary","Outstanding_Debt","Credit_History_Age","Monthly_Balance"]
for i in col:
    df_prep[col] = scaler.fit_transform(df_prep[col])
    
print(f'The amount of training data before SMOTE consists of {len(X_train)} rows')
print(f'Number of class 0 in training data: {sum(y_train==0)}')
print(f'Number of class 1 in training data: {sum(y_train==1)}\n')
print(f'Number of class 2 in training data: {sum(y_train==2)}\n')

# Applying SMOTE to the training data
smote = SMOTE( random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print(f'Amount of training data after SMOTE consisting of {len(X_train)} rows')
print(f'Number of class 0 in resampled training data: {sum(y_train==0)}')
print(f'Number of class 1 in resampled training data: {sum(y_train==1)}\n')
print(f'Number of class 2 in resampled training data: {sum(y_train==2)}\n')

def evaluate_model(y_test,y_pred):
    print("Classification Report")
    print(classification_report(y_test, y_pred))

    print("\n---------------------------------------------\n")
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Create a heatmap of the confusion matrix using Seaborn
    sns.heatmap(cm, annot=True, cmap='Greens',fmt='.0f')

    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')

    plt.show()

#cross validation
# List of classifiers to test
classifiers = [
    ('Decision Tree', DecisionTreeClassifier()),
    ('Random Forest', RandomForestClassifier()),
    #('KNN', KNeighborsClassifier(n_neighbors=5)),
    #('Logistic regresssion', LogisticRegression())
]

# Iterate over each classifier and evaluate performance 
for clf_name, clf in classifiers:
    # Perform cross-validation
    scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')

    # Calculate average performance metrics
    avg_accuracy = scores.mean()
    avg_precision = cross_val_score(clf, X_train, y_train, cv=5, scoring='precision_macro').mean()
    avg_recall = cross_val_score(clf, X_train, y_train, cv=5, scoring='recall_macro').mean()

    # Print the performance metrics
    print(f'Classifier: {clf_name}')
    print(f'Average Accuracy: {avg_accuracy:.4f}')
    print(f'Average Precision: {avg_precision:.4f}')
    print(f'Average Recall: {avg_recall:.4f}')
    print('-----------------------')

# Creating the Random Forest classifier
model_rf = RandomForestClassifier(random_state=42)

# Training the classifier
model_rf.fit(X_train, y_train)

# Making predictions on the test set
y_pred = model_rf.predict(X_test)
accuracy = round(accuracy_score(y_test, y_pred), 4)
print(accuracy)

#hyperparameter
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 800, num = 10)]
# Number of features to consider at every split
max_features = ['sqrt','log2']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(1,40,10)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10 ,15]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4 ]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
              'criterion':['entropy','gini']}
print(random_grid)
rf_randomcv=RandomizedSearchCV(estimator=model_rf,param_distributions=random_grid,n_iter=10, cv=5, scoring='accuracy',random_state=42)
### fit the randomized model
rf_randomcv.fit(X_train,y_train)
rf_randomcv.best_params_
best_random_grid=rf_randomcv.best_estimator_
from sklearn.metrics import accuracy_score
y_pred=best_random_grid.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print("Accuracy Score {}".format(accuracy_score(y_test,y_pred)))
print("Classification report: {}".format(classification_report(y_test,y_pred)))
model_rf=rf_randomcv


#pickle.dump(model_rf, open('model23.pkl', 'wb'))
joblib.dump(model_rf, 'credmodel.joblib')
