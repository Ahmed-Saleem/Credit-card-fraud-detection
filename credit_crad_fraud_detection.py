import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

data = pd.read_csv('creditcard.csv')
fraud= data.loc[data['Class']==1]
normal=data.loc[data['Class']==0]
print(len(fraud))
print(len(normal))
X= data.loc[:,data.columns != 'Class'].values
y=data.loc[:,'Class'].values

logreg=LogisticRegression(max_iter=3000)
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=42)
logreg.fit(X_train,y_train)
y_pred= logreg.predict(X_test)
print('accuracy score= ', accuracy_score(y_test, y_pred)*100,'%')
print('confusion matrix= ', confusion_matrix(y_test, y_pred))
