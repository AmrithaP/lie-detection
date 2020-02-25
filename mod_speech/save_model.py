import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

def save_model():
    path_model = r'./SVC.pkl'

    data = pd.read_excel("DS_mfcc.xlsx")
    training_set, test_set = train_test_split(
        data, 
        test_size=0.2, 
        random_state=1
    )

    X_train = training_set.iloc[:,1:-1].values
    Y_train = training_set.iloc[:,-1].values

    X_test = test_set.iloc[:,1:-1].values
    Y_test = test_set.iloc[:,-1].values

    classifier = SVC(kernel='poly', random_state = 1)
    classifier.fit(X_train, Y_train)

    with open(path_model, 'wb') as f:
        pickle.dump(classifier, f)
    return path_model
    

if __name__ == '__main__':
    save_model()