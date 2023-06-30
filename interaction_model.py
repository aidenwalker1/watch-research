from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

import pickle
import numpy as np

num_features = 5

# creates base model
def form_initial_model(data) :
    sm = SMOTE(sampling_strategy='minority')
    
    # resamples data
    X = data[:,0:num_features]
    y = data[:,num_features]
    X, y = sm.fit_resample(X,y)

    # searches for best params for boost, then forms model
    clf = ensemble.AdaBoostClassifier()
    parameters = {
              'n_estimators':[10,25,50,100, 200],
              'learning_rate':[0.0001, 0.005, 0.01,0.1]
              }
    
    grid = GridSearchCV(clf, parameters, refit = True, verbose = 3,n_jobs=-1,scoring="accuracy") 
    grid.fit(X, y)

    pickle.dump(grid, open('./model.sav', 'wb'))
    return grid

# creates main model
def form_mlp(data) :
    X = data[:,:num_features+1]
    y = data[:,num_features+1]

    mlp = MLPClassifier(hidden_layer_sizes=(16,8,4),activation='tanh',batch_size=64,solver='adam', max_iter=30000,tol=1e-8)
    
    mlp.fit(X,y)
    pickle.dump(mlp, open('./mlpmodel.sav', 'wb'))
    return mlp

def append_preds(data, model) :
    arr = np.array(data)
    preds = model.predict(arr[:,:num_features])
    newarr = []

    for i in range(0, len(preds)) :
        out = preds[i]
        newarr.append(out)

    arr = np.insert(arr, num_features,newarr, axis=1)

    return arr

# gets both main model and model used to help it
def form_models(data, already_trained) :
    success = True
    base_model = None
    mlp = None

    # tries to get base model
    try :
        base_model = pickle.load(open('./model.sav', 'rb'))
    except :
        success = False

    # trains base model if needed
    if not already_trained or not success:
        base_model = form_initial_model(data)

    success = True

    # tries to get main model
    try :
        mlp = pickle.load(open('./mlpmodel.sav', 'rb'))
    except :
        success = False

    #trains main model if needed
    if not already_trained or not success:
        data = append_preds(data, base_model) # gets data with base model helping
        mlp = form_mlp(data)
    
    return base_model, mlp

# tests model accuracy
def test_accuracy(model, X, y) :
    predictions = model.predict(X)
    correct_outputs = y
    tn, fp, fn, tp = metrics.confusion_matrix(correct_outputs, predictions).ravel()
    print("Accuracy = %f" % (metrics.accuracy_score(correct_outputs, predictions)))
    print("TN = %d FP = %d FN = %d TP = %d" % (tn, fp, fn, tp))
    print(metrics.classification_report(correct_outputs, predictions, target_names = ["No interaction", "Interaction"]))