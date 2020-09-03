from Header import *

from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score


def plotRegulParam(lambdas, X, y, n_cv=5, logScale=True, filename=None):
    """
Plots the cross-validation error as a function of the regularization parameter for logistic regression.
lambdas: array of regularization parameters
X: training vector
y: target values
n_cv: data is splitted into n_cv parts for cross-validation
logScale: plot horizontal axis on a logarithmic scale
filename: save result to file with this name
    """
    cv_scores = np.zeros(len(lambdas))
    f1_scores = np.zeros(len(lambdas))

    for i,l in enumerate(lambdas):
        maxent = linear_model.LogisticRegression(penalty='l2', C=1/l, solver='liblinear')
        cv_scores[i] = np.mean(cross_val_score(maxent, X, y, cv=n_cv, scoring='accuracy'))
        f1_scores[i] = np.mean(cross_val_score(maxent, X, y, cv=n_cv, scoring='f1'))  
    if logScale:
        plt.semilogx(lambdas, cv_scores, 'o-', color="r",label="accuracy")
        plt.semilogx(lambdas, f1_scores, 'o-', color="g",label="F1 score")
    else:
        plt.plot(lambdas, cv_scores, 'o-', color="r",label="CV score")
        plt.plot(lambdas, f1_scores, 'o-', color="g",label="F1 score")
    plt.ylim([0.5,1]); plt.legend(); plt.xlabel("Regularization parameter")
    plt.ylabel(r"score")
    if filename is not None:
        plt.tight_layout()
        plt.savefig(filename)
    else:
        plt.show()
    
def plotLearning(lamb, X, y, n_cv=5, filename=None):
    """
Plots learning curve for logistic regression.
lamb: regularization parameter
X: training vector
y: target values
n_cv: data is splitted into n_cv parts for cross-validation
filename: save result to file with this name
    """
    maxent = linear_model.LogisticRegression(penalty='l2', C=1/lamb, solver='liblinear')

    n_jobs = 4 # number of jobs
    train_sizes = np.linspace(0.2, 1.0, 20) # fraction of training data used
    train_sizes, train_scores, test_scores = \
            learning_curve(maxent, X, y, cv=n_cv, n_jobs=n_jobs,
                           train_sizes=train_sizes, scoring='f1')
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                     label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                     label="Cross-validation score")
    plt.plot(train_sizes, test_scores_mean+test_scores_std, '-', color="g")
    plt.plot(train_sizes, test_scores_mean-test_scores_std, '-', color="g")
    plt.ylim([0.5,1.05]); plt.legend(); plt.xlabel("size of training set")
    plt.ylabel(r"$F_1$ score")
    if filename is not None:
        plt.tight_layout()
        plt.savefig(filename)
    else:
        plt.show()

def precision(conf_mat, idx=0):
    """
Precision from configuration matrix
conf_max: configuration matrix. shape=(2,2)
idx: index to calculate precision for

return: precision
    """
    return conf_mat[idx,idx] / (conf_mat[0,idx] + conf_mat[1,idx])

def recall(conf_mat, idx=0):
    """
Recall from configuration matrix
conf_max: configuration matrix. shape=(2,2)
idx: index to calculate recall for

return: recall
    """
    return conf_mat[idx,idx] / (conf_mat[idx,0] + conf_mat[idx,1])

def f1Score(conf_mat, idx=0):
    """
F1 score from configuration matrix
conf_max: configuration matrix. shape=(2,2)
idx: index to calculate F1 score for

return: F1 score
    """
    p = precision(conf_mat, idx)
    r = recall(conf_mat, idx)
    return 2.*p*r / (p+r)
    
def kFoldSummary(encoder, lamb, X, y, n_cv, pr_idx=0):
    """
Prints a summary including the configuration matrix and different scores for logistic regression
encoder: Encoder containing the labels for the target values
lamb: regularization parameter
X: training vector
y: target values
n_cv: data is splitted into n_cv parts for cross-validation
pr_idx: Index to calculate precision and recall for
    """
    kf = KFold(n_splits=n_cv)
    confusMatrices = []
    f1Scores = []
    accuracies = []
    recalls = []
    precisions = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        maxent = linear_model.LogisticRegression(penalty='l2', C=1/lamb, solver='liblinear')

        maxent.fit(X_train, y_train)
        y_pred = maxent.predict(X_test)
        # Vorhersage-Genauigkeit auswerten
        cm = confusion_matrix(y_test, y_pred)
        confusMatrices.append((cm /  np.sum(cm)))
        f1Scores.append(f1_score(y_test, y_pred))
        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision(cm, pr_idx))
        recalls.append(recall(cm, pr_idx))


    confusMatrix_mean = np.mean(np.array(confusMatrices), axis=0)
    confusMatrix_std = np.std(np.array(confusMatrices), axis=0)
    print("accuracy score: {:.3f} +/- {:.3f}".format(np.mean(accuracies), 
                                                     np.std(accuracies)))
    print("f1_score: {:.3f} +/- {:.3f}".format(np.mean(f1Scores), np.std(f1Scores)))
    print("precision for index {}: {:.3f} +/- {:.3f}".format(
        pr_idx,np.mean(precisions), np.std(precisions)) )
    print("recall for index {}: {:.3f} +/- {:.3f}".format(
        pr_idx,np.mean(recalls), np.std(recalls)) ) 
    print("confusion matrix mean")
    print(encoder.classes_)
    print(confusMatrix_mean)
    print("confusion matrix std:\n",confusMatrix_std)
    print("\nPredicted label on x-axis, True label on y-axis")
    print('\nIdeal would be')
    print('{:.3f}   0\n0       {:.3f}'.format(np.sum(y==0)/len(y),np.sum(y==1)/len(y)))
