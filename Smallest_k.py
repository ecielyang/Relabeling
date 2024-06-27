from sklearn.linear_model import LogisticRegression
import sklearn.metrics as mc
import warnings
import numpy as np
from tqdm import tqdm
warnings.filterwarnings("ignore")

# In[15]:


def Flip(k, scores, test_idx, pred, X, y, thresh):
    #print("test_idx", test_idx)
    #print("old")
    #print(pred[test_idx])
    
    if pred[test_idx] > thresh:
        top_k_index = scores[test_idx].argsort()[-k:]
    else:
        top_k_index = scores[test_idx].argsort()[:k]
        
    y_k = y["train"]
    X_k = X["train"]
    #top_k_index = [random.randint(0, X["train"].shape[0])]
    
    for i in top_k_index:
        if y["train"][i] == 0:
            y_k[i] = 1
        else:
            y_k[i] = 0
        
    prediction = -np.sum(scores[test_idx][top_k_index])
    #print("prediction", prediction)

    return X_k, y_k, prediction, top_k_index


# In[16]:


def new_train(k, dev_index, scores, l2, X, model, pred, y, thresh):
    X_k, y_k, prediction, top_k_index = Flip(k, scores, dev_index, pred, X, y, thresh)
    
    if y_k.shape[0] == np.sum(y_k) or np.sum(y_k) == 0: # data contains only one class: 1
        return None, None, None

    # Fit the model again
    model_k = LogisticRegression(penalty='l2', C=1/l2)
    model_k.fit(X_k, y_k)

    # predictthe probaility with test point
    test_point = X["dev"][dev_index]
    test_point=np.reshape(test_point, (1,-1))
    
    new = model_k.predict_proba(test_point)[0][1]
    change = -(model.predict_proba(test_point)[0][1] - new)
    #change = model_k.predict_proba(test_point)[0][1]-model.predict_proba(test_point)[0][1]
    flip = (model.predict(test_point) == model_k.predict(test_point))
    
    """
    print("change    ", change)
    print("old       ", model.predict_proba(test_point)[0][1])
    print()
    """
    error = np.abs((change - prediction)/prediction)
    return change, flip, prediction,new, error, top_k_index


# # Find approximate k by IF


def approximate_k(test_idx, pred, delta_pred, y, thresh):
    old = pred[test_idx].item()
    
    if pred[test_idx] > thresh:
        top_k_index = np.flip(delta_pred[test_idx].argsort())
    else:
        top_k_index = delta_pred[test_idx].argsort()
    
    for k in range(1, y["train"].shape[0]):
        change = -np.sum(delta_pred[test_idx][top_k_index[:k]])
        
        if old > thresh and old + change < thresh:
            return k
        elif old < thresh and old + change > thresh:
            return k
        
    return None


def loss_gradient(X, y, model):
    F_train = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    error_train = model.predict_proba(X)[:, 1] - y
    gradient_train = F_train * error_train[:, None]
    return gradient_train

def IP(X, y, l2, dataname, thresh, modi=None):
    model = LogisticRegression(penalty='l2', C=1/l2)
    model.fit(X["train"], y["train"])
    pred = np.reshape(model.predict_proba(X["dev"])[:, 1], (model.predict_proba(X["dev"])[:, 1].shape[0], 1))

    y_flip = []
    for i in y["train"]:
        if i == 0:
            y_flip.append(1)
        else:
            y_flip.append(0)

    gradient_train_flip = loss_gradient(X["train"], y_flip, model)

    w = np.concatenate((model.coef_, model.intercept_[None, :]), axis=1)
    F_train = np.concatenate([X["train"], np.ones((X["train"].shape[0], 1))],
                             axis=1)  # Concatenating one to calculate the gradient with respect to intercept
    F_dev = np.concatenate([X["dev"], np.ones((X["dev"].shape[0], 1))], axis=1)

    error_train = model.predict_proba(X["train"])[:, 1] - y["train"]
    error_dev = model.predict_proba(X["dev"])[:, 1] - y["dev"]

    gradient_train = F_train * error_train[:, None]
    gradient_dev = F_dev * error_dev[:, None]

    probs = model.predict_proba(X["train"])[:, 1]
    hessian = F_train.T @ np.diag(probs * (1 - probs)) @ F_train / X["train"].shape[0] + l2 * np.eye(F_train.shape[1]) / \
              X["train"].shape[0]
    inverse_hessian = np.linalg.inv(hessian)

    eps = 1 / X["train"].shape[0]
    delta_k = -eps * inverse_hessian @ (gradient_train - gradient_train_flip).T
    grad_f = F_dev * (pred * (1 - pred))
    delta_pred = grad_f @ delta_k


    # Loop over all dev points:
    appro_ks = []
    new_predictions = []
    flip_list = []
    for test_idx in tqdm(range(X["dev"].shape[0])):
        appro_k = approximate_k(test_idx, pred, delta_pred, y, thresh)
        if appro_k != None:
            #X_k, y_k, prediction, top_k_index = Flip(appro_k, delta_pred, test_idx, pred, X, y, thresh)

            change, _, prediction,new_prediction, error, top_k_index = new_train(appro_k, test_idx, delta_pred, l2, X, model, pred, y, thresh)
            print(test_idx, appro_k, pred[test_idx], new_prediction)
            ##print()
            appro_ks.append(appro_k)
            new_predictions.append(new_prediction)

            flip_list.append(top_k_index)
            #print("appro_k", appro_k, "overlap", np.sum([1 for i in top_k_index if i in modi]))
        else:

            appro_ks.append(None)
            new_predictions.append(None)

            flip_list.append(None)

    appro_ks= np.array(appro_ks)
    new_predictions=np.array(new_predictions)
    flip_list = np.array(flip_list)
    np.save("./results/" + "appro_ks_IP" + "_alg1_" + dataname  + str(l2) + ".npy", appro_ks)
    np.save("./results/" + "new_predictions" + "_alg1_" + dataname + str(l2) + ".npy", new_predictions)
    np.save("./results/" + "old_predictions" + "_alg1_" + dataname + str(l2) + ".npy", pred)
    np.save("./results/" + "flip_list" + "_alg1_" + dataname + str(l2) + ".npy", flip_list)




