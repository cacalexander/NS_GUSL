import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

def sigmoid(x):
    # Use clipping to prevent overflow in exp
    return 1 / (1 + np.exp(-np.clip(x, -15, 15)))

def obj(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)

        p = sigmoid(y_pred)

        beta = 0.25
        gamma = 0.5

        w = 1 + beta * y_true - gamma * p
        #w = 
        #w = 1.0 + 0.5 * np.power(y_true, 1.0)

        grad = w * (p - y_true)
        hess = w * p * (1.0 - p)

        return grad, np.maximum(hess, 1e-6)

def combined_eval_metric(y_true, y_pred):
    """
    Evaluation metric for Combined MSE and Dice Loss.
    Note: y_pred is passed as raw values from the booster.
    """
    smooth = 0.1
    alpha = 0.8
    y_pred = sigmoid(y_pred)
    # 1. MSE Component
    mse = np.mean((y_true - y_pred)**2)
    
    # 2. Dice Component
    # Dice = 2*sum(xy) / (sum(x) + sum(y))
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    dice_score = (2.0 * intersection + smooth) / (union + smooth)
    dice_loss = 1.0 - dice_score
    #log_dice_loss = -np.log(dice_score + 1e-7)
    # Combined Loss
    total_loss = (alpha * mse) + ((1 - alpha) * dice_loss)
    
    return float(total_loss)

def combined_dice_mse_loss(y_true, y_pred):
    smooth = 0.1
    alpha = 0.8  # Weighting: 0.5 means equal parts MSE and Dice
    
    # --- Dice Component ---
    # We treat y_pred as probabilities/intensities
    y_pred = sigmoid(y_pred)
    dp = y_pred * (1.0 - y_pred)
    d2p = dp * (1.0 - 2.0 * y_pred)
    intersection = y_pred * y_true
    union = y_pred + y_true
    
    # Gradient of Dice Loss
    dice_grad = -2 * (y_true * (union + smooth) - (intersection + smooth)) / ((union + smooth)**2)
    dice_grad = dice_grad *dp
    #Gradient of Log Dice Loss
    #dice_grad = -2 * (y_true * (union + smooth) - (intersection + smooth)) / ((union + smooth) * (2 * intersection + smooth))
    # Hessian approximation for stability
    dice_hess = 2 * (y_true * dice_grad * (union + smooth)) / ((union + smooth)**2)
    #dice_hess = 0.01
    dice_hess = dice_hess *(dp ** 2) + dice_grad*d2p
    #dice_hess = np.maximum(dice_hess, 1e-3) # Ensure hessian is positive

    # --- MSE Component ---
    mse_grad = (y_pred - y_true)*dp
    mse_hess = np.ones_like(y_pred)* (dp ** 2) + mse_grad * d2p

    # --- Combined ---
    grad = alpha * mse_grad + (1 - alpha) * dice_grad
    hess = alpha * mse_hess + (1 - alpha) * dice_hess
    hess = np.maximum(hess, 0.001)
    return grad, hess

def plot_learning_curve(evals_result, eval_metric='logloss', path=None):
    plt.figure()
    plt.plot(evals_result['validation_0'][eval_metric], label='train')
    plt.plot(evals_result['validation_1'][eval_metric], label='val')
    plt.xlabel('Iteration')
    plt.ylabel(eval_metric)
    plt.legend()
    if path is not None:
        plt.savefig(path)
        plt.cla();plt.clf()
        #plt.show()
        #plt.close()

def train_iter_init_swin(args,level,  X_train, y_train,X_val, y_val,plot = False, savepath = None ):
    evalset = [(X_train, y_train), (X_val, y_val)]
    alpha = np.array([0.05, 0.5, 0.95])

    xgbr = xgb.XGBRegressor(tree_method='hist',device = 'cuda', max_depth=5, n_estimators=800, learning_rate=0.1, early_stopping_rounds = 10)
    #xgbr = xgb.XGBClassifier(tree_method='gpu_hist', max_depth=6, n_estimators=1000, learning_rate=0.07, early_stopping_rounds = 10)
    xgbr.fit(X_train,y_train, verbose = False , eval_set = evalset)
    if(plot):
        plot_learning_curve(xgbr.evals_result_, eval_metric = 'rmse', path = savepath)
    return xgbr
        
def train_iter_init(args,level,  X_train, y_train,X_val, y_val,g_name, sel_tr = None, res=None, plot = False, savepath = None ):
    evalset = [(X_train, y_train), (X_val, y_val)]
    alpha = np.array([0.05, 0.5, 0.95])
    weights = np.ones(len(y_train))
    if(sel_tr is not None):
        weights[sel_tr] = 1 + 0.5 * (abs(res[sel_tr]) / np.std(res[sel_tr]))
        weights = weights/np.mean(weights)
        weights = np.clip(weights, 1, 3)

    #weights[y_train>0.9] = 1.5
    #weights[y_train<0.1] = 0.75
    if(g_name[-1]!='2'):
        xgbr = xgb.XGBRegressor(objective='reg:logistic', tree_method='hist',device = 'cuda', max_depth=3, n_estimators=2000, learning_rate=0.03, reg_lambda = 10, early_stopping_rounds = 10)
    else: 
        xgbr = xgb.XGBRegressor(objective='reg:logistic', tree_method='hist',device = 'cuda', max_depth=5, n_estimators=4000, learning_rate=0.07, reg_lambda = 2, alpha = 0.1, subsample = 0.75,sampling_method='gradient_based', colsample_bytree = 0.75, colsample_bylevel = 0.75, min_child_weight = 1, early_stopping_rounds = 10)
    #xgbr = xgb.XGBClassifier(tree_method='gpu_hist', max_depth=6, n_estimators=1000, learning_rate=0.07, early_stopping_rounds = 10)
    xgbr.fit(X_train,y_train,  verbose = False , eval_set = evalset)
    if(plot):
        #plot_learning_curve(xgbr.evals_result_, eval_metric ='combined_eval_metric', path = savepath)
        plot_learning_curve(xgbr.evals_result_, eval_metric ='rmse', path = savepath)
    return xgbr



def train_iter(args,level,  X_train, y_train,X_val, y_val,g_name,sample_weights = None, plot = False, savepath = None ):
    evalset = [(X_train, y_train), (X_val, y_val)]
    sel = np.where(y_train>0.8)[0]
    #sample_weights = np.ones(len(X_train))
    #sample_weights[sel] = 1.5
    if(g_name[:3]!='grp'): 
        
        xgbr = xgb.XGBRegressor(tree_method='hist',device = 'cuda', objective = 'reg:logistic',  max_depth=args['max_depth'][2], n_estimators=args['estimators'][level-1][1], learning_rate=args['lr'], early_stopping_rounds=10, reg_lambda = 2, reg_alpha = 0.5, min_child_weight = 1, subsample = 0.75,sampling_method='gradient_based', colsample_bytree = 0.75, colsample_bylevel = 0.75)
       
    elif(g_name[-1]=='0'):
        xgbr = xgb.XGBRegressor(tree_method='hist',device = 'cuda', objective = 'reg:pseudohubererror', max_depth=args['max_depth'][int(g_name[-1])], n_estimators=args['estimators'][level-1][int(g_name[-1])], learning_rate=args['lr'], early_stopping_rounds=10, subsample = 0.8, colsample_bytree = 0.8)
    elif(g_name[-1]=='1'): 
        xgbr = xgb.XGBRegressor(tree_method='hist',device = 'cuda', objective = 'reg:pseudohubererror', max_depth=args['max_depth'][int(g_name[-1])], n_estimators=args['estimators_2'][level-1], learning_rate=args['lr'], early_stopping_rounds=10, subsample = 0.8, colsample_bytree = 0.75)
    elif(g_name[-1]=='2'): 
        xgbr = xgb.XGBRegressor(tree_method='hist',device = 'cuda',objective = 'reg:pseudohubererror',  max_depth=args['max_depth'][int(g_name[-1])], n_estimators=args['estimators'][level-1][int(g_name[-1])], learning_rate=args['lr'], early_stopping_rounds=10, subsample = 0.8, colsample_bytree = 0.75)
    xgbr.fit(X_train,y_train, verbose = False, eval_set = evalset)
    if(plot):
        plot_learning_curve(xgbr.evals_result_, eval_metric = 'rmse', path = savepath)
    return xgbr

def train_clf(args, X_train, y_train,X_val, y_val,round, pos_weight = None, plot = False, savepath = None):
    evalset = [(X_train, y_train), (X_val, y_val)]
    if(pos_weight is not None):
        xgbc = xgb.XGBClassifier(tree_method='hist',device = 'cuda', max_depth=args['clf_max_depth'][round -1], n_estimators=args['clf_estimators'][round-1], learning_rate=args['clf_lr'], max_delta_step = 1, early_stopping_rounds=30, scale_pos_weight = pos_weight, subsample = 0.75, sampling_method='gradient_based', colsample_bylevel = 0.8, colsample_bytree = 0.8, reg_lambda = 10)
    else: 
        xgbc = xgb.XGBClassifier(tree_method='hist',device = 'cuda', max_depth=args['clf_max_depth'][round -1], n_estimators=args['clf_estimators'][round-1], learning_rate=args['clf_lr'], max_delta_step = 1, early_stopping_rounds=30, subsample = 0.75, sampling_method='gradient_based', colsample_bylevel = 0.8, colsample_bytree = 0.8, reg_lambda = 20)
    xgbc.fit(X_train,y_train,eval_set= evalset, verbose = False )
    if(plot):
        plot_learning_curve(xgbc.evals_result(), eval_metric = args['clf_eval_metric'], path = savepath)
    return xgbc

def train_clf_inst(args, X_train, y_train,X_val, y_val,round, pos_weight = None, plot = False, savepath = None):
    evalset = [(X_train, y_train), (X_val, y_val)]
    if(pos_weight is not None):
        xgbc = xgb.XGBClassifier(tree_method='hist',device = 'cuda', max_depth=args['iclf_max_depth'][round -1], n_estimators=args['iclf_estimators'][round-1], learning_rate=args['iclf_lr'], scale_pos_weight = pos_weight, reg_lambda = 15, alpha = 1, subsample = 0.8, sampling_method='gradient_based', colsample_bytree = 0.8, early_stopping_rounds=100)
    else: 
        xgbc = xgb.XGBClassifier(tree_method='hist',device = 'cuda', max_depth=args['clf_max_depth'][round -1], n_estimators=args['clf_estimators'][round-1], learning_rate=args['clf_lr'], early_stopping_rounds=10)
    xgbc.fit(X_train,y_train, eval_set= evalset, verbose = False )
    if(plot):
        plot_learning_curve(xgbc.evals_result(), eval_metric = args['clf_eval_metric'], path = savepath)
    return xgbc
    