import torch.nn as nn
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import sklearn
from lifelines import CoxPHFitter
from sksurv import metrics
from . import Model
import torch
import pandas as pd
import numpy as np
import os

def evaluate_5CV(test_dataloader, index, folds, folder):
    cv_test_index = index
    cv_predicted_df_dict = {}

    for fold in folds:
        cv_test_predicted_df = pd.DataFrame(np.zeros((len(cv_test_index),3)), index=cv_test_index, columns=['H&E_score','gold','layers'])
        file = np.sort(os.listdir(f"{folder}/{str(fold)}"))[-1] # Get the last model file in the folder
        print(f"Fold {fold} model:", file)
        model_path = f"{folder}/{str(fold)}/{file}"
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model_test = Model.Net()
        model_test.load_state_dict(checkpoint['model_state_dict'])
        model_test.cuda()
        criterion = nn.CrossEntropyLoss()
        epoch_test = checkpoint['epoch']
        loss_test = checkpoint['loss']
        model_test.eval()

        loss_epoch_test=[]
        y_proba = []
        y_layers = []
        y_gold = []
        y_pred = []
        with torch.no_grad():
            for b, (X, y) in enumerate(test_dataloader):
                outputs, a1, a2 = model_test(X.cuda())
                _, preds = torch.max(outputs, 1)
                y_proba += torch.flatten(outputs[:,1]).cpu().tolist()
                y_gold += y.data.cpu().tolist()
                y_pred += preds.cpu().tolist()
    #             loss = criterion(outputs.float(), torch.tensor([[1,0] if label==0 else [0,1] for label in y_test]).to(device).float())
    #             loss_epoch_test.append(loss.item())

            auc=roc_auc_score(y_gold,y_proba)
            print(f"Fold {fold} AUROC: {auc}")
        cv_test_predicted_df.loc[cv_test_index,'H&E_score'] = y_proba
        cv_test_predicted_df.loc[cv_test_index,'gold'] = y_gold
#         cv_test_predicted_df.loc[cv_test_index,'layers'] = y_layers
        cv_predicted_df_dict[fold] = cv_test_predicted_df
#         result_score_pfs = manifest_pfs.join(cv_test_predicted_df, how='inner')[['case.id','PFS_STATUS','PFS_MONTHS','y','score','gold']].groupby('case.id').mean()
#         result_score_pfs = result_score_pfs[result_score_pfs['PFS_STATUS']==1]
#         result_score_pfs

    # Average the scores of the 5 folds
    temp = cv_test_predicted_df[["H&E_score"]]
    for fold in folds:
        temp["H&E_score"] += cv_predicted_df_dict[fold]["H&E_score"]
    cv_test_predicted_df["H&E_score"] = temp["H&E_score"]/len(folds)
    
    return cv_test_predicted_df

def roc_curve_multi(df_with_score_and_survival, survival_duration_col, survival_status_col, score_col, prediction_durations, lte_or_ste_str):
    fig, ax = plt.subplots(figsize=(8,8))
    for d in prediction_durations:
        if lte_or_ste_str=='<=':
            gold = df_with_score_and_survival[survival_duration_col]<=d
            if score_col == 'Cox_score' or score_col == 'age_at_initial_pathologic_diagnosis':
                scores = df_with_score_and_survival[score_col].astype(float)
            else:
                scores = -1*df_with_score_and_survival[score_col].astype(float)
        elif lte_or_ste_str=='>=':
            gold = df_with_score_and_survival[survival_duration_col]>=d
            if score_col == 'Cox_score' or score_col == 'age_at_initial_pathologic_diagnosis':
                scores = -1*df_with_score_and_survival[score_col].astype(float)
            else:
                scores = df_with_score_and_survival[score_col].astype(float)
        elif lte_or_ste_str=='<':
            gold = df_with_score_and_survival[survival_duration_col]<d
            if score_col == 'Cox_score' or score_col == 'age_at_initial_pathologic_diagnosis':
                scores = df_with_score_and_survival[score_col].astype(float)
            else:
                scores = -1*df_with_score_and_survival[score_col].astype(float)
        elif lte_or_ste_str=='>':
            gold = df_with_score_and_survival[survival_duration_col]>d
            if score_col == 'Cox_score' or score_col == 'age_at_initial_pathologic_diagnosis':
                scores = -1*df_with_score_and_survival[score_col].astype(float)
            else:
                scores = df_with_score_and_survival[score_col].astype(float)
    
    
        
        fpr, tpr, _ = sklearn.metrics.roc_curve(gold, scores)
        auc = round(sklearn.metrics.roc_auc_score(gold, scores), 4)
        ax.plot(fpr,tpr,label=f"{survival_duration_col}{lte_or_ste_str}{d} AUC="+str(auc))
    
#     RocCurveDisplay.from_predictions(
#     gold,
#     scores,
#     name=f"{survival_duration_col}{lte_or_ste_str}{prediction_duration}",
#     color="darkorange",
#     ax=ax
#     )
#     # plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
#     plt.axis("square")
#     plt.xlabel("False Positive Rate")
#     plt.ylabel("True Positive Rate")
#     # plt.title("Receiver Operating Characteristic")
    if score_col == 'H&E_score':
        plt.title("H&E_score")
    elif score_col == 'age_at_initial_pathologic_diagnosis':
        plt.title("Diagnostic_age")
    elif score_col == 'Cox_score':
        plt.title("Cox_score")
    plt.legend()
    plt.show()
    
def correlation_plot(df_with_score_and_survival, survival_duration_col, score_col):  
    plt.scatter(df_with_score_and_survival[survival_duration_col], df_with_score_and_survival[score_col])
    plt.xlabel(survival_duration_col)
    plt.ylabel(score_col)
    r, p = spearmanr(df_with_score_and_survival[survival_duration_col], df_with_score_and_survival[score_col])
    plt.title(f"Spearman R: {r}, p-value: {p}")
    plt.show()

def coxph_multi(df_with_score_and_survival, survival_duration_col, survival_status_col, included_col):
    from lifelines import CoxPHFitter
    cph = CoxPHFitter()
    cph.fit(df_with_score_and_survival[included_col], duration_col=survival_duration_col, event_col=survival_status_col)
    cph.print_summary()
    print(cph.summary)
    cph.plot()
    plt.show()
    return cph
    
def coxph_uni(df_with_score_and_survival, survival_duration_col, survival_status_col, included_col):
    from lifelines import CoxPHFitter
    cph = CoxPHFitter()
    cph.fit(df_with_score_and_survival[included_col], duration_col=survival_duration_col, event_col=survival_status_col)
    cph.print_summary()
    print(cph.summary)
    cph.plot()
    plt.show()
    
    
def model_performance(df_with_score_and_pfs, df_with_score_and_os, cox_pfs=None, cox_os=None):
    correlation_plot(df_with_score_and_pfs, "PFS_MONTHS", "H&E_score")
    correlation_plot(df_with_score_and_os, "OS_MONTHS", "H&E_score")
    correlation_plot(df_with_score_and_pfs, "PFS_MONTHS", "age_at_initial_pathologic_diagnosis")
    correlation_plot(df_with_score_and_os, "OS_MONTHS", "age_at_initial_pathologic_diagnosis")
    roc_curve_multi(df_with_score_and_pfs, "PFS_MONTHS", "PFS_STATUS", "H&E_score", [1,3,6,12], "<=")
#     roc_curve_multi(df_with_score_and_pfs, "PFS_MONTHS", "PFS_STATUS", "H&E_score", [24,30,36], ">=")
    roc_curve_multi(df_with_score_and_os, "OS_MONTHS", "OS_STATUS", "H&E_score", [18,24,30], "<=")
    roc_curve_multi(df_with_score_and_os, "OS_MONTHS", "OS_STATUS", "H&E_score", [36,42,60,72], ">=")
    roc_curve_multi(df_with_score_and_pfs, "PFS_MONTHS", "PFS_STATUS", "age_at_initial_pathologic_diagnosis", [1,3,6,12], "<=")
#     roc_curve_multi(df_with_score_and_pfs, "PFS_MONTHS", "PFS_STATUS", "age_at_initial_pathologic_diagnosis", [24,30,36], ">=")
    roc_curve_multi(df_with_score_and_os, "OS_MONTHS", "OS_STATUS", "age_at_initial_pathologic_diagnosis", [18,24,30], "<=")
    roc_curve_multi(df_with_score_and_os, "OS_MONTHS", "OS_STATUS", "age_at_initial_pathologic_diagnosis", [36,42,60,72], ">=")

    if cox_os is None:
        cox_os = coxph_multi(df_with_score_and_os, 'OS_MONTHS','OS_STATUS', ['OS_MONTHS','OS_STATUS','H&E_score','age_at_initial_pathologic_diagnosis'])
#         reg_os = LinearRegression().fit(X_os, y_os)
    if cox_pfs is None:
        cox_pfs = coxph_multi(df_with_score_and_pfs, 'PFS_MONTHS','PFS_STATUS', ['PFS_MONTHS','PFS_STATUS','H&E_score','age_at_initial_pathologic_diagnosis'])
#         reg_pfs = LinearRegression().fit(X_pfs, y_pfs)
    df_with_score_and_pfs['Cox_score'] = cox_pfs.predict_partial_hazard(df_with_score_and_pfs[['H&E_score','age_at_initial_pathologic_diagnosis']])
    df_with_score_and_os['Cox_score'] = cox_os.predict_partial_hazard(df_with_score_and_os[['H&E_score','age_at_initial_pathologic_diagnosis']])
    correlation_plot(df_with_score_and_pfs, "PFS_MONTHS", "Cox_score")
    correlation_plot(df_with_score_and_os, "OS_MONTHS", "Cox_score")
    roc_curve_multi(df_with_score_and_pfs, "PFS_MONTHS", "PFS_STATUS", "Cox_score", [1,3,6,12], "<=")
#     roc_curve_multi(df_with_score_and_pfs, "PFS_MONTHS", "PFS_STATUS", "Cox_score", [24,30,36], ">=")
    roc_curve_multi(df_with_score_and_os, "OS_MONTHS", "OS_STATUS", "Cox_score", [18,24,30], "<=")
    roc_curve_multi(df_with_score_and_os, "OS_MONTHS", "OS_STATUS", "Cox_score", [36,42,60,72], ">=")
    coxph_multi(df_with_score_and_pfs, 'PFS_MONTHS', 'PFS_STATUS', ['PFS_MONTHS', 'PFS_STATUS',"age_at_initial_pathologic_diagnosis","H&E_score"])
    coxph_multi(df_with_score_and_os, 'OS_MONTHS', 'OS_STATUS', ['OS_MONTHS','OS_STATUS',"age_at_initial_pathologic_diagnosis","H&E_score"])
    coxph_uni(df_with_score_and_os, 'OS_MONTHS','OS_STATUS', ['OS_MONTHS','OS_STATUS','H&E_score'])
    coxph_uni(df_with_score_and_pfs, 'PFS_MONTHS','PFS_STATUS', ['PFS_MONTHS','PFS_STATUS','H&E_score'])
    coxph_uni(df_with_score_and_os, 'OS_MONTHS','OS_STATUS', ['OS_MONTHS','OS_STATUS','age_at_initial_pathologic_diagnosis'])
    coxph_uni(df_with_score_and_pfs, 'PFS_MONTHS','PFS_STATUS', ['PFS_MONTHS','PFS_STATUS','age_at_initial_pathologic_diagnosis'])
    # print('PFS c-index',metrics.concordance_index_censored(df_with_score_and_pfs['PFS_STATUS'].apply(lambda x: True if x==1 else False).values,df_with_score_and_pfs['PFS_MONTHS'].values,(df_with_score_and_pfs['Cox_score']).values))
    # print('OS c-index',metrics.concordance_index_censored(df_with_score_and_os['OS_STATUS'].apply(lambda x: True if x==1 else False).values,df_with_score_and_os['OS_MONTHS'].values,(df_with_score_and_os['Cox_score']).values))
    return cox_os, cox_pfs


from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

def kmplot(T, E, L, title, xlabel, ylabel):
    kmf = KaplanMeierFitter()
    for i,t in enumerate(T):
        kmf.fit(t, E[i], label=L[i])
        ax = kmf.plot_survival_function()
    plt.title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()

def kmplots(df_os, df_pfs, score_col):
    if score_col == 'age_at_initial_pathologic_diagnosis':
        title = 'Age'
    else:
        title = score_col
    t_os=np.quantile(df_os[score_col],0.5)
    t_pfs=np.quantile(df_pfs[score_col],0.5)
    t1_os=np.quantile(df_os[score_col],0.3)
    t2_os=np.quantile(df_os[score_col],0.7)
    t1_pfs=np.quantile(df_pfs[score_col],0.3)
    t2_pfs=np.quantile(df_pfs[score_col],0.7)
    print(t_pfs)
    
    
    T1 = df_os[df_os[score_col]>t_os]['OS_MONTHS']
    T2 = df_os[df_os[score_col]<=t_os]['OS_MONTHS']
    E1 = df_os[df_os[score_col]>t_os]['OS_STATUS']
    E2 = df_os[df_os[score_col]<=t_os]['OS_STATUS']
    results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
    kmplot([T1,T2], [E1,E2], [f"High {title}", f"Low {title}"], "OS, $\it{p}$-value:" +f"{results.p_value:.3e}","OS, months","Proportion survival")

    kmf = KaplanMeierFitter()
    T1 = df_pfs[df_pfs[score_col]>t_pfs]['PFS_MONTHS']
    T2 = df_pfs[df_pfs[score_col]<=t_pfs]['PFS_MONTHS']
    E1 = df_pfs[df_pfs[score_col]>t_pfs]['PFS_STATUS']
    E2 = df_pfs[df_pfs[score_col]<=t_pfs]['PFS_STATUS']
    results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
    kmplot([T1,T2], [E1,E2], [f"High {title}", f"Low {title}"], "PFS, $\it{p}$-value:"+ f"{results.p_value:.3e}","PFS, months","Proportion survival")

    T1 = df_pfs[df_pfs[score_col]<=t1_pfs]['PFS_MONTHS']
    T2 = df_pfs[(df_pfs[score_col]>t1_pfs)&(df_pfs[score_col]<=t2_pfs)]['PFS_MONTHS']
    T3 = df_pfs[df_pfs[score_col]>t2_pfs]['PFS_MONTHS']
    E1 = df_pfs[df_pfs[score_col]<=t1_pfs]['PFS_STATUS']
    E2 = df_pfs[(df_pfs[score_col]>t1_pfs)&(df_pfs[score_col]<=t2_pfs)]['PFS_STATUS']
    E3 = df_pfs[df_pfs[score_col]>t2_pfs]['PFS_STATUS']
    results = logrank_test(T1, T3, event_observed_A=E1, event_observed_B=E3)
    kmplot([T1,T2,T3], [E1,E2,E3], [f"Q1 {title}",f"Q2 {title}",f"Q3 {title}"], "PFS, $\it{p}$-value (Q1 vs Q3):" + f"{results.p_value:.3e}","PFS, months","Proportion survival")

    T1 = df_os[df_os[score_col]<=t1_os]['OS_MONTHS']
    T2 = df_os[(df_os[score_col]>t1_os)&(df_os[score_col]<=t2_os)]['OS_MONTHS']
    T3 = df_os[df_os[score_col]>t2_os]['OS_MONTHS']
    E1 = df_os[df_os[score_col]<=t1_os]['OS_STATUS']
    E2 = df_os[(df_os[score_col]>t1_os)&(df_os[score_col]<=t2_os)]['OS_STATUS']
    E3 = df_os[df_os[score_col]>t2_os]['OS_STATUS']
    results = logrank_test(T1, T3, event_observed_A=E1, event_observed_B=E3)
    kmplot([T1,T2,T3], [E1,E2,E3], [f"Q1 {title}",f"Q2 {title}",f"Q3 {title}"], "OS, $\it{p}$-value (Q1 vs Q3):" + f"{results.p_value:.3e}","OS, months","Proportion survival")

import torchvision.utils as utils
import cv2

def visualize_attention(I_train,a,up_factor,no_attention=False):
    import torch.nn.functional as F
    img = I_train.permute((1,2,0)).cpu().numpy()
    # compute the heatmap
#     if up_factor > 1:
    a = F.interpolate(a, scale_factor=up_factor, mode='bilinear', align_corners=False)
    attn = utils.make_grid(a, nrow=6, normalize=True, scale_each=True)
    attn = attn.permute((1,2,0)).mul(255).byte().cpu().numpy()
    attn = cv2.applyColorMap(attn, cv2.COLORMAP_JET)
    attn = cv2.cvtColor(attn, cv2.COLOR_BGR2RGB)
    attn = np.float32(attn) / 255
    # add the heatmap to the image
#     img=cv2.resize(img,(176,60))
    if no_attention:
        return torch.from_numpy(img)
    else:
        vis = 0.6 * img + 0.4 * attn
        return torch.from_numpy(vis)

def attention_visualization(model_path, dataloader):
    import torch.nn.functional as F
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model_test = Model.Net().to(device)
    model_test.load_state_dict(checkpoint['model_state_dict'])
    model_test.cuda()
    criterion = nn.CrossEntropyLoss()
    epoch_test = checkpoint['epoch']
    loss_test = checkpoint['loss']
    model_test.eval()

    loss_epoch_test=[]
    y_proba = []
    y_layers = []
    y_gold = []
    y_pred = []
    with torch.no_grad():
        for b, (X, y) in enumerate(dataloader):
            I_train = utils.make_grid(X[0:6,:,:,:], nrow=len(X), normalize=True, scale_each=True)
            outputs, a1, a2 = model_test(X.cuda())
            orig=visualize_attention(I_train,a1,up_factor=2,no_attention=True)
    #         first=visualize_attention(I_train,a1,up_factor=4,no_attention=False)
            first=visualize_attention(I_train,a2,up_factor=16,no_attention=False)
            fig, (ax1, ax2) = plt.subplots(2, 1,figsize=(10, 10))
            ax1.imshow(orig)
            ax2.imshow(first)
            ax1.title.set_text('Input')
            ax2.title.set_text('Module2')
            plt.show()
            break