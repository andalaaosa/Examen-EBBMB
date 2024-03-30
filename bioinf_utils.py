import pandas as pd
import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve


def matriz_confucion(pred, y):
    mc = confusion_matrix(pred, y)
    plt.figure(figsize=(5,5))
    sns.heatmap(mc, annot=True, fmt=".3f", square = True, cmap = 'Blues_r', cbar=False);
    plt.ylabel('Valores Reales');
    plt.xlabel('Predicciones');
    plt.show()

def scores(pred, y, label=""):
    p = precision_score(pred, y)
    r = recall_score(pred, y)
    f1 = f1_score(pred, y)

    print(f"{label} | Precision: {p*100:.2f} % | Recall: {r*100:.2f}% | f1 Score:{f1*100:.2f}%")
    return p, r, f1

def curva_roc(preds,y,labels):
    for i,p in enumerate(preds):
        fpr, tpr, tr = roc_curve(y[i],p)
        roc_score = roc_auc_score(y[i],p)
        plt.plot(fpr, tpr, linewidth=2, label=f"{labels[i]}-ROC AUC {roc_score:.2f}")
        
    plt.plot([0,1], [0,1], 'k--')#Diagonal
    plt.xlabel('Tasa de falsos positivos')
    plt.ylabel('Sensitividad')
    plt.legend(loc='best')
    plt.grid()
    plt.show()



