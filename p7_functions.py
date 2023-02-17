import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

#from sklearn.preprocessing import LabelEncoder

import os
from os import listdir
import sys

# Suppress warnings 
#import warnings
#warnings.filterwarnings('ignore')

import shap
import lime 
from lime import lime_tabular

#########################################################################################################################################

def return_size(df):
    """Retourne la taille du dataframe en gigabytes"""
    return round(sys.getsizeof(df) / 1e9, 2)

def convert_types(df, print_info = False):
    
    original_memory = df.memory_usage().sum()
    
    # Itération sur chaque colonne
    for c in df:
        
        # Convertit les ID et booléens en intergers
        if ('SK_ID' in c):
            df[c] = df[c].fillna(0).astype(np.int32)
            
        # Convertit les objets en catégories
        elif (df[c].dtype == 'object') and (df[c].nunique() < df.shape[0]):
            df[c] = df[c].astype('category')
        
        # Booléens mappés sur les entiers
        elif list(df[c].unique()) == [1, 0]:
            df[c] = df[c].astype(bool)
        
        # Float64 à float32
        elif df[c].dtype == float:
            df[c] = df[c].astype(np.float32)
            
        # Int64 à int32
        elif df[c].dtype == int:
            df[c] = df[c].astype(np.int32)
        
    new_memory = df.memory_usage().sum()
    
    if print_info:
        print(f'Utilisation de la mémoire avant traitement : {round(original_memory / 1e9, 2)} gb.')
        print(f'Nouvelle utilisation de la mémoire: {round(new_memory / 1e9, 2)} gb.')
        
    return df

#########################################################################################################################################

# One-hot encoding pour les features catégorielles avec get_dummies
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == object]
    df = pd.get_dummies(df, columns = categorical_columns, 
                       dummy_na = nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

#########################################################################################################################################

def app_train_engineering(df, num_rows = None, nan_as_category = False):

    # Retire les 4 candidatures pour le sexe n'est pas renseigné
    df[df['CODE_GENDER'] != 'XNA']
    
    # Supprime le jour de la semaine de la demande
    df = df.drop(columns = ['WEEKDAY_APPR_PROCESS_START', 'HOUR_APPR_PROCESS_START'])
    
    # Features catégorielles avec enconding binaire
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    
    # One-hot-encoder pour les variables catégoriels
    df, cat_cols = one_hot_encoder(df, nan_as_category)
    
    # Valeurs NaN pour DAYS_EMPLOYED : 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)

    # Imputation de valeurs manquantes par la moyenne
    df['AMT_GOODS_PRICE'] = np.where(df['AMT_GOODS_PRICE'].isna(), df['AMT_GOODS_PRICE'].mean() ,df['AMT_GOODS_PRICE'])
    df['AMT_ANNUITY'] = np.where(df['AMT_ANNUITY'].isna(), df['AMT_ANNUITY'].mean() ,df['AMT_ANNUITY'])
    df['DAYS_EMPLOYED'] = np.where(df['DAYS_EMPLOYED'].isna(), df['DAYS_EMPLOYED'].mean() ,df['DAYS_EMPLOYED'])
    df['CNT_FAM_MEMBERS'] = np.where(df['CNT_FAM_MEMBERS'].isna(),1 ,df['CNT_FAM_MEMBERS'])

    # Remplace les dernières valeurs manquantes par 0 (concerne les variables sur la propriété)
    df.fillna(0, inplace=True)


    # Autres features
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']

    return df

#########################################################################################################################################

def agg_numeric(df, parent_var, df_name):
    """Aggrégation des valeurs numériques dans un dataframe.
    
    Paramètres
    --------
        df (dataframe): 
            dataframe utilisé pour calculer les statistiques
        parent_var (string): 
            Variable de regroupement du dataframe
        df_name (string): 
            variable utilisé pour renommer les colonnes
        
    Return
    --------
        agg (dataframe): 
            Un dataframe avec les statistiques aggrégées pour toutes les colonnes numériques
            Pour chaque variable, la moyenne, min, max et somme sont calculées.
            Les colonnes sont renommées. 
            a dataframe with the statistics aggregated for     
    """
    
    # Supprime les variables d'identification autres que la variable de regroupement
    for col in df:
        if col != parent_var and 'SK_ID' in col:
            df = df.drop(columns = col)
            
    group_ids = df[parent_var]
    numeric_df = df.select_dtypes('number')
    numeric_df[parent_var] = group_ids

    # Group by et calcul les statistiques
    agg = numeric_df.groupby(parent_var).agg(['count', 'mean', 'max', 'min', 'sum']).reset_index()

    # Nouveaux noms de colonnes
    columns = [parent_var]

    # Iterération sur les noms de varibales
    for var in agg.columns.levels[0]:
        # Passe la variable de regroupement
        if var != parent_var:
            # Itération sur les stats
            for stat in agg.columns.levels[1][:-1]:
                # Créé une nouvelle colonne 
                columns.append('%s_%s_%s' % (df_name, var, stat))

    agg.columns = columns
    
    # Supprime les colonnes avec des valeurs redondantes
    _, idx = np.unique(agg, axis = 1, return_index=True)
    agg = agg.iloc[:, idx]
    
    return agg

#########################################################################################################################################

def agg_categorical(df, parent_var, df_name):
    """Calcul le nombre de chaque observation et réalise une normalisation pour
    chaque variable de regroupement. 
    
    Paramètres
    --------
    df : dataframe 
        dataframe utilisé pour le comptage.
        
    parent_var : string
        Variable de regroupement. Le dataframe final aura une ligne
        pour chaque valeur unique de cette variable.
        
    df_name : string
        Renomme les variables

    
    Return
    --------
    categorical : dataframe
        DataFrame avec le compte et le compte normalisé pour chaque catégorie unique de chaque variable catégorielle,
        avec une ligne pour chaque valeur unique de parent_var
        
    """
    
    # Sélectionne les variables catégorielles
    categorical = pd.get_dummies(df.select_dtypes('category'))

    # Garde l'Identifiant
    categorical[parent_var] = df[parent_var]

    # Groupby et calcul la somme et la moyenne
    categorical = categorical.groupby(parent_var).agg(['sum', 'mean'])
    
    column_names = []
    
    # Itération sur les colonnes de niveau 0
    for var in categorical.columns.levels[0]:
        # Itération sur les colonnes de niveau 1 (stats)
        for stat in ['count', 'count_norm']:
            # Renomme
            column_names.append('%s_%s_%s' % (df_name, var, stat))
    
    categorical.columns = column_names
    
    # Supprime les doublons
    _, idx = np.unique(categorical, axis = 1, return_index = True)
    categorical = categorical.iloc[:, idx]
    
    return categorical

#########################################################################################################################################

# Indication sur les données manquantes d'un dataset
def missing_values_table(df):
        # Total des valeurs manquantes
        mis_val = df.isna().sum()
        
        # Pourcentage des NaN
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Création d'un dataframe
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Renomme les colonnes
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Tri les NaN (décroissant)
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print informations sur le df analysé
        print ("Le Dataframe sélectionné a " + str(df.shape[1]) + " colonnes.\n"      
            "Il y a " + str(mis_val_table_ren_columns.shape[0]) +
              " colonnes avec des données manquantes.")
        
        # Retourne le dataframe
        return mis_val_table_ren_columns

#########################################################################################################################################        
def remove_missing_columns(df, threshold = 50):
    # Calcul les stats manquante (%)
    df_miss = pd.DataFrame(df.isnull().sum())
    df_miss['percent'] = 100 * df_miss[0] / len(df)
     
    # liste des valeurs manquantes supérieures aux seuil
    df_miss_columns = list(df_miss.index[df_miss['percent'] > threshold])
    
    # Print
    print('Il y a %d de colonnes avec plus de %d%% de valeurs manquantes.' % (len(df_miss_columns), threshold))
    
    # Drop the df_miss_columns
    df = df.drop(columns = df_miss_columns)
    
    return df  
    
#########################################################################################################################################    
# Barplot sur application_train pour les types de prêts et pourcentage, pour les prêts non remboursés (TARGET = 1)
def plot_stats(app_train, feature, label_rotation = False, horizontal_layout = True):
    temp = app_train[feature].value_counts()
    df = pd.DataFrame({feature: temp.index, 'Nombre de contrats' : temp.values})
    
    # Calcul le pourcentage de TARGET = 1 par catégorie
    cat_perc = app_train[[feature, 'TARGET']].groupby([feature], as_index = False).mean()
    cat_perc.sort_values(by = 'TARGET', ascending = False, inplace = True)
    print(cat_perc)
    
    # Barplot 
    if(horizontal_layout):
        fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize = (12,6))
    else:
        fig, (ax1, ax2) = plt.subplots(nrows = 2, figsize = (12,14))
    #sns.set_color_codes(palette = "rocket")
    sns.set_color_codes("pastel")
    s = sns.barplot(ax = ax1, x = feature, y = "Nombre de contrats", data=df)
    if(label_rotation):
        s.set_xticklabels(s.get_xticklabels(), rotation = 90)
    
    s = sns.barplot(ax = ax2, x = feature, y = 'TARGET',
                    order = cat_perc[feature], data = cat_perc)
    if(label_rotation):
        s.set_xticklabels(s.get_xticklabels(), rotation = 90)
    plt.ylabel('Pourcentage de TARGET = 1 [%]', fontsize = 10)
    plt.tick_params(axis = 'both', which = 'major', labelsize = 10)
    
    total = float(len(app_train))
    for p in ax1.patches:
        percentage = '{:.1f}%'.format(100\
                                      * p.get_height()\
                                      /total)
        x = p.get_x() + p.get_width()
        y = p.get_height()
        ax1.annotate(percentage, (x, y),
                    ha='center', fontsize = 8)
    plt.show();
    
#########################################################################################################################################

# Calcul la feature importance du modèle
def feature_importance_model(clf, X_train):
    df = pd.DataFrame({})
    
    # Feature importance
    if hasattr(clf[-1], "feature_importances_"):
        df['Importance'] = clf[-1].feature_importances_
    elif hasattr(clf[-1], "coef_"):
        df['Importance'] = clf[-1].coef_[0]
    
    # Variable associée
    df['Variable'] = X_train.columns
    
     # Tri par importance
    df = df.sort_values('Importance', ascending = False)
    df.reset_index(drop = True, inplace = True)
    
    # importance cumulée
    df.insert(1, 'Importance_cumulee', np.cumsum(df['Importance']))
    
    return df

#########################################################################################################################################

# Plot la feature importance et la feature importance cumulée
def plot_feature_importance(df, threshold = 0.9):

    # Horizontal bar chart de la feature importance
    plt.figure(figsize = (6, 5))
    ax = plt.subplot()
    
    # Inverse l'index pour commencer par la plus grande feature importance
    ax.barh(list(reversed(list(df.index[:15]))), 
            df['Importance'].head(15), 
            align = 'center', edgecolor = 'k')
       
    # Labels et yticks
    ax.set_yticks(list(reversed(list(df.index[:15]))))
    ax.set_yticklabels(df['Variable'].head(15))
    
    # Plot 
    plt.xlabel('Importance'); plt.title('Feature Importances')
    plt.show()
    
    # Plot l'importance cumulée
    plt.figure(figsize = (5, 4))
    plt.plot(list(range(len(df))), df['Importance_cumulee'], 'r-')
    plt.xlabel('Nombre de variables'); plt.ylabel('Importance cumulée'); 
    plt.title('Feature Importance Cumulée');
    plt.show();
    
    # Nombre de feature pour atteindre le seuil de 90%
    importance_index = np.min(np.where(df['Importance_cumulee'] > threshold))
    print("%d variables requises pour %0.2f d'importance cumulée" % (importance_index + 1, threshold))
    
#########################################################################################################################################   

# Interprétabilité du modèle avec Shap
def local_interpretability_shap(clf, X_test):
    pred = clf.predict_proba(X_test)
    explainer = shap.TreeExplainer(clf[-1])
    #observations = clf[0].transform(X_test)
    observations = X_test
    shap_values = explainer.shap_values(observations)
    return shap_values, observations
    
#########################################################################################################################################

def local_interpretability_lime(clf, X_train, X_test, index):
    explainer = lime_tabular.LimeTabularExplainer(
        training_data = np.array(X_train),
        feature_names = X_train.columns,
        class_names = [0, 1],
        mode = 'classification'
    )
    exp = explainer.explain_instance(
        data_row = X_test.iloc[index], 
        predict_fn = clf.predict_proba
    )

    return exp