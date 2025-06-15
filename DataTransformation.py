# DataTransformation.py
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

"""
Funzione che seleziona le caratteristiche (feature) più importanti in base all'informazione mutua.
Restituisce una lista con i nomi delle caratteristiche che hanno un valore di informazione mutua
maggiore o uguale alla soglia specificata.

Parametri:
rank (list): Lista di tuple (nome_feature, valore_informazione_mutua) in ordine decrescente
threshold (float): Soglia minima per l'informazione mutua 

Restituisce:
list: Lista di nomi delle caratteristiche selezionate
"""
def topFeatureSelect(rank, threshold):
    # Inizializzo una lista vuota per le caratteristiche selezionate
    selected_features = []
    
    # Itero attraverso la lista di tuple ordinata
    for feature, mi_value in rank:
        # Verifico se il valore di informazione mutua è maggiore o uguale alla soglia
        if mi_value >= threshold:
            selected_features.append(feature)
        else:
            # Dato che la lista è ordinata in modo decrescente, si può interrompere
            # il ciclo non appena trovo un valore inferiore alla soglia
            break
    
    print(f"{len(selected_features)} features selezionate con informazione mutua >= {threshold}")
    return selected_features


"""
Funzione che applica l'analisi delle componenti principali (PCA) ai dati.
La funzione calcola le componenti principali, crea i nomi delle componenti
e restituisce l'oggetto PCA, i nomi delle componenti e la varianza spiegata.

Parametri:
X (DataFrame): DataFrame contenente le variabili indipendenti

Restituisce:
tuple: (oggetto PCA, lista dei nomi delle componenti, lista delle varianze spiegate)
"""
def pca(X):
    
    # Inizializzo l'oggetto PCA senza riduzione di dimensionalità (mantieni tutte le componenti)
    pca_obj = PCA()
    
    # Adatto il modello PCA ai dati
    pca_obj.fit(X)
    
    # Creo una lista di nomi per le componenti principali
    pc_names = [f"PC{i+1}" for i in range(pca_obj.n_components_)]
    
    # Ottiengo la varianza spiegata da ciascuna componente
    explained_variance = pca_obj.explained_variance_ratio_
    
    print(f"Analisi PCA completata: {len(pc_names)} componenti principali ottenute")
    return pca_obj, pc_names, explained_variance


"""
Funzione che applica la trasformazione PCA ai dati.
La funzione trasforma il DataFrame originale in un nuovo DataFrame
con le componenti principali.

Parametri:
X (DataFrame): DataFrame da trasformare
pca_obj (PCA): Oggetto PCA precedentemente addestrato
pc_names (list): Lista dei nomi delle componenti principali

Restituisce:
DataFrame: Nuovo DataFrame con le componenti principali come colonne
"""
def applyPCA(X, pca_obj, pc_names):
   
    
    # Applico la trasformazione PCA ai dati
    X_transformed = pca_obj.transform(X)
    
    # Creo un nuovo DataFrame con le componenti principali
    pca_df = pd.DataFrame(X_transformed, columns=pc_names)
    
    print(f"Trasformazione PCA applicata: DataFrame con {pca_df.shape[1]} componenti principali")
    return pca_df


"""
Funzione che determina il numero di componenti principali necessarie
per raggiungere una determinata soglia di varianza spiegata cumulativa.

Parametri:
explained_variance (array): Array o lista delle varianze spiegate per ciascuna componente principale
threshold (float): Soglia di varianza spiegata da raggiungere (es. 0.95 = 95%)

Restituisce:
int: Numero di componenti principali da mantenere
"""
def NumberOfTopPCSelect(explained_variance, threshold):
    
    # Calcolo la varianza spiegata cumulativa
    cumulative_variance = np.cumsum(explained_variance)
    
    # Trovo il numero di componenti necessarie per raggiungere la soglia
    for i, cum_var in enumerate(cumulative_variance):
        if cum_var >= threshold:
            # Restituisco l'indice + 1 perché gli indici partono da 0
            return i + 1
    
    # Se tutte le componenti non sono sufficienti per raggiungere la soglia,
    # restituisco il numero totale di componenti
    return len(explained_variance)