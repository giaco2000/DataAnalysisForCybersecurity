# DataPreprocessing.py
import os
import time
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

"""
Informazioni Generali sul File:
- Questo modulo gestisce la fase di pre-elaborazione dei dati nel processo KDD. 

"""

# Funzioni Importate

"""
Funzione che itera attraverso le colonne del DataFrame "x_train" e per ogni colonna estrae il conenuto
assegnandolo alla variabile col_data. Con il metodo col_data.describe() vengono poi calcolate diverse
statistiche come conteggio, media, deviazione standard, minimo, massimo e quartili.
Per ogni colonna vengono quindi stampate le caratteristiche.
""" 


def preElaborationData(X):
    for col in X.columns:
        col_data = X[col]
        col_description = col_data.describe()
        print(f"Statistiche descrittive per la colonna '{col}':\n{col_description}")
        print("\n")

"""
Questa funzione prende in input un DataFrame e restituisce quattro liste contenenti, per ciascuna colonna, 
i seguenti valori statistici: deviazione standard, media, valore minimo e valore massimo.

Per ogni colonna del DataFrame, viene utilizzato il metodo `describe()` per ottenere le statistiche descrittive. I valori di interesse vengono estratti e aggiunti alle rispettive liste, che costituiscono l'output finale della funzione.
"""

def attributeList(X):
    std_values = []
    mean_values = []
    min_values = []
    max_values = []

    for col in X.columns:
        col_data = X[col]
        col_description = col_data.describe()
        std_values.append(col_description["std"])
        mean_values.append(col_description["mean"])
        min_values.append(col_description["min"])
        max_values.append(col_description["max"])

    return std_values, mean_values, min_values, max_values



"""
Questa funzione controlla ciascuna colonna del DataFrame verificando se il valore massimo è uguale al minimo. 
In tal caso, la colonna viene aggiunta a una lista perché priva di variabilità e quindi poco informativa. 
Al termine del controllo su tutte le colonne, quelle presenti nella lista vengono rimosse dal DataFrame.
"""

def removeColumns(X):
    columns_to_remove = []  # Lista per registrare i nomi delle colonne rimosse

    for col in X.columns:
        col_data = X[col]
        col_description = col_data.describe()

        if col_description["min"] == col_description["max"]:
            columns_to_remove.append(col)

    x_cleaned = X.drop(columns=columns_to_remove)  # Rimuovi le colonne inutili

    return x_cleaned, columns_to_remove




"""
Questa funzione verifica la presenza di valori mancanti nel DataFrame. 
Controlla il campo "count" per ciascuna colonna e lo confronta con il numero totale di righe (es. 12000). 
Se il valore è inferiore, significa che ci sono dati mancanti in quella colonna.
"""

def attributiMancanti(X):
    columnsWrongCount = []
    counter = 0
    for col in X.columns:
        col_data = X[col]
        col_description = col_data.describe()
        if col_description["count"] != 12000:
            count = col_description["count"]
            columnsWrongCount.append(col)
            counter = counter + 1
            print(f"La colonna '{col}' ha un count ha degli attributi mancanti, count: '{count}'")
    if counter == 0:
        print("\nNon ci sono colonne con attributi mancanti\n")

    return columnsWrongCount


def DistribuzioneClassi(y, plot=False):
    """
    Calcola e visualizza la distribuzione delle classi nel dataset.
    
    Parametri:
    y : pandas.DataFrame
        DataFrame contenente la colonna 'Label' con le etichette delle classi.
    plot : bool, optional
        Se True, visualizza un istogramma della distribuzione delle classi.
        
    Returns:
    pandas.Series
        Conteggio delle occorrenze per ciascuna classe.
    """
    conteggio_valori = y['Label'].value_counts()
    
    # Se richiesto, visualizza l'istogramma
    if plot:
        plt.figure(figsize=(8, 5))
        ax = conteggio_valori.plot(kind='bar', color=['lightgreen', 'salmon'])
        plt.title('Distribuzione delle classi nel dataset')
        plt.xlabel('Classe')
        plt.ylabel('Numero di campioni')
        plt.xticks(ticks=[0, 1], labels=['Goodware (0)', 'Malware (1)'], rotation=0)
        
        # Aggiungi i valori sopra le barre
        for i, v in enumerate(conteggio_valori):
            ax.text(i, v + 50, str(v), ha='center')
        
        # Calcola e mostra la percentuale di ogni classe
        total = conteggio_valori.sum()
        for i, v in enumerate(conteggio_valori):
            percentage = (v / total) * 100
            ax.text(i, v/2, f"{percentage:.1f}%", ha='center', color='white', fontweight='bold')
        
        # Salva l'istogramma
        plt.savefig('results/class_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    return conteggio_valori


def cleaning_test(test_data, columns_to_remove):

    test_data_cleaned = test_data.drop(columns=columns_to_remove, errors='ignore')
    return test_data_cleaned

def TopAndBottom(rank, top_n, bottom_n):
    """
    Visualizza gli elementi con valori più alti (top) e più bassi (bottom) dalla lista ordinata.
    
    Parametri:
    rank : list
        Lista di tuple (feature, score) ordinata per score decrescente.
    top_n : int
        Numero di elementi top da visualizzare.
    bottom_n : int
        Numero di elementi bottom da visualizzare. Se 0, mostra gli ultimi 5 elementi con valore non zero.
    """
    # Verifica che rank non sia vuoto
    if not rank:
        print("La lista rank è vuota.")
        return
    
    # Limita top_n e bottom_n alla lunghezza di rank
    top_n = min(top_n, len(rank))
    
    # Estrai i primi top_n elementi dalla lista ordinata
    top_elements = rank[:top_n]

    # Se bottom_n è 0, estraggo solo gli ultimi 5 elementi con valore diverso da zero
    if bottom_n == 0:
        # Creo una lista di elementi con valore non zero, poi prendo gli ultimi 5
        non_zero_elements = [element for element in rank if element[1] != 0]
        bottom_elements = non_zero_elements[-5:] if len(non_zero_elements) > 5 else non_zero_elements
    else:
        # Limito bottom_n alla lunghezza di rank
        bottom_n = min(bottom_n, len(rank))
        # Estraggo gli ultimi bottom_n elementi dalla lista ordinata
        bottom_elements = rank[-bottom_n:]

    print(f"\nTop {top_n} elements:")
    for element in top_elements:
        print(f"{element[0]}: {element[1]}")

    print(f"\nBottom {len(bottom_elements)} elements:")
    for element in bottom_elements:
        print(f"{element[0]}: {element[1]}")

"""
Funzione che crea boxplot per visualizzare la distribuzione di ciascuna variabile indipendente
rispetto alle classi (0 per goodware, 1 per malware). La funzione utilizza il metodo boxplot
di pandas con il parametro 'by' per raggruppare i dati in base alla colonna 'Label' del DataFrame Y.

Per ogni colonna di X, viene creato un boxplot che mostra come i valori della variabile sono
distribuiti tra le due classi, permettendo di identificare visivamente quali variabili
mostrano una separazione più chiara tra malware e goodware.

I boxplot generati vengono salvati nella cartella "BoxPlot" nella directory principale del progetto.
"""
def preBoxPlotAnalysisData(X, Y):
    # Imposto il backend 'Agg' per evitare problemi con l'interfaccia grafica
    matplotlib.use('Agg')
    
    # Creo la cartella BoxPlot se non esiste
    boxplot_dir = "BoxPlot"
    if not os.path.exists(boxplot_dir):
        os.makedirs(boxplot_dir)
        print(f"Cartella '{boxplot_dir}' creata con successo.")
    
    total_variables = len(X.columns)
    print(f"Creazione di boxplot per tutte le {total_variables} variabili...")
    
    # Creo un nuovo DataFrame che combina X e Y
    data = X.copy()
    data['Label'] = Y['Label'].values
    
    # Genero boxplot per tutte le colonne
    start_time = time.time()
    for i, col in enumerate(X.columns):
        # Creo una nuova figura per ogni boxplot
        plt.figure(figsize=(6, 4), dpi=80)
        
        # Genero il boxplot
        ax = data.boxplot(column=col, by='Label')
        plt.title(f'Boxplot {col}', fontsize=10)
        plt.suptitle('')  # Rimuove il titolo automatico
        
        # Salvo il boxplot
        save_path = os.path.join(boxplot_dir, f'boxplot_{col}.png')
        plt.savefig(save_path)
        plt.close()
        
        # Mostro il progresso ogni 100 boxplot o quando arriviamo al 10%, 20%, ecc.
        if (i + 1) % 100 == 0 or (i + 1) / total_variables * 10 % 1 == 0:
            elapsed = time.time() - start_time
            perc_completato = (i + 1) / total_variables * 100
            stimato_totale = elapsed / (i + 1) * total_variables
            rimanente = stimato_totale - elapsed
            
            print(f"Generati {i + 1}/{total_variables} boxplot ({perc_completato:.1f}%)...")
            print(f"Tempo trascorso: {elapsed/60:.1f} minuti, Tempo stimato rimanente: {rimanente/60:.1f} minuti")
    
    total_time = time.time() - start_time
    print(f"Creazione dei boxplot completata in {total_time/60:.1f} minuti.")
    print(f"I file sono stati salvati in '{boxplot_dir}'.")


"""
Funzione che calcola e ordina l'informazione mutua tra le variabili indipendenti in X
e la variabile dipendente (classe) in Y.

L'informazione mutua è una misura statistica che quantifica quanto una variabile contribuisce
alla predizione della classe. Valori più alti indicano che la variabile contiene informazioni
più rilevanti per la classificazione.

La funzione restituisce una lista ordinata di tuple (nome_variabile, valore_informazione_mutua),
dove le variabili con informazione mutua più alta appaiono per prime.
"""
def mutualInfoRank(X, Y):
    print("Computing mutual info ranking...")
    
    
    # Imposta un seed per la riproducibilità
    seed = 42
    np.random.seed(seed)
    
    # Ottiengo la lista dei nomi delle variabili indipendenti
    independentList = list(X.columns.values)
    
    # Calcolo l'informazione mutua tra ogni variabile e la classe
    res = dict(zip(independentList,
        mutual_info_classif(X, np.ravel(Y), discrete_features=False, random_state=seed)
    ))
    
    # Ordino le variabili in base al valore di informazione mutua (ordine decrescente)
    sorted_x = sorted(res.items(), key=lambda kv: kv[1], reverse=True)
    
    print("Computing mutual info ranking...completed")
    return sorted_x
