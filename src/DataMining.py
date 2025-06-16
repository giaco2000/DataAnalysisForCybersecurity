# DataMining.py
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.metrics import f1_score
import numpy as np
import pickle
import os
import matplotlib.backends.backend_pdf as pdf_backend
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from DataTransformation import topFeatureSelect, NumberOfTopPCSelect
"""
    Implementa la Stratified K-fold).
    
    Questa tecnica divide il dataset in k partizioni (fold) mantenendo la stessa 
    proporzione delle classi presente nei dati originali, garantendo che ogni fold 
    sia rappresentativo del dataset completo.
    
    Parametri:
    ----------
    X : pandas.DataFrame
        DataFrame contenente le variabili indipendenti.
    Y : pandas.DataFrame
        DataFrame contenente la variabile dipendente (target/label).
    folds : int
        Numero di fold in cui dividere il dataset.
    seed : int
        Seme per la generazione dei numeri casuali, garantisce la riproducibilità.
        
    Returns:
    --------
    tuple
        Tuple di liste (ListXTrain, ListXTest, ListYTrain, ListYTest) dove:
        - ListXTrain: lista di DataFrame contenenti i dati di training per ogni fold
        - ListXTest: lista di DataFrame contenenti i dati di test per ogni fold
        - ListYTrain: lista di DataFrame contenenti le etichette di training per ogni fold
        - ListYTest: lista di DataFrame contenenti le etichette di test per ogni fold
        
    Note:
    -----
    La funzione utilizza StratifiedKFold di scikit-learn per garantire che 
    la distribuzione delle classi sia preservata in ciascun fold.
    """
def stratifiedKfold(X, Y, folds, seed):
    # Inizializzo le liste per memorizzare i set di training e test per ogni fold
    ListXTrain = []
    ListXTest = []
    ListYTrain = []
    ListYTest = []
    
    # Creo l'oggetto StratifiedKFold
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    
    # Ottiengo gli indici per ogni split train-test
    for train_index, test_index in skf.split(X, np.ravel(Y)):
        # Divido X in train e test per il fold corrente
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        # Divido Y in train e test per il fold corrente
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        
        # Aggiungo i dataset alle rispettive liste
        ListXTrain.append(X_train)
        ListXTest.append(X_test)
        ListYTrain.append(Y_train)
        ListYTest.append(Y_test)

    # Sposto il messaggio fuori dal ciclo.
    print(f"Creati {folds} fold per la cross-validation")
    return ListXTrain, ListXTest, ListYTrain, ListYTest


"""
    Crea e addestra un albero di decisione (Decision Tree) per la classificazione.
    
    Parametri:
    X : pandas.DataFrame
        DataFrame contenente le variabili indipendenti per l'addestramento.
    y : pandas.DataFrame o array-like
        Variabile dipendente (target/label) per l'addestramento.
    c : str
        Criterio da utilizzare per la valutazione delle divisioni nell'albero.
        Valori possibili: 'gini' (impurità Gini) o 'entropy' (entropia).
        
    Returns:
    sklearn.tree.DecisionTreeClassifier
        Crea un Oggetto: ovvero l'albero di decisione addestrato.
        
    Note:
    L'albero viene configurato con un valore di min_samples_split pari a 500,
    il che significa che un nodo sarà suddiviso solo se contiene almeno 500 campioni.
    Questo aiuta a prevenire l'overfitting creando un albero più generalizzato.
    
    La funzione stampa anche informazioni sull'albero creato come il numero di nodi,
    il numero di foglie e la profondità massima.
    """
def decisionTreeLearner(X, y, c):
    seed = 42
    
    # Creo l'albero di decisione con i parametri specificati
    dt = DecisionTreeClassifier(criterion=c, min_samples_split=500, random_state=seed)
    
    # Addestro l'albero sui dati forniti
    dt.fit(X, np.ravel(y))
    
    # Ottenngo informazioni sull'albero
    n_nodes = dt.tree_.node_count
    n_leaves = dt.tree_.n_leaves
    
    print(f"Albero di decisione creato con criterio '{c}':")
    print(f"- Numero di nodi: {n_nodes}")
    print(f"- Numero di foglie: {n_leaves}")
    print(f"- Profondità massima: {dt.tree_.max_depth}")
    
    return dt


"""
Funzione che visualizza graficamente un albero di decisione e lo salva come immagine.

Parametri:
T (DecisionTreeClassifier): Albero di decisione da visualizzare
feature_names (list, optional): Lista dei nomi delle feature. Default: None
max_depth (int, optional): Profondità massima dell'albero da visualizzare. Default: None
"""
def showTree(T, feature_names=None, max_depth=None, model_name="default"):
    
    # Creo una directory per le immagini degli alberi se non esiste
    trees_dir = "trees_images"
    if not os.path.exists(trees_dir):
        os.makedirs(trees_dir)
        print(f"Cartella '{trees_dir}' creata per le immagini degli alberi")

    # Creo una figura di dimensioni maggiori per visualizzare meglio l'albero
    fig = plt.figure(figsize=(15, 10))
    
    # Visualizzo l'albero con opzioni migliorate
    plot_tree(T, 
              feature_names=feature_names, 
              filled=True, 
              rounded=True, 
              class_names=["Goodware", "Malware"])
    
    # Salvo l'albero come immagine
    depth_str = f"_depth{max_depth}" if max_depth else ""
    base_filename = f"decision_tree_{model_name}{depth_str}"
    pdf_path = os.path.join(trees_dir, f"{base_filename}.pdf")
    png_path = os.path.join(trees_dir, f"{base_filename}.png")

    # Creo un oggetto PdfPages
    pdf_pages = pdf_backend.PdfPages(pdf_path)
    # Salvo la figura corrente nel PDF
    pdf_pages.savefig(fig, bbox_inches='tight')
    # Chiude il PDF
    pdf_pages.close()
    
    # Salvo anche come PNG per compatibilità
    plt.savefig(png_path, dpi=1000, bbox_inches='tight')
    
    # VIene mostrato l'albero
    plt.show()
    
    print(f"Albero di decisione visualizzato e salvato come:")
    print(f"- PDF: '{pdf_path}'")
    print(f"- PNG: '{png_path}'")



"""
    Parametri:
    ListXTrain : list
        Lista di DataFrame contenenti i dati di training per ogni fold.
    ListYTrain : list
        Lista di DataFrame contenenti le etichette di training per ogni fold.
    ListXTest : list
        Lista di DataFrame contenenti i dati di test per ogni fold.
    ListYTest : list
        Lista di DataFrame contenenti le etichette di test per ogni fold.
    rank : list
        Lista di tuple (feature, score) ordinata per informazione mutua decrescente.
    min_t : float
        Valore minimo della soglia di informazione mutua da considerare.
    max_t : float
        Valore massimo della soglia di informazione mutua da considerare.
    step : float
        Incremento della soglia di informazione mutua ad ogni iterazione.
    save_path : str
        Percorso dove salvare i risultati della configurazione ottimale.
        
    Return:
    dict
        Dizionario contenente la configurazione ottimale con chiavi:
        - 'best_criterion': criterio di divisione (gini/entropy)
        - 'best_TH': soglia ottimale di informazione mutua
        - 'bestN': lista delle feature selezionate
        - 'best_fscore': F1-score medio ottenuto con la configurazione ottimale
    """
def determineDecisionTreekFoldConfiguration(ListXTrain, ListYTrain, ListXTest, ListYTest, rank, min_t, max_t, 
                                            step,save_path):
    """
    Determina la configurazione ottimale per un albero di decisione attraverso K-fold.
    Esplora sistematicamente diverse combinazioni di criteri e soglie di informazione mutua.
    """
    # Verifico se esiste già un file con i risultati per evitare di ricalcolare
    if save_path and os.path.exists(save_path):
        print(f"File di configurazione trovato: '{save_path}'")
        with open(save_path, 'rb') as file:
            result = pickle.load(file)
        return result  # Restituisce direttamente i risultati salvati

    print("Nessun file già salvato, eseguo il metodo.")
    
    # Inizializzazione delle variabili per tenere traccia della migliore configurazione
    best_fscore = -1  # Inizializzato a -1 perché qualsiasi F-score sarà maggiore
    best_criterion = None  # Criterio migliore (gini o entropy) Misura la probabilità di classificare erroneamente un elemento se selezionato casualmente
    best_TH = None  # Soglia migliore di informazione mutua
    bestN = None  # Migliori feature selezionate
    
    # Lista dei criteri da testare
    criterion = ['gini', 'entropy']  # I due criteri possibili per l'albero di decisione

    # Esploro tutte le combinazioni di criteri e soglie
    for criteria in criterion:  # Itero prima sui criteri
        for thre in np.arange(min_t, max_t, step):  # Poi su diverse soglie di informazione mutua
            fscores = []  # Lista per memorizzare gli F-score per ogni fold
            
            # Seleziono le feature con informazione mutua >= soglia corrente
            selectedFeatures = topFeatureSelect(rank, thre)
            
            # Verifico che ci siano feature selezionate
            if len(selectedFeatures) > 0:
                
                # Eseguo la validazione incrociata su tutti i fold
                for i in range(len(ListXTrain)):
                    # Seleziono solo le colonne corrispondenti alle feature scelte
                    x_train_feature_selected = ListXTrain[i].loc[:, selectedFeatures]
                    
                    # Addestro un albero di decisione con il criterio corrente
                    dtl = decisionTreeLearner(x_train_feature_selected, ListYTrain[i], criteria)

                    # Applico la stessa selezione di feature ai dati di test
                    x_test = ListXTest[i].loc[:, selectedFeatures]
                    
                    # Faccio le previsioni sul set di test
                    y_pred = dtl.predict(x_test)
                    
                    # Calcolo e memorizziamo l'F-score
                    fscores.append(f1_score(ListYTest[i], y_pred))

                # Calcolo l'F-score medio su tutti i fold
                avg_fscore = np.mean(fscores)
                print(f"Average F1 score: '{avg_fscore}'")
                
                # Aggiorno i parametri migliori se abbiamo trovato un F-score migliore
                if avg_fscore > best_fscore:
                    best_fscore = avg_fscore
                    best_criterion = criteria
                    best_TH = thre
                    bestN = selectedFeatures

                # In caso di parità nell'F-score, preferisco il modello più semplice
                # (cioè quello con meno feature)
                if avg_fscore == best_fscore:
                    if len(selectedFeatures) < len(bestN):
                        best_fscore = avg_fscore
                        best_criterion = criteria
                        best_TH = thre
                        bestN = selectedFeatures

    # Creo un dizionario con i risultati ottimali
    result = {
        'best_criterion': best_criterion,
        'best_TH': best_TH,
        'bestN': bestN,
        'best_fscore': best_fscore
    }

    # Salvo il risultato se è specificato un percorso
    if save_path:
        with open(save_path, 'wb') as file:
            pickle.dump(result, file)
        print(f"Risultati salvati nella cartella: {save_path}")

    return result



"""
    Determina la configurazione ottimale per un albero di decisione usando PCA e validazione incrociata.
    
    Questa funzione esplora sistematicamente diverse combinazioni di criteri di divisione (gini/entropy)
    e soglie di varianza spiegata per la selezione delle componenti principali. Per ogni combinazione, 
    valuta le prestazioni usando l'F1-score sul set di test di ogni fold della validazione incrociata.
    
    Parametri:
    ListXTrain : list
        Lista di DataFrame contenenti i dati PCA di training per ogni fold.
    ListYTrain : list
        Lista di DataFrame contenenti le etichette di training per ogni fold.
    ListXTest : list
        Lista di DataFrame contenenti i dati PCA di test per ogni fold.
    ListYTest : list
        Lista di DataFrame contenenti le etichette di test per ogni fold.
    explained_variance : array
        Array contenente la varianza spiegata da ciascuna componente principale.
    min_t : float
        Valore minimo della soglia di varianza spiegata da considerare.
    max_t : float
        Valore massimo della soglia di varianza spiegata da considerare.
    step : float
        Incremento della soglia di varianza spiegata ad ogni iterazione.
    save_path : str
        Percorso dove salvare i risultati della configurazione ottimale.
        
    Returns:
    dict
        Dizionario contenente la configurazione ottimale con chiavi:
        - 'best_criterionPCA': criterio di divisione (gini/entropy)
        - 'best_THPCA': soglia ottimale di varianza spiegata
        - 'bestNPCA': numero di componenti principali selezionate
        - 'best_evalPCA': F1-score medio ottenuto con la configurazione ottimale
    """
def determineDecisionTreekFoldConfigurationPCA(ListXTrain, ListYTrain, ListXTest, ListYTest, explained_variance, 
                                               min_t,max_t, step, save_path):
    
    # Verifico se esiste già un file con i risultati
    if save_path and os.path.exists(save_path):
        print(f"File di configurazione PCA trovato: '{save_path}'")
        with open(save_path, 'rb') as file:
            result = pickle.load(file)
        return result

    print("\nNessun file già salvato, eseguo il metodo.\n")

    """
    Determina la configurazione ottimale per un albero di decisione usando PCA.
    Esplora diverse combinazioni di criteri e soglie di varianza spiegata.
    """

    # Inizializzo delle variabili per tenere traccia della migliore configurazione
    best_criterionPCA = None  # Criterio migliore (gini o entropy)
    bestTHPCA = None          # Soglia migliore di varianza spiegata
    bestNPCA = None           # Numero ottimale di componenti principali
    bestEvalPCA = 0           # F-score migliore (inizializzato a 0)

    # Definizione dei criteri da esplorare
    criterion = ['gini', 'entropy']  # I due criteri possibili per l'albero di decisione

    # Esploro tutte le combinazioni di criteri e soglie di varianza
    for criteria in criterion:
        # Itero sulle soglie di varianza con intervallo "step"
        for thre in np.arange(min_t, max_t, step): 
            avg_fscore = 0
            fscores = []
            
            # Calcolo il numero di componenti principali necessarie per raggiungere
            # la percentuale di varianza spiegata specificata dalla soglia corrente
            n = NumberOfTopPCSelect(explained_variance, thre) 
            
            if n > 0:  # Verifico di avere almeno una componente selezionata
                # Per ogni fold della validazione incrociata
                for i in range(len(ListXTrain)):
                    # Seleziono le prime n componenti principali
                    x_train_feature_selected = ListXTrain[i].iloc[:, :n]
                    x_test = ListXTest[i].iloc[:, :n]

                    # Addestro un albero di decisione con il criterio corrente
                    clf = decisionTreeLearner(
                        x_train_feature_selected, ListYTrain[i], criteria)

                    # Facco le previsioni sul set di test
                    y_pred = clf.predict(x_test)
                    
                    # Calcolo e memorizziamo l'F-score
                    fscores.append(f1_score(ListYTest[i], y_pred))

            if len(fscores) > 1:  # Verifichiamo di avere almeno 2 fold validi
                # Calcolo l'F-score medio su tutti i fold
                avg_fscore = np.mean(fscores)
                print(f"Average F1 score: '{avg_fscore}'")
                
                # Aggiorno i parametri migliori se abbiamo trovato un F-score migliore
                if avg_fscore > bestEvalPCA:
                    bestEvalPCA = avg_fscore
                    best_criterionPCA = criteria
                    bestTHPCA = thre
                    bestNPCA = n

                # Se il punteggio è uguale ma il numero di componenti selezionate (n) è inferiore, 
                # preferisco la configurazione con meno componenti (per mantenere il modello più semplice)
                if avg_fscore == bestEvalPCA:
                    if (n < bestNPCA):
                        bestEvalPCA = avg_fscore
                        best_criterionPCA = criteria
                        bestTHPCA = thre
                        bestNPCA = n

    # Creo un dizionario con i risultati ottimali
    result = {
        'best_criterionPCA': best_criterionPCA,  # Criterio ottimale
        'best_THPCA': bestTHPCA,                 # Soglia varianza ottimale
        'bestNPCA': bestNPCA,                    # Numero componenti ottimale
        'best_evalPCA': bestEvalPCA              # F-score ottenuto
    }

     # Salvo il risultato se è specificato un percorso
    if save_path:
        with open(save_path, 'wb') as file:
            pickle.dump(result, file)
        print(f"Risultati salvati nella cartella: {save_path}")

    return result


"""
    Crea e addestra un modello Random Forest per la classificazione.
    Parametri:
    x : pandas.DataFrame
        DataFrame contenente le variabili indipendenti per l'addestramento.
    y : pandas.DataFrame o array-like
        Variabile dipendente (target/label) per l'addestramento.
    n_tree : int
        Numero di alberi nella foresta.
    c : str
        Criterio da utilizzare per la valutazione delle divisioni ('gini' o 'entropy').
    rand : str
        Strategia di selezione delle feature per ogni divisione ('sqrt' o 'log2').
    bootstrap_s : float
        Frazione di campioni da utilizzare per il bootstrap di ogni albero (0.0-1.0).
    seed : int
        Seme per la generazione di numeri casuali, garantisce la riproducibilità.
"""
def randomForestLearner(x, y, n_tree, c, rand, bootstrap_s, seed):
    rlf = RandomForestClassifier(n_estimators=n_tree, criterion=c,
                                 max_features=rand, max_samples=bootstrap_s, random_state=seed)
    rlf.fit(x, np.ravel(y))

    return rlf


def determineRFkFoldConfiguration(ListXTrain, ListYTrain, ListXTest, ListYTest, rank, min_t, max_t, step,
                                            save_path):
    
    # Verifioa se esiste già un file con i risultati
    if save_path and os.path.exists(save_path):
        print(f"File di configurazione Random Forest trovato: '{save_path}'")
        with open(save_path, 'rb') as file:
            result = pickle.load(file)
        return result

    print("\nNessun file già salvato, eseguo il metodo.\n")

    """
    Determina la configurazione ottimale per Random Forest attraverso validazione incrociata.
    Esplora sistematicamente diverse combinazioni di parametri per massimizzare l'F-score.
    """
    # Inizializzo delle variabili per tenere traccia della migliore configurazione
    best_criterion = None    # Criterio migliore (gini o entropy)
    best_TH = None           # Soglia migliore di informazione mutua
    bestN = None             # Migliori feature selezionate
    best_fscore = 0          # F-score migliore (inizializzato a 0)
    best_n_tree = 0          # Numero ottimale di alberi
    best_rand = 0            # Strategia ottimale di selezione delle feature
    best_bootstrap_s = 0     # Dimensione ottimale del bootstrap

    # Definizione dei parametri da esplorare
    criterion = ['gini', 'entropy']                 # Criteri per la divisione dei nodi
    randomization = ['sqrt', 'log2']                # Strategie di selezione delle feature
    number_of_trees = [10, 20, 30]                  # Numero di alberi nella foresta
    bootstrap_size = [0.7, 0.8, 0.9]                # Frazione di campioni per bootstrap

    # Esploro tutte le possibili combinazioni di parametri
    for criteria in criterion:
        for rand in randomization:
            for n_tree in number_of_trees:
                for b_size in bootstrap_size:
                    for thre in np.arange(min_t, max_t, step):  # Itero sulle soglie di informazione mutua
                        avg_fscore = 0
                        fscores = []
                        
                        # Seleziono le feature con informazione mutua >= soglia corrente
                        selectedFeatures = topFeatureSelect(rank, thre)
                        
                        if len(selectedFeatures) > 0:
                            # Per ogni fold della validazione incrociata
                            for i in range(len(ListXTrain)):
                                # Seleziono solo le colonne corrispondenti alle feature scelte
                                x_train_feature_selected = ListXTrain[i].loc[:, selectedFeatures]
                                x_test = ListXTest[i].loc[:, selectedFeatures]
                                
                                # Addestro un modello Random Forest con i parametri correnti
                                rfl = randomForestLearner(x_train_feature_selected, ListYTrain[i], n_tree, criteria,
                                                          rand, b_size, seed=42)

                                # Faccio le previsioni sul set di test
                                y_pred = rfl.predict(x_test)
                                
                                # Calcolo e memorizziamo l'F-score
                                fscores.append(f1_score(ListYTest[i], y_pred))

                        if len(fscores) > 1:  # Verifico di avere almeno 2 fold validi
                            # Calcolo l'F-score medio su tutti i fold
                            avg_fscore = np.mean(fscores)
                            print(f"Average F1 score: '{avg_fscore}'")
                            
                            # Aggiorno i parametri migliori se abbiamo trovato un F-score migliore
                            if avg_fscore > best_fscore:
                                best_fscore = avg_fscore
                                best_criterion = criteria
                                best_TH = thre
                                bestN = selectedFeatures
                                best_n_tree = n_tree
                                best_rand = rand
                                best_bootstrap_s = b_size

                            # In caso di parità nell'F-score, preferisco il modello più semplice
                            # (cioè quello con meno feature)
                            if avg_fscore == best_fscore:
                                if len(selectedFeatures) < len(bestN):
                                    best_fscore = avg_fscore
                                    best_criterion = criteria
                                    best_TH = thre
                                    bestN = selectedFeatures
                                    best_n_tree = n_tree
                                    best_rand = rand
                                    best_bootstrap_s = b_size

    # Creo un dizionario con tutti i risultati ottimali
    result = {
        'best_criterion': best_criterion,   # Criterio ottimale
        'best_TH': best_TH,                 # Soglia MI ottimale
        'bestN': bestN,                     # Feature selezionate
        'best_fscore': best_fscore,         # F-score ottenuto
        'best_n_tree': best_n_tree,         # Numero alberi ottimale
        'best_rand': best_rand,             # Strategia feature ottimale
        'best_bootstrap_s': best_bootstrap_s # Bootstrap size ottimale
    }

    # Salvo il risultato se è specificato un percorso
    if save_path:
        with open(save_path, 'wb') as file:
            pickle.dump(result, file)
        print(f"Risultati salvati nella cartella: {save_path}")

    return result




def determineRFkFoldConfigurationPCA(ListXTrain, ListYTrain, ListXTest, ListYTest, explained_variance, 
                                     min_t, max_t, step, save_path):
    
    # Verifico se esiste già un file con i risultati
    if save_path and os.path.exists(save_path):
        print(f"File di configurazione PCA per Random Forest trovato: '{save_path}'")
        with open(save_path, 'rb') as file:
            result = pickle.load(file)
        return result

    print("\nNessun file già salvato, eseguo il metodo.\n")

    """
    Determina la configurazione ottimale per Random Forest usando PCA.
    Esplora diverse combinazioni di parametri e soglie di varianza spiegata.
    """

    # Inizializzo delle variabili per tenere traccia della migliore configurazione
    best_criterionPCA = None    # Criterio migliore (gini o entropy)
    best_THPCA = None           # Soglia migliore di varianza spiegata
    bestNPCA = None             # Numero ottimale di componenti principali
    best_fscorePCA = 0          # F-score migliore (inizializzato a 0)
    best_n_treePCA = 0          # Numero ottimale di alberi
    best_randPCA = 0            # Strategia ottimale di selezione delle feature
    best_bootstrap_sPCA = 0     # Dimensione ottimale del bootstrap

    # Definisco dei parametri da esplorare
    criterion = ['gini', 'entropy']       # Criteri per la divisione dei nodi
    randomization = ['sqrt', 'log2']      # Strategie di selezione delle feature
    number_of_trees = [10, 20, 30]        # Numero di alberi nella foresta
    bootstrap_size = [0.7, 0.8, 0.9]      # Frazione di campioni per bootstrap

    # Esploro tutte le possibili combinazioni di parametri (grid search)
    for criteria in criterion:
        for rand in randomization:
            for n_tree in number_of_trees:
                for b_size in bootstrap_size:
                    for thre in np.arange(min_t, max_t, step):  # Iteriamo sulle soglie di varianza
                        avg_fscore = 0
                        fscores = []
                        
                        # Calcolo il numero di componenti principali necessarie per raggiungere
                        # la percentuale di varianza spiegata specificata dalla soglia corrente
                        n = NumberOfTopPCSelect(explained_variance, thre)
                        
                        if n > 0:  # Verifico di avere almeno una componente selezionata
                            # Per ogni fold della validazione incrociata
                            for i in range(len(ListXTrain)):
                                # Seleziono le prime n componenti principali
                                x_train_feature_selected = ListXTrain[i].iloc[:, :n]
                                x_test = ListXTest[i].iloc[:, :n]

                                # Addestro un modello Random Forest con i parametri correnti
                                rfl = randomForestLearner(
                                    x_train_feature_selected, ListYTrain[i], n_tree, criteria, rand, b_size, seed=42)

                                # Faccio le previsioni sul set di test
                                y_pred = rfl.predict(x_test)
                                
                                # Calcolo e memorizziamo l'F-score
                                fscores.append(f1_score(ListYTest[i], y_pred))

                        if len(fscores) > 1:  # Verifico di avere almeno 2 fold validi
                            # Calcolo l'F-score medio su tutti i fold
                            avg_fscore = np.mean(fscores)
                            print(f"Average F1 score: '{avg_fscore}'")
                            
                            # Aggiorno i parametri migliori se abbiamo trovato un F-score migliore
                            if avg_fscore > best_fscorePCA:
                                best_fscorePCA = avg_fscore
                                best_criterionPCA = criteria
                                best_THPCA = thre
                                bestNPCA = n
                                best_n_treePCA = n_tree
                                best_randPCA = rand
                                best_bootstrap_sPCA = b_size

                            # Se il punteggio è uguale ma il numero di componenti selezionate (n) è inferiore,
                            # preferisco la configurazione con meno componenti (per un modello più semplice)
                            if avg_fscore == best_fscorePCA:
                                if n < bestNPCA:
                                    best_fscorePCA = avg_fscore
                                    best_criterionPCA = criteria
                                    best_THPCA = thre
                                    bestNPCA = n
                                    best_n_treePCA = n_tree
                                    best_randPCA = rand
                                    best_bootstrap_sPCA = b_size

    # Creo un dizionario con i risultati ottimali
    result = {
        'best_criterionPCA': best_criterionPCA,    # Criterio ottimale
        'best_THPCA': best_THPCA,                  # Soglia varianza ottimale
        'bestNPCA': bestNPCA,                      # Numero componenti ottimale
        'best_fscorePCA': best_fscorePCA,          # F-score ottenuto
        'best_n_treePCA': best_n_treePCA,          # Numero alberi ottimale
        'best_randPCA': best_randPCA,              # Strategia feature ottimale
        'best_bootstrap_sPCA': best_bootstrap_sPCA # Bootstrap size ottimale
    }

    if save_path:
        with open(save_path, 'wb') as file:
            pickle.dump(result, file)
        print(f"Risultati salvati nella cartella: {save_path}")

    return result



def KNNLearner(x, y, n_neighbors):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    
    knn.fit(x, np.ravel(y)) # Addestra il modello KNN sui dati forniti
    
    return knn


def determineKNNkFoldConfiguration(ListXTrain, ListYTrain, ListXTest, ListYTest, rank, min_t, max_t, step,
                                             save_path):

    # Verifica se esiste già un file con i risultati
    if save_path and os.path.exists(save_path):
        print(f"File di configurazione PCA per KNN trovato: '{save_path}'")
        with open(save_path, 'rb') as file:
            result = pickle.load(file)
        return result

    print("\nNessun file già salvato, eseguo il metodo.\n")
    """
    Determina la configurazione ottimale per KNN attraverso validazione incrociata.
    Esplora diverse combinazioni di valori di k (numero di vicini) e soglie di informazione mutua.
    """

    # Inizializzo delle variabili per tenere traccia della migliore configurazione
    best_TH = None          # Soglia migliore di informazione mutua
    bestN = None            # Migliori feature selezionate
    best_fscore = 0         # F-score migliore (inizializzato a 0)
    best_Kneighbors = 0     # Numero ottimale di vicini
    
    # Definizione dei valori di k (vicini) da esplorare
    neighbors = [1, 2, 3]  # Valori del parametro k 

    # Esploro tutte le combinazioni di k e soglie di informazione mutua
    for neighbor in neighbors:
        for thre in np.arange(min_t, max_t, step):  # Itero sulle soglie di informazione mutua
            avg_fscore = 0
            fscores = []
            
            # Seleziono le feature con informazione mutua >= soglia corrente
            selectedFeatures = topFeatureSelect(rank, thre)
            
            if len(selectedFeatures) > 0:  # Verifico di avere almeno una feature selezionata
                # Per ogni fold della validazione incrociata
                for i in range(len(ListXTrain)):
                    # Seleziono solo le colonne corrispondenti alle feature scelte
                    x_train_feature_selected = ListXTrain[i].loc[:, selectedFeatures]
                    x_test = ListXTest[i].loc[:, selectedFeatures]
                    
                    # Addestro un modello KNN con il valore k corrente
                    knn = KNNLearner(x_train_feature_selected, ListYTrain[i], neighbor)

                    # Faccio le previsioni sul set di test
                    y_pred = knn.predict(x_test)
                    
                    # Calcolo e memorizziamo l'F-score
                    fscores.append(f1_score(ListYTest[i], y_pred))

            if len(fscores) > 1:  # Verifichiamo di avere almeno 2 fold validi
                # Calcolo l'F-score medio su tutti i fold
                avg_fscore = np.mean(fscores)
                print(f"Average F1 score: '{avg_fscore}'")
                
                # Aggiorno i parametri migliori se abbiamo trovato un F-score migliore
                if avg_fscore > best_fscore:
                    best_fscore = avg_fscore
                    best_TH = thre
                    bestN = selectedFeatures
                    best_Kneighbors = neighbor

                # In caso di parità nell'F-score, preferiamo il modello più semplice
                # (cioè quello con meno feature)
                if avg_fscore == best_fscore:
                    if len(selectedFeatures) < len(bestN):
                        best_fscore = avg_fscore
                        best_TH = thre
                        bestN = selectedFeatures
                        best_Kneighbors = neighbor

    # Creo un dizionario con i risultati ottimali
    result = {
        'best_TH': best_TH,                 # Soglia MI ottimale
        'bestN': bestN,                     # Feature selezionate
        'best_fscore': best_fscore,         # F-score ottenuto
        'best_Kneighbors': best_Kneighbors, # Valore k ottimale
    }

    # Salvo il risultato se è specificato un percorso
    if save_path:
        with open(save_path, 'wb') as file:
            pickle.dump(result, file)
        print(f"Risultati salvati nella cartella: {save_path}")

    return result


def determineKNNkFoldConfigurationPCA(ListXTrain, ListYTrain, ListXTest, ListYTest, explained_variance,
                                                  min_t,
                                                  max_t, step, save_path):
    # Verifico se esiste già un file con i risultati
    if save_path and os.path.exists(save_path):
        print(f"File di configurazione PCA per KNN trovato: '{save_path}'")
        with open(save_path, 'rb') as file:
            result = pickle.load(file)
        return result

    print("\nNessun file già salvato, eseguo il metodo.\n")

    """
    Determina la configurazione ottimale per KNN usando PCA.
    Esplora diverse combinazioni di valori di k e soglie di varianza spiegata.
    """

    # Inizializzo delle variabili per tenere traccia della migliore configurazione
    bestTHPCA = None            # Soglia migliore di varianza spiegata
    bestNPCA = None             # Numero ottimale di componenti principali
    bestEvalPCA = 0             # F-score migliore (inizializzato a 0)
    best_KneighborsPCA = 0      # Numero ottimale di vicini
    
    # Definizione dei valori di k vicini da esplorare
    neighbors = [1, 2, 3]  

    # Esploro tutte le combinazioni di k e soglie di varianza
    for neighbor in neighbors:
        for thre in np.arange(min_t, max_t, step):  # Itero sulle soglie di varianza
            avg_fscore = 0
            fscores = []
            
            # Calcolo il numero di componenti principali necessarie per raggiungere
            # la percentuale di varianza spiegata specificata dalla soglia corrente
            n = NumberOfTopPCSelect(explained_variance, thre)
            
            if n > 0:  # Verifico di avere almeno una componente selezionata
                # Per ogni fold della validazione incrociata
                for i in range(len(ListXTrain)):
                    # Seleziono le prime n componenti principali
                    x_train_feature_selected = ListXTrain[i].iloc[:, n]
                    x_test = ListXTest[i].iloc[:, n]
                    
                    # Addestro un modello KNN con il valore k corrente
                    knn = KNNLearner(x_train_feature_selected, ListYTrain[i], neighbor)

                    # Faccio le previsioni sul set di test
                    y_pred = knn.predict(x_test)
                    
                    # Calcolo e memorizziamo l'F-score
                    fscores.append(f1_score(ListYTest[i], y_pred))

            if len(fscores) > 1:  # Verifichiamo di avere almeno 2 fold validi
                # Calcolo l'F-score medio su tutti i fold
                avg_fscore = np.mean(fscores)
                print(f"Average F1 score: '{avg_fscore}'")
                
                # Aggiorno i parametri migliori se abbiamo trovato un F-score migliore
                if avg_fscore > bestEvalPCA:
                    bestEvalPCA = avg_fscore
                    bestTHPCA = thre
                    bestNPCA = n
                    best_KneighborsPCA = neighbor

                # Se il punteggio è uguale ma il numero di componenti selezionate (n) è inferiore,
                # preferisco la configurazione con meno componenti (per un modello più semplice)
                if avg_fscore == bestEvalPCA:
                    if n < bestNPCA:
                        bestEvalPCA = avg_fscore
                        bestTHPCA = thre
                        bestNPCA = n
                        best_KneighborsPCA = neighbor

    # Creo un dizionario con i risultati ottimali
    result = {
        'best_THPCA': bestTHPCA,                # Soglia varianza ottimale
        'bestNPCA': bestNPCA,                   # Numero componenti ottimale
        'best_EvalPCA': bestEvalPCA,            # F-score ottenuto
        'best_KneighborsPCA': best_KneighborsPCA, # Valore k ottimale
    }

    # Salvo il risultato se è specificato un percorso
    if save_path:
        with open(save_path, 'wb') as file:
            pickle.dump(result, file)
        print(f"Risultati salvati nella cartella: {save_path}")

    return result


def EnsembleLearner(x, y, dt, rf, knn):
    el = VotingClassifier(
        estimators=[('dt', dt), ('rf', rf), ('knn', knn)], voting='hard')

    el.fit(x, np.ravel(y))
    return el


def determineEnsembleMIkFoldConfiguration(ListXTrain, ListYTrain, ListXTest, ListYTest, rank, min_t, max_t,
                                                  step, DTMI, RFMI, KNNMI, save_path):
    
    # Verifico se esiste già un file con i risultati
    if save_path and os.path.exists(save_path):
        print(f"File di configurazione Ensemble MI trovato: '{save_path}'")
        with open(save_path, 'rb') as file:
            result = pickle.load(file)
        return result

    print("\nNessun file già salvato, eseguo il metodo.\n")

    """
    Determina la configurazione ottimale per un modello Ensemble con selezione feature basata su MI.
    Combina Decision Tree, Random Forest e KNN pre-addestrati in un ensemble con voto di maggioranza.
    """

    # Inizializzo delle variabili per tenere traccia della migliore configurazione
    best_TH = None        # Soglia migliore di informazione mutua
    bestN = None          # Migliori feature selezionate
    best_fscore = 0       # F-score migliore (inizializzato a 0)

    # Esploro diverse soglie di informazione mutua
    for thre in np.arange(min_t, max_t, step):
        avg_fscore = 0
        fscores = []
        
        # Seleziono le feature con informazione mutua >= soglia corrente
        selectedFeatures = topFeatureSelect(rank, thre)
        
        if len(selectedFeatures) > 0:  # Verifico di avere almeno una feature selezionata
            # Per ogni fold della validazione incrociata
            for i in range(len(ListXTrain)):
                # Seleziono solo le colonne corrispondenti alle feature scelte
                x_train_feature_selected = ListXTrain[i].loc[:, selectedFeatures]
                x_test = ListXTest[i].loc[:, selectedFeatures]
                
                # Creo e addestro un modello ensemble combinando i tre classificatori pre-addestrati
                # Nota: i classificatori sono già stati addestrati separatamente con configurazione ottimale
                el = EnsembleLearner(x_train_feature_selected, ListYTrain[i], DTMI, RFMI, KNNMI)

                # Faccio le previsioni sul set di test
                y_pred = el.predict(x_test)
                
                # Calcolo e memorizziamo l'F-score
                fscores.append(f1_score(ListYTest[i], y_pred))

        if len(fscores) > 1:  # Verifichiamo di avere almeno 2 fold validi
            # Calcolo l'F-score medio su tutti i fold
            avg_fscore = np.mean(fscores)
            print(f"Average F1 score: '{avg_fscore}'")
            
            # Aggiorno i parametri migliori se abbiamo trovato un F-score migliore
            if avg_fscore > best_fscore:
                best_fscore = avg_fscore
                best_TH = thre
                bestN = selectedFeatures

            # In caso di parità nell'F-score, preferisco il modello più semplice
            # (cioè quello con meno feature)
            if avg_fscore == best_fscore:
                if len(selectedFeatures) < len(bestN):
                    best_fscore = avg_fscore
                    best_TH = thre
                    bestN = selectedFeatures

    # Creo un dizionario con i risultati ottimali
    result = {
        'best_TH': best_TH,       # Soglia MI ottimale
        'bestN': bestN,           # Feature selezionate
        'best_fscore': best_fscore, # F-score ottenuto
    }

    # Salvo il risultato se è specificato un percorso
    if save_path:
        with open(save_path, 'wb') as file:
            pickle.dump(result, file)
        print(f"Risultati salvati nella cartella: {save_path}")

    return result


def determineEnsemblePCAkFoldConfiguration(ListXTrain, ListYTrain, ListXTest, ListYTest, explained_variance, 
                                           min_t,max_t, step, DTPCA, RFPCA, KNNPCA, save_path):
    
    # Verifica se esiste già un file con i risultati
    if save_path and os.path.exists(save_path):
        print(f"File di configurazione Ensemble PCA trovato: '{save_path}'")
        with open(save_path, 'rb') as file:
            result = pickle.load(file)
        return result

    print("\nNessun file già salvato, eseguo il metodo.\n")

    """
    Determina la configurazione ottimale per un modello Ensemble usando PCA.
    Combina Decision Tree, Random Forest e KNN pre-addestrati con PCA in un ensemble.
    """

    # Inizializzo delle variabili per tenere traccia della migliore configurazione
    bestTHPCA = None       # Soglia migliore di varianza spiegata
    bestNPCA = None        # Numero ottimale di componenti principali
    bestEvalPCA = 0        # F-score migliore (inizializzato a 0)

    # Esploro diverse soglie di varianza
    for thre in np.arange(min_t, max_t, step):  # Itero sulle soglie di varianza
        avg_fscore = 0
        fscores = []
        
        # Calcolo il numero di componenti principali necessarie per raggiungere
        # la percentuale di varianza spiegata specificata dalla soglia corrente
        n = NumberOfTopPCSelect(explained_variance, thre)
        
        if n > 0:  # Verifichiamo di avere almeno una componente selezionata
            # Per ogni fold della validazione incrociata
            for i in range(len(ListXTrain)):
                # Seleziono le prime n componenti principali
                # Nota: qui prendiamo tutte le componenti dall'inizio, senza saltare la prima
                x_train_feature_selected = ListXTrain[i].iloc[:, :n]
                x_test = ListXTest[i].iloc[:, :n]
                
                # Creo e addestro un modello ensemble combinando i tre classificatori PCA pre-addestrati
                # Nota: i classificatori sono già stati addestrati separatamente con configurazione ottimale
                el = EnsembleLearner(x_train_feature_selected, ListYTrain[i], DTPCA, RFPCA, KNNPCA)

                # Faccio le previsioni sul set di test
                y_pred = el.predict(x_test)
                
                # Calcolo e memorizziamo l'F-score
                fscores.append(f1_score(ListYTest[i], y_pred))

        if len(fscores) > 1:  # Verifichiamo di avere almeno 2 fold validi
            # Calcolo l'F-score medio su tutti i fold
            avg_fscore = np.mean(fscores)
            print(f"Average F1 score: '{avg_fscore}'")
            
            # Aggiorno i parametri migliori se abbiamo trovato un F-score migliore
            if avg_fscore > bestEvalPCA:
                bestEvalPCA = avg_fscore
                bestTHPCA = thre
                bestNPCA = n

            # Se il punteggio è uguale ma il numero di componenti selezionate (n) è inferiore,
            # preferisco la configurazione con meno componenti (per un modello più semplice)
            if avg_fscore == bestEvalPCA:
                if n < bestNPCA:
                    bestEvalPCA = avg_fscore
                    bestTHPCA = thre
                    bestNPCA = n

    # Creo un dizionario con i risultati ottimali
    result = {
        'bestTHPCA': bestTHPCA,    # Soglia varianza ottimale
        'bestNPCA': bestNPCA,      # Numero componenti ottimale
        'bestEvalPCA': bestEvalPCA, # F-score ottenuto
    }

    # Salvo il risultato se è specificato un percorso
    if save_path:
        with open(save_path, 'wb') as file:
            pickle.dump(result, file)
        print(f"Risultati salvati nella cartella: {save_path}")

    return result