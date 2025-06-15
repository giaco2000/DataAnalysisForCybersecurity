# main.py

"""
Informazioni Generali sul File:
- Questo file è il punto centrale della pipeline KDD. 
  Esegue tutte le fasi chiamando le funzioni dai diversi moduli.
"""

# Funzioni Importate
from DataSelection import load
from DataPreProcessing import *
from DataTransformation import topFeatureSelect, NumberOfTopPCSelect, pca, applyPCA
from DataMining import *
from utils import calcola_o_carica
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# Creo una directory per i file serializzati
SERIALIZED_DIR = "serialized"
os.makedirs(SERIALIZED_DIR, exist_ok=True)

# Definisco i percorsi per i vari file pickle
MUTUAL_INFO_PATH = os.path.join(SERIALIZED_DIR, "mutual_info_ranking.pkl")
DT_MI_CONFIG_PATH = os.path.join(SERIALIZED_DIR, "dt_mi_config.pkl")
DT_PCA_CONFIG_PATH = os.path.join(SERIALIZED_DIR, "dt_pca_config.pkl")
RF_MI_CONFIG_PATH = os.path.join(SERIALIZED_DIR, "rf_mi_config.pkl")
KNN_MI_CONFIG_PATH = os.path.join(SERIALIZED_DIR, "knn_mi_config.pkl")
RF_PCA_CONFIG_PATH = os.path.join(SERIALIZED_DIR, "rf_pca_config.pkl")
KNN_PCA_CONFIG_PATH = os.path.join(SERIALIZED_DIR, "knn_pca_config.pkl")
ENSEMBLE_MI_CONFIG_PATH = os.path.join(SERIALIZED_DIR, "ensemble_mi_config.pkl")
ENSEMBLE_PCA_CONFIG_PATH = os.path.join(SERIALIZED_DIR, "ensemble_pca_config.pkl")

# Funzioni Esportate
if __name__ == "__main__":
    """
    Main script to execute the KDD pipeline.
    """

    # Carico i dati
    TrainXpath = "EmberTrain/EmberXTrain.csv"
    TrainYpath = "EmberTrain/EmberYTrain.csv"
    
    x_train = load(TrainXpath)
    y_train = load(TrainYpath)

    TestXpath = "EmberTest/EmberXTest.csv"
    TestYpath = "EmberTest/EmberYTest.csv"

    x_test = load(TestXpath)
    y_test = load(TestYpath)

    # Per ora il main si limita a caricare i dati, 
    print("Dati caricati con successo!")

    # Stampo le dimensioni dei dataset di train e test
    print("Dimensione di x_train:", x_train.shape)
    print("Dimensione di y_train:", y_train.shape)
    print("Dimensione di x_test:", x_test.shape)
    print("Dimensione di y_test:", y_test.shape)

    # Funzione per visualizzare la preElaborazione dei dati
    preElaborationData(x_train)

    # Rimozione delle colonne inutili
    x_train_cleaned, removed_columns = removeColumns(x_train)

    # Stampa dei nomi delle colonne rimosse e della dimensione della lista con le colonne rimanenti
    print("Nomi delle colonne rimosse: ", removed_columns)
    print("\n")
    print("Dimensione di x_train_cleaned: ", x_train_cleaned.shape)


    # Calcolo quante occorrenze per ogni classe ci sono e stampo l'histogramma
    labelCount = DistribuzioneClassi(y_train, plot=True)
    # Stampo il numero di occorrenze per ogni classe
    print("\nOccorrenze per ogni classe: ", labelCount)

    # Controllo i valori mancanti verificando se il count è < 1200 e li salvo in una lista
    columnsWrongCount = attributiMancanti(x_train)
    # Dopo la rimozione delle colonne inutili e la verifica dei valori mancanti eseguo questo codice
    # per visualizzare la distribuzione delle variabili rispetto alle classi

    # Analisi dei dati con boxplot
    
    # print("\nCreazione dei boxplot per l'analisi dei dati...")
    # preBoxPlotAnalysisData(x_train_cleaned, y_train)
    
    seed = 42
    folds = 5

    # Validazione incrociata stratificata
    print("\nApplicazione della stratifield cross-validation...")
    ListXTrain, ListXTest, ListYTrain, ListYTest = stratifiedKfold(x_train_cleaned, y_train, folds, seed)

    # Stampo le dimensioni dei nuovi vettori creati per il primo fold
    print(f"Fold {0} - Y Train shape:", ListYTrain[0].shape)
    print(f"Fold {0} - X Train shape:", ListXTrain[0].shape)
    print(f"Fold {0} - Y Test shape:", ListYTest[0].shape)
    print(f"Fold {0} - X Test shape:", ListXTest[0].shape)

    # Rimozione delle colonne inutili dai dati di test come fatto per il training
    x_test_cleaned = cleaning_test(x_test, removed_columns)
    print(f"Dimensione di x_test_cleaned: {x_test_cleaned.shape}")


    # MUTUAL INFO

    # Configura i parametri per la ricerca
    threshold = 0.1
    minThreshold = 0
    stepThreshold = 0.02
    top_n = 5  
    bottom_n = 5  
    

    # Applica la serializzazione al calcolo dell'informazione mutua
    print("\nCalcolo dell'informazione mutua tra variabili e classe...")
    rank = calcola_o_carica(MUTUAL_INFO_PATH, mutualInfoRank, x_train_cleaned, y_train)
    print(f"\nCalcolata l'informazione mutua per {len(rank)} variabili")
    
    TopAndBottom(rank, top_n, bottom_n)

    
    # Seleziona le caratteristiche con informazione mutua >= 0.1
    print("\nSelezione delle caratteristiche basata su informazione mutua...")
    selectedfeatures = topFeatureSelect(rank, threshold)

    # Creo un dataset con le caratteristiche selezionate
    X_selected = x_train.loc[:, selectedfeatures]
    print(f"Dimensione del dataset con caratteristiche selezionate: {X_selected.shape}")

    # Determino il valore massimo di informazione mutua per stabilire il range di soglie
    max_mi = 0.0
    for feature, score in rank:
        if score > max_mi:
            max_mi = score
    maxThreshold = max_mi + stepThreshold

    # Salvo comunque il risultato in formato testo per consultazione
    with open("mutual_info_ranking.txt", "w") as f:
        f.write("Rank\tFeature\tInformazione Mutua\n")
        for i, (feature, score) in enumerate(rank):
            f.write(f"{i+1}\t{feature}\t{score:.6f}\n")
    
    print("I risultati completi sono stati salvati nel file 'mutual_info_ranking.txt'")

    # Uso la serializzazione per l'ottimizzazione del Decision Tree
    print("\nDeterminazione della configurazione ottimale per l'albero di decisione...")
    dt_config_result = calcola_o_carica(
        DT_MI_CONFIG_PATH,
        determineDecisionTreekFoldConfiguration,
        ListXTrain, ListYTrain, ListXTest, ListYTest, rank, 
        minThreshold, maxThreshold, stepThreshold,
        save_path=DT_MI_CONFIG_PATH
    )
    
    bestCriterion = dt_config_result['best_criterion']
    bestTH = dt_config_result['best_TH']
    bestN = dt_config_result['bestN']
    bestEval = dt_config_result['best_fscore']
    
    print(f"\nFeature Ranking by MI: Best criterion {bestCriterion}, best MI threshold {bestTH}, best N {len(bestN)}, Best CV F {bestEval}")

    # Seleziono le feature migliori in base alla soglia trovata
    toplist = topFeatureSelect(rank, bestTH)

    # Creo e visualizzo l'albero di decisione ottimale
    DT = decisionTreeLearner(x_train_cleaned.loc[:, toplist], y_train, bestCriterion)
    showTree(DT, model_name="MI")


    # FASE DI VALUTAZIONE SUL TEST SET, PRIMA DOBBIAMO TRASFORMARE IL TEST SET COME IL TRAINING
    
    x_test_dt = x_test_cleaned.loc[:, toplist]
    prediction_dt = DT.predict(x_test_dt)
    cm_dt = confusion_matrix(y_test, prediction_dt)
    cr_dt = classification_report(y_test, prediction_dt)

    print("\nValutazione su dati di test - Decision Tree con MI:")
    print(f"Confusion Matrix:\n{cm_dt}")
    print(f"\nClassification Report:\n{cr_dt}")

    # Visualizzo e salvo la matrice di confusione
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_dt, display_labels=['Goodware', 'Malware'])
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title('Matrice di confusione - Decision Tree con MI')
    plt.savefig('results/confusion_matrix_DT_MI.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Salvo il report di classificazione
    with open('results/classification_report_DT_MI.txt', 'w') as f:
        f.write("Classification Report - Decision Tree con MI\n\n")
        f.write(cr_dt)
    

    # RANDOM FOREST 
    print("\n" + "="*80)
    print("RANDOM FOREST CON MI E STRATIFIED CROSS-VALIDATION")
    print("="*80)

    print("\nDeterminazione della configurazione ottimale per la Random Forest...")
    rf_config_result = calcola_o_carica(
        RF_MI_CONFIG_PATH,
        determineRFkFoldConfiguration,
        ListXTrain, ListYTrain, ListXTest, ListYTest,
        rank, minThreshold, maxThreshold, stepThreshold,
        save_path=RF_MI_CONFIG_PATH
    )

    bestCriterionRF = rf_config_result['best_criterion']
    best_TH_RF = rf_config_result['best_TH']
    bestFeaturesRF = rf_config_result['bestN']
    best_fscore_RF = rf_config_result['best_fscore']
    best_n_tree = rf_config_result['best_n_tree']
    best_rand = rf_config_result['best_rand']
    best_bootstrap_s = rf_config_result['best_bootstrap_s']

    print(f"\nRandom Forest: Best criterion {bestCriterionRF}, max_features {best_rand}, " + 
      f"max_samples {best_bootstrap_s}, n_estimators {best_n_tree}, " + 
      f"MI threshold {best_TH_RF}, features {len(bestFeaturesRF)}, Best CV F {best_fscore_RF}")

    # Addestra un Random Forest con la configurazione ottimale sull'intero dataset
    print("\nAddestramento del Random Forest con la configurazione ottimale...")
    toplist = topFeatureSelect(rank, best_TH_RF)
    RF = randomForestLearner(x_train_cleaned.loc[:, toplist], y_train, 
                       best_n_tree, bestCriterionRF, best_rand, best_bootstrap_s, seed)

    x_test_rf = x_test_cleaned.loc[:, toplist]
    prediction_rf = RF.predict(x_test_rf)
    cm_rf = confusion_matrix(y_test, prediction_rf)
    cr_rf = classification_report(y_test, prediction_rf)

    print("\nValutazione su dati di test - Random Forest con MI:")
    print(f"Confusion Matrix:\n{cm_rf}")
    print(f"\nClassification Report:\n{cr_rf}")

    # Visualizzo e salvo la matrice di confusione
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=['Goodware', 'Malware'])
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title('Matrice di confusione - Random Forest con MI')
    plt.savefig('results/confusion_matrix_RF_MI.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Salvo il report di classificazione
    with open('results/classification_report_RF_MI.txt', 'w') as f:
        f.write("Classification Report - Random Forest con MI\n\n")
        f.write(cr_rf)
    

    # KNN
    print("\n" + "="*80)
    print("KNN CON MI E STRATIFIED CROSS-VALIDATION")
    print("="*80)

    knn_config_result = calcola_o_carica(
    KNN_MI_CONFIG_PATH,
    determineKNNkFoldConfiguration,
    ListXTrain, ListYTrain, ListXTest, ListYTest,
    rank, minThreshold, maxThreshold, stepThreshold,
    save_path=KNN_MI_CONFIG_PATH
    )

    best_Kneighbors = knn_config_result['best_Kneighbors']
    bestThresholdKNN = knn_config_result['best_TH']
    bestFeaturesKNN = knn_config_result['bestN']
    best_eval_knn = knn_config_result['best_fscore']

    print(f"\nKNN: Best k {best_Kneighbors}, MI threshold {bestThresholdKNN}, " + 
    f"features {len(bestFeaturesKNN)}, Best CV F {best_eval_knn}")

    # Addestra un KNN con la configurazione ottimale sull'intero dataset
    print("\nAddestramento del KNN con la configurazione ottimale...")
    toplist = topFeatureSelect(rank, bestThresholdKNN)
    KNN = KNNLearner(x_train_cleaned.loc[:, toplist], y_train, best_Kneighbors)

    x_test_knn = x_test_cleaned.loc[:, toplist]
    prediction_knn = KNN.predict(x_test_knn)
    cm_knn = confusion_matrix(y_test, prediction_knn)
    cr_knn = classification_report(y_test, prediction_knn)

    print("\nValutazione su dati di test - KNN con MI:")
    print(f"Confusion Matrix:\n{cm_knn}")
    print(f"\nClassification Report:\n{cr_knn}")

    # Visualizzo e salvo la matrice di confusione
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_knn, display_labels=['Goodware', 'Malware'])
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title('Matrice di confusione - KNN con MI')
    plt.savefig('results/confusion_matrix_KNN_MI.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Salvo il report di classificazione
    with open('results/classification_report_KNN_MI.txt', 'w') as f:
        f.write("Classification Report - KNN con MI\n\n")
        f.write(cr_knn)
    

    print("\n" + "="*80)
    print("ENSEMBLE MODEL CON STRATIFIED CROSS-VALIDATION E MI)")
    print("="*80)


    # Uso la serializzazione per l'ottimizzazione dell'Ensemble
    ensemble_mi_result = calcola_o_carica(
    ENSEMBLE_MI_CONFIG_PATH,
    determineEnsembleMIkFoldConfiguration,
    ListXTrain, ListYTrain, ListXTest, ListYTest,
    rank, minThreshold, maxThreshold, stepThreshold, 
    DT, RF, KNN,  # Passa i modelli già addestrati
    save_path=ENSEMBLE_MI_CONFIG_PATH
    )

    # Estrazione dei valori dal dizionario restituito
    best_threshold_ens = ensemble_mi_result['best_TH']
    best_features_ens = ensemble_mi_result['bestN']
    best_fscore_ens = ensemble_mi_result['best_fscore']

    print(f"\nEnsemble con MI: threshold {best_threshold_ens}, features {len(best_features_ens)}, F1 {best_fscore_ens}")

    # Addestra l'ensemble sul dataset completo
    toplist = topFeatureSelect(rank, best_threshold_ens)
    ENSEMBLE = EnsembleLearner(x_train_cleaned.loc[:, toplist], y_train, DT, RF, KNN)

    x_test_ens = x_test_cleaned.loc[:, toplist]
    prediction_ens = ENSEMBLE.predict(x_test_ens)
    cm_ens = confusion_matrix(y_test, prediction_ens)
    cr_ens = classification_report(y_test, prediction_ens)

    print("\nValutazione su dati di test - Ensemble Learner con MI:")
    print(f"Confusion Matrix:\n{cm_ens}")
    print(f"\nClassification Report:\n{cr_ens}")

    # Visualizzo e salvo la matrice di confusione
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_ens, display_labels=['Goodware', 'Malware'])
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title('Matrice di confusione - Ensemble con MI')
    plt.savefig('results/confusion_matrix_ENS_MI.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Salvo il report di classificazione
    with open('results/classification_report_ENS_MI.txt', 'w') as f:
        f.write("Classification Report - ENSEMBLE con MI\n\n")
        f.write(cr_ens)
    
    


    # PCA
    # Configura i parametri per la ricerca
    minThreshold = 0.95  # Inizio da una soglia alta per la varianza spiegata
    stepThreshold = 0.01
    maxThreshold = 1.01  # Arrivo fino a 1.0 (100% della varianza)


    # Applico PCA al dataset selezionato
    print("\nApplicazione dell'analisi delle componenti principali...")
    
    Pca, pcalist, explained_variance = pca(x_train_cleaned)
    print("\nLunghezza PCA List , sul dataset completo dopo pulizia: ", len(pcalist))
    print("Explained variance , sul dataset completo dopo pulizia : ", explained_variance)

    X_pca = applyPCA(x_train_cleaned, Pca, pcalist)

    # Determino il numero di componenti principali per raggiungere il 99% di varianza spiegata
    n = NumberOfTopPCSelect(explained_variance, 0.99)
    print(f"Numero di componenti principali per ottenere il 99% di varianza spiegata: {n}")

    # Creo un dataset con le componenti principali selezionate
    X_pca_selected = X_pca.iloc[:, 1:(n + 1)]
    print(f"Dimensione del dataset con componenti principali selezionate (X_pca_selected): {X_pca_selected.shape}")

    # Preparo dataset PCA per cross-validation
    print("\nPreparazione dei dati PCA per la cross-validation...")
    ListXTrainPCA,ListXTestPCA, ListYTrainPCA,  ListYTestPCA = stratifiedKfold(X_pca, y_train, folds, seed)

    # DECISION TREE CON PCA

    # Determino la configurazione ottimale per il Decision Tree con PCA
    print("\nDeterminazione della configurazione ottimale per l'albero di decisione con PCA...")
    dt_pca_config_result = calcola_o_carica(
        DT_PCA_CONFIG_PATH,
        determineDecisionTreekFoldConfigurationPCA,
        ListXTrainPCA, ListYTrainPCA, ListXTestPCA, ListYTestPCA,
        explained_variance, minThreshold, maxThreshold, stepThreshold,
        save_path=DT_PCA_CONFIG_PATH
    )

    bestCriterionPCA = dt_pca_config_result['best_criterionPCA']
    bestThresholdPCA = dt_pca_config_result['best_THPCA']
    bestNPCA = dt_pca_config_result['bestNPCA']
    bestEvalPCA = dt_pca_config_result['best_evalPCA']

    print(f"\nPCA: Best criterion {bestCriterionPCA}, Best Threshold {bestThresholdPCA}, Best components N {bestNPCA}, Best CV F {bestEvalPCA}")

    # Addestro un Decision Tree con la configurazione PCA ottimale
    print("\nAddestramento dell'albero di decisione con la configurazione PCA ottimale...")
    # Seleziono le prime bestNPCA componenti principali
    X_pca_optimal = X_pca.iloc[:, :bestNPCA]
    DT_PCA = decisionTreeLearner(X_pca_optimal, y_train, bestCriterionPCA)

    # Visualizzo l'albero di decisione con PCA
    print("\nVisualizzazione dell'albero di decisione con PCA...")
    showTree(DT_PCA, feature_names=[f"PC{i+1}" for i in range(bestNPCA)], max_depth=3, model_name="PCA")

    # Fase di valutazione decision tree pca
    x_test_pca = applyPCA(x_test_cleaned, Pca, pcalist)

    x_test_pca_features = x_test_pca.iloc[:, :bestNPCA]

    prediction_dt_pca = DT_PCA.predict(x_test_pca_features)

    cm_dt_pca = confusion_matrix(y_test, prediction_dt_pca)
    cr_dt_pca = classification_report(y_test, prediction_dt_pca)

    print("\nValutazione su dati di test - Decision Tree con PCA:")
    print(f"Confusion Matrix:\n{cm_dt_pca}")
    print(f"\nClassification Report:\n{cr_dt_pca}")

    # Visualizzo e salvo la matrice di confusione
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_dt_pca, display_labels=['Goodware', 'Malware'])
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title('Matrice di confusione - Decision Tree con PCA')
    plt.savefig('results/confusion_matrix_DT_PCA.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Salvo il report di classificazione
    with open('results/classification_report_DT_PCA.txt', 'w') as f:
        f.write("Classification Report - Decision Tree con PCA\n\n")
        f.write(cr_dt_pca)
    
    # RANDOM FOREST CON PCA
    # Determino la configurazione ottimale per Random Forest con PCA
    print("\n" + "="*80)
    print("RANDOM FOREST CON PCA E STRATIFIED CROSS-VALIDATION")
    print("="*80)

    rf_pca_config_result = calcola_o_carica(
        RF_PCA_CONFIG_PATH,
        determineRFkFoldConfigurationPCA,
        ListXTrainPCA, ListYTrainPCA, ListXTestPCA, ListYTestPCA,
        explained_variance, minThreshold, maxThreshold, stepThreshold,
        save_path=RF_PCA_CONFIG_PATH
    )

    bestCriterionRF_PCA = rf_pca_config_result['best_criterionPCA']
    bestMaxFeaturesRF_PCA = rf_pca_config_result['best_randPCA']
    bestMaxSamplesRF_PCA = rf_pca_config_result['best_bootstrap_sPCA']
    bestNEstimatorsRF_PCA = rf_pca_config_result['best_n_treePCA']
    bestThresholdRF_PCA = rf_pca_config_result['best_THPCA']
    bestNPC_RF = rf_pca_config_result['bestNPCA']
    bestEvalRF_PCA = rf_pca_config_result['best_fscorePCA']

    print(f"\nRandom Forest con PCA: Best criterion: {bestCriterionRF_PCA}, Best Rand: {bestMaxFeaturesRF_PCA}, " + 
    f"Best Bootstrap: {bestMaxSamplesRF_PCA}, Best number Tree: {bestNEstimatorsRF_PCA}, " + 
    f"Best Threshold: {bestThresholdRF_PCA}, Best N {bestNPC_RF}, Best CV F (Fscore): {bestEvalRF_PCA}")

    # Addestro un Random Forest con la configurazione PCA ottimale
    print("\nAddestramento del Random Forest con la configurazione PCA ottimale...")
    X_pca_rf_optimal = X_pca.iloc[:, :bestNPC_RF]
    RF_PCA = randomForestLearner(X_pca_rf_optimal, y_train, 
                        bestNEstimatorsRF_PCA, bestCriterionRF_PCA, bestMaxFeaturesRF_PCA, bestMaxSamplesRF_PCA, seed)
    x_test_rf_pca = applyPCA(x_test_cleaned, Pca, pcalist)
    x_test_rf_pca_features = x_test_rf_pca.iloc[:, :bestNPC_RF]
    prediction_rf_pca = RF_PCA.predict(x_test_rf_pca_features)

    cm_rf_pca = confusion_matrix(y_test, prediction_rf_pca)
    cr_rf_pca = classification_report(y_test, prediction_rf_pca)

    print("\nValutazione su dati di test - Random Forest con PCA:")
    print(f"Confusion Matrix:\n{cm_rf_pca}")
    print(f"\nClassification Report:\n{cr_rf_pca}")

    # Visualizzo e salvo la matrice di confusione
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_rf_pca, display_labels=['Goodware', 'Malware'])
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title('Matrice di confusione - Random Forest con PCA')
    plt.savefig('results/confusion_matrix_RF_PCA.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Salvo il report di classificazione
    with open('results/classification_report_RF_PCA.txt', 'w') as f:
        f.write("Classification Report - Random Forest con PCA\n\n")
        f.write(cr_rf_pca)
    
    # KNN CON PCA
    # Determino la configurazione ottimale per KNN con PCA
    print("\n" + "="*80)
    print("KNN CON PCA E STRATIFIED CROSS-VALIDATION")
    print("="*80)

    knn_pca_config_result = calcola_o_carica(
        KNN_PCA_CONFIG_PATH,
        determineKNNkFoldConfigurationPCA,
        ListXTrainPCA, ListYTrainPCA, ListXTestPCA, ListYTestPCA,
        explained_variance, minThreshold, maxThreshold, stepThreshold,
        save_path=KNN_PCA_CONFIG_PATH
    )

    best_k_pca = knn_pca_config_result['best_KneighborsPCA']
    bestThresholdKNN_PCA = knn_pca_config_result['best_THPCA']
    bestNPC_KNN = knn_pca_config_result['bestNPCA']
    best_eval_knn_pca = knn_pca_config_result['best_EvalPCA']

    print(f"\nKNN con PCA: Best k (vicini): {best_k_pca}, Best threshold: {bestThresholdKNN_PCA}, " + 
    f"Best N: {bestNPC_KNN}, Best CV F (fscore): {best_eval_knn_pca}")

    # Addestro un KNN con la configurazione PCA ottimale
    print("\nAddestramento del KNN con la configurazione PCA ottimale...")
    X_pca_knn_optimal = X_pca.iloc[:, :bestNPC_KNN]
    KNN_PCA = KNNLearner(X_pca_knn_optimal, y_train, best_k_pca)

    x_test_knn_pca = applyPCA(x_test_cleaned, Pca, pcalist)
    x_test_knn_pca_features = x_test_knn_pca.iloc[:, :bestNPC_KNN]

    prediction_knn_pca = KNN_PCA.predict(x_test_knn_pca_features)

    cm_knn_pca = confusion_matrix(y_test, prediction_knn_pca)
    cr_knn_pca = classification_report(y_test, prediction_knn_pca)

    print("\nValutazione su dati di test - KNN con PCA:")
    print(f"Confusion Matrix:\n{cm_knn_pca}")
    print(f"\nClassification Report:\n{cr_knn_pca}")

    # Visualizzo e salvo la matrice di confusione
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_knn_pca, display_labels=['Goodware', 'Malware'])
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title('Matrice di confusione - KNN con PCA')
    plt.savefig('results/confusion_matrix_KNN_PCA.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Salvo il report di classificazione
    with open('results/classification_report_KNN_PCA.txt', 'w') as f:
        f.write("Classification Report - KNN con PCA\n\n")
        f.write(cr_knn_pca)
    

    # ENSEMBLE CON PCA
    # Test del modello ensemble con PCA
    print("\n" + "="*80)
    print("ENSEMBLE MODEL CON STRATIFIED CROSS-VALIDATION (PCA)")
    print("="*80)


    # Uso la serializzazione per l'ottimizzazione dell'Ensemble con PCA
    ensemble_pca_result = calcola_o_carica(
    ENSEMBLE_PCA_CONFIG_PATH,
    determineEnsemblePCAkFoldConfiguration,
    ListXTrainPCA, ListYTrainPCA, ListXTestPCA, ListYTestPCA,
    explained_variance, minThreshold, maxThreshold, stepThreshold,
    DT_PCA, RF_PCA, KNN_PCA,  # Passa i modelli PCA già addestrati
    save_path=ENSEMBLE_PCA_CONFIG_PATH
    )

    # Estrazione dei valori dal dizionario restituito
    best_threshold_ens_pca = ensemble_pca_result['bestTHPCA']
    best_n_pc_ens = ensemble_pca_result['bestNPCA']
    best_fscore_ens_pca = ensemble_pca_result['bestEvalPCA']

    print(f"\nEnsemble con PCA: Best threshold: {best_threshold_ens_pca}, Best N: {best_n_pc_ens}, Best CV F (Fscore) {best_fscore_ens_pca}")

    # Addestro l'ensemble sul dataset completo
    X_ens_pca = X_pca.iloc[:, :best_n_pc_ens]
    ENSEMBLE_PCA = EnsembleLearner(X_ens_pca, y_train, DT_PCA, RF_PCA, KNN_PCA)

    x_test_ens_pca = applyPCA(x_test_cleaned, Pca, pcalist)
    x_test_ens_pca_features = x_test_ens_pca.iloc[:, :best_n_pc_ens]

    prediction_ens_pca = ENSEMBLE_PCA.predict(x_test_ens_pca_features)
    cm_ens_pca = confusion_matrix(y_test, prediction_ens_pca)
    cr_ens_pca = classification_report(y_test, prediction_ens_pca)

    print("\nValutazione su dati di test - Ensemble Learner con PCA:")
    print(f"Confusion Matrix:\n{cm_ens_pca}")

    print(f"\nClassification Report:\n{cr_ens_pca}")
    # Visualizzo e salvo la matrice di confusione
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_ens_pca, display_labels=['Goodware', 'Malware'])
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title('Matrice di confusione - Ensemble con PCA')
    plt.savefig('results/confusion_matrix_ENS_PCA.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Salvo il report di classificazione    
    with open('results/classification_report_ENS_PCA.txt', 'w') as f:
        f.write("Classification Report - ENSEMBLE con PCA\n\n")
        f.write(cr_ens_pca)