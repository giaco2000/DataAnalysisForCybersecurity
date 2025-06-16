"""
Informazioni Generali sul File:
- Questo modulo gestisce la fase di selezione dei dati del processo KDD.
  In particolare, carica i dataset emberXTrain e emberYTrain utilizzando pandas.

"""

# Funzioni Importate
import pandas as pd


# funzione per il caricamento dei dati, inserendo anche il controllo dell'eccezione se i file dovessero mancare
def load(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print(f"File non trovato: {file_path}")
        return None