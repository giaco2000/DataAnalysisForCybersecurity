# utils.py
import os
import pickle

def calcola_o_carica(file_path, calcolo_funzione, *args, **kwargs):
    """
    Controlla se esiste un file pickle, se sì lo carica, 
    altrimenti esegue la funzione e salva il risultato.
    
    Args:
        file_path (str): Percorso del file pickle da caricare o salvare
        calcolo_funzione (callable): Funzione da eseguire se il file non esiste
        *args, **kwargs: Argomenti da passare alla funzione
        
    Returns:
        Il risultato caricato dal file o calcolato dalla funzione
    """
    # Controlla se il file esiste
    if os.path.exists(file_path):
        print(f"Caricamento da file: {file_path}")
        with open(file_path, 'rb') as f:
            result = pickle.load(f)
        return result
    
    # Se non esiste, eseguo il calcolo
    print(f"Esecuzione del calcolo...")
    result = calcolo_funzione(*args, **kwargs)
    
    # Salvo il risultato
    print(f"Salvataggio in: {file_path}")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(result, f)
    
    return result