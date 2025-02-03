import os
import pandas as pd

# Percorso principale della cartella contenente tutte le cartelle
cartella_principale = "/Users/valeriapontillo/Desktop/ICPC2024_appendix/results/rq1/infoGainRQ1/infogainanalysis" # Cambia con il percorso corretto

# Mappatura dei nomi delle variabili con spazi ai nomi desiderati
mappatura_nomi_variabili = {
    "Assertion Roulette": "AssertionRoulette",
    "Conditional Test Logic": "ConditionalTestLogic",
    "Constructor Initialization": "ConstructorInitialization",
    "Default Test": "DefaultTest",
    "Exception Catching Throwing": "ExceptionCatchingThrowing",
    "General Fixture": "GeneralFixture",
    "Mystery Guest": "MysteryGuest",
    "Print Statement": "PrintStatement",
    "Redundant Assertion": "RedundantAssertion",
    "Sensitive Equality": "SensitiveEquality",
    "Verbose Test": "VerboseTest",
    "Sleepy Test": "SleepyTest",
    "Eager Test": "EagerTest",
    "Lazy Test": "LazyTest",
    "Duplicate Assert": "DuplicateAssert",
    "Unknown Test": "UnknownTest",
    "Resource Optimism": "ResourceOptimism",
    "Magic Number Test": "MagicNumberTest",
    "Dependent Test": "DependentTest"
}


# Funzione per estrarre il nome del progetto dalla cartella
def estrai_nome_progetto(nome_cartella):
    return nome_cartella.split('_')[0]


# Funzione per modificare i nomi delle variabili in base alla mappatura
def modifica_nomi_variabili(df):
    # Sostituisci i nomi delle variabili se corrispondono a quelli nel dizionario
    df['Variabile'] = df['Variabile'].replace(mappatura_nomi_variabili)
    return df


# Funzione per leggere i file e salvare i dati
def salva_dati_da_cartella(cartella_progetto):
    primo_file = True
    dati = []

    # Percorso della cartella IG all'interno della cartella del progetto
    cartella_ig = os.path.join(cartella_progetto, "IG")

    # Scorre tutti i file nella cartella IG
    for root, dirs, files in os.walk(cartella_ig):
        for file in files:
            if file.endswith(".txt"):  # Supponendo che i file siano in formato .txt, cambia se necessario
                percorso_file = os.path.join(root, file)

                try:
                    # Leggi il file come DataFrame
                    df = pd.read_csv(percorso_file, sep=",", header=None, engine='python', error_bad_lines=False)


                    # Assegna nomi alle colonne
                    df.columns = ['Variabile', 'Valore']

                    # Modifica i nomi delle variabili
                    df = modifica_nomi_variabili(df)

                    if primo_file:
                        # Salva tutto dal primo file
                        dati.append(df)
                        primo_file = False
                    else:
                        # Per i file successivi, prendi solo la colonna 'Valore'
                        df_valori = df[['Valore']]
                        dati.append(df_valori)

                except pd.errors.ParserError as e:
                    print(f"Errore nel file {percorso_file}: {e}")


    # Concatenazione di tutti i DataFrame
    if dati:
        df_finale = pd.concat(dati, axis=1)
        return df_finale
    else:
        return None
    exit(0)

# Funzione principale per processare tutte le cartelle nella cartella principale
def processa_cartelle_principali(cartella_principale):
    # Scorre tutte le cartelle nella cartella principale
    for nome_cartella in os.listdir(cartella_principale):
        percorso_cartella_progetto = os.path.join(cartella_principale, nome_cartella)

        if os.path.isdir(percorso_cartella_progetto):
            # Estrai il nome del progetto dalla cartella
            nome_progetto = estrai_nome_progetto(nome_cartella)

            # Processa i file nella cartella IG
            df_finale = salva_dati_da_cartella(percorso_cartella_progetto)

            # Se ci sono dati, salva in un CSV
            if df_finale is not None:
                output_file = os.path.join(cartella_principale, f"{nome_progetto}_ig.csv")
                df_finale.to_csv(output_file, index=False)
                print(f"Salvato: {output_file}")


# Chiamata alla funzione per processare tutte le cartelle nella cartella principale
processa_cartelle_principali(cartella_principale)
