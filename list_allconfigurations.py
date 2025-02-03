import os

# Specifica la directory
directory = "/Users/valeriapontillo/Downloads/new_data_IST/per_refactoring"

output_file = "allConfigurations_perrefactoring"

# Modelli di classificazione
classifiers = ["svm", "randomforest", "naivebayes", "decisiontree", "logisticregression", "extratree"]

# Tecniche di bilanciamento
balancing_methods = ["randomunder", "nearmissunder1", "nearmissunder2", "nearmissunder3", "randomover", "smoteover", "adasyin"]

# Ottieni tutti i file .csv e rimuovi l'estensione
csv_files = [f[:-4] for f in os.listdir(directory) if f.endswith(".csv")]

# Lista per salvare tutte le configurazioni
configurations = []

for file_name in csv_files:
    # Prima configurazione: modello + nome file
    for classifier in classifiers:
        configurations.append(f"{classifier} {file_name}")

    # Seconda configurazione: modello + nome file + tecnica di bilanciamento
    for classifier in classifiers:
        for method in balancing_methods:
            configurations.append(f"{classifier} {file_name} {method}")

# Scrivi le configurazioni nel file .txt, una per riga
with open(output_file, "w") as f:
    f.write("\n".join(configurations))

print(f"Nomi salvati in {output_file}")
print("Numero totale di combinazioni generate:", len(configurations))
