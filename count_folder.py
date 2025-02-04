import os
from collections import Counter

# Specificare la directory
directory_path = "./data/"  # Cambia con il percorso della tua directory

# Ottenere la lista delle folder presenti nella directory
folders = [name for name in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, name))]

# Estrarre il prefisso prima del primo "_"
prefixes = [folder.split("_")[0] for folder in folders]

# Contare le occorrenze
distribution = Counter(prefixes)

# Stampare il risultato
for prefix, count in sorted(distribution.items()):
    print(f"{prefix}: {count}")

print(len(distribution))

