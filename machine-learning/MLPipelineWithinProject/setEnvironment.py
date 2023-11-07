import os

import pandas as pd

file_list = "/Users/valeriapontillo/Desktop/ICPC2024_appendix/sample"

dir_refactoring = "/Users/valeriapontillo/Desktop/ICPC2024_appendix/sample_refactoring"

list_file = os.listdir(dir_refactoring)

for file in list_file:
    if file != ".DS_Store" and file == "ReplaceTryCatchWithRule":
        projects = os.listdir(dir_refactoring+"/"+file)
        for project in projects:
            df = pd.read_csv(dir_refactoring+"/"+file+"/"+project)
            columnsName = df.columns[-1]
            columns_add = ['is'+columnsName]
            df[columns_add] = False


            for index, row in df.iterrows():
                if row[columnsName] != 0:
                    df.at[index, columns_add] = True

            df.to_csv(dir_refactoring+"/"+file+"/"+project, index=False)



# Ottieni l'elenco dei file nella directory
'''projects = os.listdir(file_list)

for project in projects:
    if project != ".DS_Store":
        project_df = pd.read_csv(file_list + "/" + project)

        nuovi_nomi_colonne = {col: col.replace(' ', '') for col in project_df.columns}
        project_df = project_df.rename(columns=nuovi_nomi_colonne)

        for i in range(30,80):
            colonnes_da_copiare = list(range(0, 29))
            df_intermedio = project_df.iloc[:, colonnes_da_copiare]
            df_support = project_df.iloc[:,i]

            project_refactorings = pd.concat([df_intermedio, df_support], axis=1)

            # Ottenere il nome dell'ultima colonna del DataFrame
            ultimo_nome_colonna = project_refactorings.columns[-1]

            # Controlla se il nome dell'ultima colonna è presente nella lista di nomi di file
            if ultimo_nome_colonna in list_file:
                if ultimo_nome_colonna == "ReplaceConditionalByParameterizedTest":
                        print(ultimo_nome_colonna)

                        project_refactorings.to_csv("/Users/valeriapontillo/Desktop/ICPC2024_appendix/sample_refactoring/"+ultimo_nome_colonna+"/"+project, index=False)

'''
'''for project in projects:
    if project != ".DS_Store":
        project_df = pd.read_csv(file_list+"/"+project)

        colonnes_da_copiare = list(range(0, 30))
        refactoring = ['Merge Method', 'Merge Class', 'Split Class', 'Invert Condition', 'Extract Interface', 'Replace Pipeline With Loop', 'Extract Subclass','Push Down Attribute',
'Split Conditional','Inline Attribute','Replace Attribute','Replace Try/Catch With Rule','Merge Attribute','Replace Loop With Pipeline','Push Down Method',
'Encapsulate Attribute','Extract Superclass','Merge Variable','Split Attribute','Split Variable','Replace NOT operator','Replace Try/Catch With AssertThrows',
'Replace Attribute With Variable','Extract Attribute','Pull Up Method','Replace conditional by ParameterizedTest','Extract Class','Pull Up Attribute',
'Replace @test(expected) with assertThrows','Extract And Move Method','Move Class','Move Attribute','Replace Rule With AssertThrows','Add Assert Argument',
'Inline Method','Move And Inline Method','Split Conditional Statement in Assertions','Replace Anonymous With Lambda','Modify Class Annotation',
'Replace Variable With Attribute','Inline Variable','Replace Reserved Words','Remove Class Annotation','Add Class Annotation','Move Method','Extract Method',
'Extract Variable','Remove Method Annotation','Modify Method Annotation','Add Method Annotation']

        df_intermedio = project_df.iloc[:, colonnes_da_copiare]

        project_refactorings = pd.concat([df_intermedio, project_df[refactoring]], axis=1)

        project_refactorings = project_refactorings.reset_index()

        project_refactorings.to_csv(file_list+"/"+project, index= False)



# Leggi il file di testo
file_list = "/Users/valeriapontillo/Desktop/listeRefactoring.txt"  # Sostituisci con il tuo percorso effettivo

with open(file_list, 'r') as file:
    names = file.read().splitlines()

# Creazione delle directory
for name in names:
    directory_path = os.path.join(".", '/Users/valeriapontillo/Desktop/ICPC2024_appendix/sample_refactoring/')  # Percorso della nuova directory
    try:
        os.mkdir(directory_path+name)
        print(f"Creata la directory: {directory_path+name}")
    except FileExistsError:
        print(f"La directory '{directory_path+name}' esiste già.")

print("Operazione completata.")'''




