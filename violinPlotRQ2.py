import os

import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


input_file =  "/yourpath/results/rq2/infogain/violinPlotData/"
refactorings = os.listdir(input_file)

print(refactorings)

for refactoring in refactorings:
    if refactoring != ".DS_Store":
        print(refactoring)

        file = pd.read_csv(input_file + "/" + refactoring)
        refactoring = os.path.splitext(os.path.basename(refactoring))[0]


        metrics1 = ['codeChurnTotal', 'codeChurnMax', 'codeChurnAvg', 'commits', 'Contributors',
                    'minorContributors',
                    'contributorsExperience','addLinesCount','addLinesMax','addLinesAvg','removedLinesCount','removedLinesMax',
                    'removedLinesAvg']
        metrics2 = ['LOC', 'NOM', 'WMC', 'RFC', 'AsD']
        metrics3 = ['Assertion Roulette', 'Conditional Test Logic', 'Constructor Initialization', 'Duplicate Assert', 'Eager Test',
                    'Exception Catching Throwing']
        metrics4 = ['General Fixture', 'IgnoredTest', 'Lazy Test', 'Mystery Guest', 'Magic Number Test']
        metrics5 = ['Redundant Assertion', 'Resource Optimism', 'Print Statement', 'Sensitive Equality', 'Sleepy Test', 'Unknown Test']

        result_dict = {}
        df = pd.DataFrame(columns=['Metric', 'Value'])
        for index, row in file.iterrows():
            metric_label = row['Metric']
            values = row[1:].tolist()
            result_dict[metric_label] = values

    # add data to the final dataframe
            df = pd.concat([df, pd.DataFrame({'Metric': [metric_label] * len(values), 'Value': values})], ignore_index=True)



        df_controlmetrics = df[df['Metric'].isin(metrics1)]

        df_testcodemetrics = df[df['Metric'].isin(metrics2)]

        df_testsmell1 = df[df['Metric'].isin(metrics3)]
        df_testsmell2 = df[df['Metric'].isin(metrics4)]
        df_testsmell3 = df[df['Metric'].isin(metrics5)]

        mapping1 = {'codeChurnTotal':'CCT', 'codeChurnMax':'CCM', 'codeChurnAvg': 'CCA',
                                                      'commits' : 'Co', 'Contributors': 'Con', 'minorContributors': 'MCon',
                                                      'contributorsExperience': 'ConE', 'addLinesCount': 'ALC', 'addLinesMax':'ALM',
                                                      'addLinesAvg': 'ALA', 'removedLinesCount': 'RLC', 'removedLinesMax': 'RLM',
                                                      'removedLinesAvg': 'RLA'}

        mapping2 = {'LOC': 'LOC', 'NOM': 'NOM', 'WMC': 'WMC', 'RFC': 'RFC', 'AsD': 'AD'}

        mapping3 = {'Assertion Roulette': 'AR', 'Constructor Initialization': 'CI', 'Conditional Test Logic': 'CTL',
            'Duplicate Assert': 'DA', 'Eager Test': 'ET',
            'Exception Catching Throwing': 'ECT'}
        mapping4 = {'General Fixture': 'GF', 'IgnoredTest': 'IgT', 'Lazy Test': 'LT', 'Mystery Guest': 'MG', 'Magic Number Test': 'MNT'}
        mapping5 = {'Redundant Assertion': 'RA', 'Resource Optimism': 'RO', 'Print Statement': 'PS',
            'Sensitive Equality': 'SE', 'Sleepy Test': 'ST', 'Unknown Test': 'UT'}


        df_controlmetrics['Metric'] = df_controlmetrics['Metric'].replace(mapping1)
        df_testcodemetrics['Metric'] = df_testcodemetrics['Metric'].replace(mapping2)
        df_testsmell1['Metric'] = df_testsmell1['Metric'].replace(mapping3)
        df_testsmell2['Metric'] = df_testsmell2['Metric'].replace(mapping4)
        df_testsmell3['Metric'] = df_testsmell3['Metric'].replace(mapping5)


# Create violin plot
        plt.figure(figsize=(10, 6))
        sns.violinplot(x='Metric', y='Value', data=df_controlmetrics, inner=None, color="purple")
        sns.boxplot(x='Metric', y='Value', data=df_controlmetrics, width=0.2, color="white", linewidth=1, boxprops=dict(alpha=0))  # Boxplot trasparenti
        plt.title('Violin Plot for Process Metrics Dimension in ' + refactoring)
        plt.xlabel('Metric')
        plt.ylabel('Value')
        plt.savefig("/yourpath/results/rq2/infogain/violinPlot/" + refactoring+"/violinPlot_processMetrics.pdf", format='pdf')

        plt.figure(figsize=(10, 6))
        sns.violinplot(x='Metric', y='Value', data=df_testcodemetrics, inner=None, color="skyblue")
        sns.boxplot(x='Metric', y='Value', data=df_testcodemetrics, width=0.2, color="white", linewidth=1, boxprops=dict(alpha=0))  # Boxplot trasparenti
        plt.title('Violin Plot for Test Code Metrics Dimension in' + refactoring)
        plt.xlabel('Metric')
        plt.ylabel('Value')
        plt.savefig(
                "/yourpath/results/rq2/infogain/violinPlot/" + refactoring + "/violinPlot_testCodeMetrics.pdf",
                format='pdf')

        plt.figure(figsize=(10, 6))
        sns.violinplot(x='Metric', y='Value', data=df_testsmell1, inner=None, color = "blue")
        sns.boxplot(x='Metric', y='Value', data=df_testsmell1, width=0.2, color="white", linewidth=1, boxprops=dict(alpha=0))  # Boxplot trasparenti
        plt.title('Violin Plot for Test Smells Dimension in ' + refactoring)
        plt.xlabel('Metric')
        plt.ylabel('Value')
        plt.savefig("/yourpath/results/rq2/infogain/violinPlot/" + refactoring+"/violinPlot_testSmells1.pdf", format='pdf')


        plt.figure(figsize=(10, 6))
        sns.violinplot(x='Metric', y='Value', data=df_testsmell2, inner=None, color="blue")
        sns.boxplot(x='Metric', y='Value', data=df_testsmell2, width=0.2, color="white", linewidth=1,
                        boxprops=dict(alpha=0))  # Boxplot trasparenti
        plt.title('Violin Plot for Test Smells Dimension in ' + refactoring)
        plt.xlabel('Metric')
        plt.ylabel('Value')
        plt.savefig("/yourpath/results/rq2/infogain/violinPlot/" + refactoring+"/violinPlot_testSmells2.pdf", format='pdf')


        plt.figure(figsize=(10, 6))
        sns.violinplot(x='Metric', y='Value', data=df_testsmell3, inner=None, color="blue")
        sns.boxplot(x='Metric', y='Value', data=df_testsmell3, width=0.2, color="white", linewidth=1,
                        boxprops=dict(alpha=0))  # Boxplot trasparenti
        plt.title('Violin Plot for Test Smells Dimension in ' + refactoring)
        plt.xlabel('Metric')
        plt.ylabel('Value')
        plt.savefig("/yourpath/results/rq2/infogain/violinPlot/" + refactoring+"/violinPlot_testSmells3.pdf", format='pdf')
