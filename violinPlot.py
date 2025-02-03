import glob

import pandas as pd
import seaborn as sns

import os

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

CCT=[0.682908,0.636514,0.636514]
CCM=[0.682908,0.636514,0.636514]
CCA=[0.682908,0.636514,0.636514]
Co=[0.484866,0.636514,0.636514]
Con=[0.088782,0.636514,0.075671]
MCon=[0.088782,0.636514,0.075671]
ConE=[0.088782,0.636514,0.075671]
ALC=[0.682908,0.636514,0.636514]
ALM=[0.484866,0.636514,0.405465]
ALA=[0.682908,0.636514,0.636514]
RLC=[0.682908,0.636514,0.636514]
RLM=[0.484866,0.636514,0.405465]
RLA=[0.682908,0.636514,0.636514]
AR=[0.361574,0.174416,0.261624]
CTL=[0.004143,0.174416,0.030575]
CI=[0.0,0.0,0.0]
DT=[0.0,0.0,0.0]
ECT=[0.484866,0.636514,0.405465]
GF=[0.0,0.0,0.0]
MG=[0.0,0.0,0.0]
RP=[0.0,0.0,0.0]
RA=[0.0,0.0,0.0]
SE=[0.202185,0.636514,0.174416]
VT=[0.0,0.0,0.0]
ST=[0.004143,0.174416,0.030575]
ET=[0.484866,0.636514,0.405465]
LT=[0.484866,0.636514,0.405465]
DA=[0.410116,0.636514,0.318257]
UT=[0.212074,0.636514,0.174416]
IgT=[0.137325,0.0,0.0]
RO=[0.0,0.0,0.0]
MNT=[0.484866,0.636514,0.405465]
DpT=[0.0,0.0,0.0]
LOC=[0.484866,0.636514,0.405465]
NOM=[0.484866,0.636514,0.405465]
WMC=[0.484866,0.636514,0.405465]
RFC=[0.484866,0.636514,0.405465]
AD=[0.484866,0.636514,0.405465]


# create dataframe with metrics
import pandas as pd

df_controlmetrics = pd.DataFrame({
    'Metric': ['CCT'] * len(CCT) + ['CCM'] * len(CCM) + ['CCA'] * len(CCA)
    + ['Co'] * len(Co) + ["Con"] * len(Con) + ['MCon'] * len(MCon)
    + ["ConE"] * len(ConE)
    + ['ALC'] * len(ALC) + ["ALM"] * len(ALM) + ['ALA'] * len(ALA)
    + ["RLC"] * len(RLC)
    + ['RLM'] * len(RLM) + ["RLA"] * len(RLA),
    'Value': CCT + CCM + CCA + Co + Con + MCon + ConE
    + ALC + ALM + ALM + RLC + RLM + RLA
})

print(df_controlmetrics)





# create dataframe with metrics
import pandas as pd

df_testcodemetrics = pd.DataFrame({
    'Metric': ['LOC'] * len(LOC) + ['NOM'] * len(NOM) + ['WMC'] * len(WMC)
    + ['RFC'] * len(RFC) + ["AD"] * len(AD),
    'Value': LOC + NOM + WMC + RFC + AD
})

# Create a figure and a set of subplots with 1 row and 3 columns
fig, axs = plt.subplots(1, 2, figsize=(18, 6))  # 1 row, 3 columns

# First plot
sns.violinplot(x='Metric', y='Value', data=df_testcodemetrics, inner=None, color="skyblue", ax=axs[0])
sns.boxplot(x='Metric', y='Value', data=df_testcodemetrics, width=0.2, color="white", linewidth=1, boxprops=dict(alpha=0), ax=axs[0])
axs[0].set_title('Violin Plot for Test Code Metrics Dimension')
axs[0].set_xlabel('Metric')
axs[0].set_ylabel('Value')

# Second plot
sns.violinplot(x='Metric', y='Value', data=df_controlmetrics, inner=None, color="purple", ax=axs[1])
sns.boxplot(x='Metric', y='Value', data=df_controlmetrics, width=0.2, color="white", linewidth=1, boxprops=dict(alpha=0), ax=axs[1])
axs[1].set_title('Violin Plot for Process Metrics Dimension')
axs[1].set_xlabel('Metric')
axs[1].set_ylabel('Value')


# Adjust layout
plt.tight_layout()

# Save all plots to a single PDF
plt.savefig("/Users/valeriapontillo/Desktop/ICPC2024_appendix/results/rq1/infoGainRQ1/resultInfoGain/violinPlot/violinPlots1_htmlunitdriver.pdf", format='pdf')



df_testsmell1 = pd.DataFrame({
    'Metric': ['AR'] * len(AR) + ['CI'] * len(CI) + ['CTL'] * len(CTL) + ['DA'] * len(DA) + ["ET"] * len(ET) + ['ECT'] * len(ECT),
    'Value': AR + CI + CTL + DA + ET + ECT
})

df_testsmell2 =pd.DataFrame({ 'Metric': ['GF'] * len(GF) + ['IgT'] * len(IgT) + ["LT"] * len(LT) + ['MG'] * len(MG) + ['MNT'] * len(MNT),
'Value': GF + IgT + LT + MG + MNT})

df_testsmell3 = pd.DataFrame({
    'Metric': ['RA'] * len(RA) + ['RO'] * len(RO) + ["PS"] * len(RP) +
              ['SE'] * len(SE) + ['ST'] * len(ST) + ['UT'] * len(UT),
    'Value': RA + RO + RP +
       SE + ST + UT
})



# Create a figure and a set of subplots with 1 row and 3 columns
fig, axs = plt.subplots(1, 3, figsize=(18, 6))  # 1 row, 3 columns

# First plot
sns.violinplot(x='Metric', y='Value', data=df_testsmell1, inner=None, color="blue", ax=axs[0])
sns.boxplot(x='Metric', y='Value', data=df_testsmell1, width=0.2, color="white", linewidth=1, boxprops=dict(alpha=0), ax=axs[0])
#axs[0].set_title('Violin Plot for Test Smells Dimension 1')
axs[0].set_xlabel('Metric')
axs[0].set_ylabel('Value')

# Second plot
sns.violinplot(x='Metric', y='Value', data=df_testsmell2, inner=None, color="blue", ax=axs[1])
sns.boxplot(x='Metric', y='Value', data=df_testsmell2, width=0.2, color="white", linewidth=1, boxprops=dict(alpha=0), ax=axs[1])
axs[1].set_title('Violin Plot for Test Smells Dimension')
axs[1].set_xlabel('Metric')
axs[1].set_ylabel('Value')

# Third plot
sns.violinplot(x='Metric', y='Value', data=df_testsmell3, inner=None, color="blue", ax=axs[2])
sns.boxplot(x='Metric', y='Value', data=df_testsmell3, width=0.2, color="white", linewidth=1, boxprops=dict(alpha=0), ax=axs[2])
#axs[2].set_title('Violin Plot for Test Smells Dimension 3')
axs[2].set_xlabel('Metric')
axs[2].set_ylabel('Value')

# Adjust layout
plt.tight_layout()

# Save all plots to a single PDF
plt.savefig("/Users/valeriapontillo/Desktop/ICPC2024_appendix/results/rq1/infoGainRQ1/resultInfoGain/violinPlot/violinPlots2_htmlunitdriver.pdf", format='pdf')

'''
plt.figure(figsize=(10, 6))
sns.violinplot(x='Metric', y='Value', data=df_testsmell1, inner=None, color = "blue")
sns.boxplot(x='Metric', y='Value', data=df_testsmell1, width=0.2, color="white", linewidth=1, boxprops=dict(alpha=0))  # Boxplot trasparenti
plt.title('Violin Plot for Test Smells Dimension')
plt.xlabel('Metric')
plt.ylabel('Value')
plt.savefig("/Users/valeriapontillo/Desktop/violinPlot_testsmell1.pdf", format='pdf')
#plt.savefig("/your_path/results/rq1/violinPlot_testsmell1.pdf", format='pdf')

plt.figure(figsize=(10, 6))
sns.violinplot(x='Metric', y='Value', data=df_testsmell2, inner=None, color = "blue")
sns.boxplot(x='Metric', y='Value', data=df_testsmell2, width=0.2, color="white", linewidth=1, boxprops=dict(alpha=0))  # Boxplot trasparenti
plt.title('Violin Plot for Test Smells Dimension')
plt.xlabel('Metric')
plt.ylabel('Value')
plt.savefig("/Users/valeriapontillo/Desktop/violinPlot_testsmell2.pdf", format='pdf')
#plt.savefig("/your_path/results/rq1/violinPlot_testsmell2.pdf", format='pdf')

plt.figure(figsize=(10, 6))
sns.violinplot(x='Metric', y='Value', data=df_testsmell3, inner=None, color = "blue")
sns.boxplot(x='Metric', y='Value', data=df_testsmell3, width=0.2, color="white", linewidth=1, boxprops=dict(alpha=0))  # Boxplot trasparenti
plt.title('Violin Plot for Test Smells Dimension')
plt.xlabel('Metric')
plt.ylabel('Value')
plt.savefig("/Users/valeriapontillo/Desktop/violinPlot_testsmell3.pdf", format='pdf')
#plt.savefig("/your_path/results/rq1/violinPlot_testsmell3.pdf", format='pdf')'''