# Databricks notebook source
# MAGIC %pip install hydra-core

# COMMAND ----------

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats
from statsmodels.stats import multitest


from hydra import compose, initialize
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png2x')

plt.rcParams['figure.dpi'] = 150
set_matplotlib_formats('png')

# COMMAND ----------

# Read configuration
with initialize(version_base=None,config_path="./config"):
    cfg = compose(config_name="config_misame3.yaml")

# COMMAND ----------

def get_fontsizes():
    plt.rc('font', size=12)          # controls default text sizes
    plt.rc('axes', titlesize=14)     # fontsize of the axes title
    plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=10)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=10)    # fontsize of the tick labels
    plt.rc('legend', fontsize=12)    # legend fontsize
    plt.rc('figure', titlesize=16)  # fontsize of the figure title



# COMMAND ----------

def volcano_plot(df, title,x_col='LR_Effect_Size', y_col='LR_P_Value',FDR_threshold='fdr_bh',highlight='no',pavlue_th='no',color='pink'):
    plt.figure(figsize=(5,5))
    if pavlue_th == 'no':
    
      # Specify the desired FDR level
      desired_fdr = 0.1

      # Sort p-values in ascending order
      sorted_unadjusted_p_values = sorted(df[y_col])
      rejected, p_values_corrected, _, _ = multitest.multipletests(sorted_unadjusted_p_values, method=FDR_threshold)

      # Find the index of the adjusted p-value closest to the desired FDR level
      closest_index = np.argmax((p_values_corrected >= desired_fdr))

      # Find the corresponding unadjusted p-value
      closest_unadjusted_p_value = sorted_unadjusted_p_values[closest_index]

    else:
      closest_unadjusted_p_value = pavlue_th


    print('unadjusted p_value threshold: ' + str(closest_unadjusted_p_value))   
    print('significant markers: '+str(len(df[(df[y_col] < closest_unadjusted_p_value) 
                     ]['feature_label'].unique())))


    # plt.xlim(x_min, x_max)
    if highlight =='no':
      group = ['significant biomarkers' if p_val < closest_unadjusted_p_value else 'total biomarkers' for p_val in df[y_col]]
      df['group']=group
      # Create a volcano plot
      sns.scatterplot(x=x_col, y= -np.log10(df[y_col]),
                      # hue=(-np.log10(df[y_col]) < -np.log10(closest_unadjusted_p_value)),
                      hue=group,
                      palette={'total biomarkers': 'grey', 'significant biomarkers': 'blue'},
                      data=df)

    #   legend = plt.legend(bbox_to_anchor=(1.05, 0.9), loc='upper left')



    if highlight!='no':
        print('significant PD markers: '+str(len(df[(df[y_col] < closest_unadjusted_p_value) 
                     & (df.feature_label.isin(highlight))]['feature_label'].unique())))
        group = ['significant biomarkers' if p_val < closest_unadjusted_p_value else 'total biomarkers' for p_val in df[y_col]]
        df['group']=group
        df.loc[df.feature_label.isin(highlight),'group']='PD biomarkers'


        sns.scatterplot(x=x_col, y= -np.log10(df[y_col]),
                        hue='group',
                        hue_order=['PD biomarkers','significant biomarkers','total biomarkers'],
                        style='group',
                        palette={
                            'total biomarkers': 'grey',
                            'significant biomarkers': '#01579B',
                            # 'significant biomarkers': 'grey',
                            'PD biomarkers': color,
                        },
                        sizes=50,
                        markers={
                            'total biomarkers': '.',
                            'significant biomarkers': '.',
                            'PD biomarkers': 'o'},
                        #   sizes={
                            #   'total biomarkers': 100,
                            #   'significant biomarkers': 100,
                            #   'validated biomarkers': 155},
                        alpha=0.5,
                        data=df)
        
        sns.scatterplot(x=x_col, y= -np.log10(df[y_col]),

                
                #  color= '#01579B',# blue
                    color= color,
                data=df[df.feature_label.isin(highlight)])
    # legend = plt.legend(bbox_to_anchor=(1.05, 0.9), loc='upper left')
    # Decrease the number of x ticks
    # x_ticks = [-1, -0.5, 0, 0.5, 1]  # Specify the tick locations
    # x_tick_labels = [-1, -0.5, 0, 0.5, 1]  # Specify the tick labels

    # plt.xticks(x_ticks, x_tick_labels, fontsize=18)  # Set the tick 
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    plt.tight_layout()

    # Add labels and title
    plt.xlabel(x_col,fontsize=20)
    plt.ylabel('-log10({})'.format(y_col),fontsize=20)
    plt.title(title,fontsize=20)
    # plt.legend(['Significant small molecules', 'small molecules'])

    # Add a horizontal line for significance threshold
    plt.axhline(-np.log10(closest_unadjusted_p_value))
    plt.show()
    return closest_unadjusted_p_value

# COMMAND ----------

for independent_variable in cfg.analysis.independent_variable_list:
    # Load regression result DataFrame (adjust this based on your actual result storage)
    result_df = pd.read_csv(f'/dbfs/mnt/client-002sap21p015-imic/04_data_analysis/results/hydra_linear_regression_res_{cfg.analysis.time_point}_{independent_variable}.csv')

    # Create and show the volcano plot
    # create_and_show_volcano_plot(result_df)
    volcano_plot(result_df, x_col='LR_Effect_Size', y_col='LR_P_Value',FDR_threshold='fdr_bh',highlight='no',pavlue_th='no',color='pink',title = independent_variable)



# COMMAND ----------

# MAGIC %md # check result

# COMMAND ----------

cfg.analysis.independent_variable_list

# COMMAND ----------

independent_variable = 'c_haz0'
hydra_result_df = pd.read_csv(f'/dbfs/mnt/client-002sap21p015-imic/04_data_analysis/results/hydra_linear_regression_res_{cfg.analysis.time_point}_{independent_variable}.csv')

# COMMAND ----------

hydra_result_df.display()

# COMMAND ----------

result_df = pd.read_csv(f'/dbfs/mnt/client-002sap21p015-imic/04_data_analysis/results/linear_regression_res_{cfg.analysis.time_point}_{independent_variable}.csv')
result_df.display()

# COMMAND ----------


