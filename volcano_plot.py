# Databricks notebook source
# MAGIC %run ../initialization/notebook_setup_plotting

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## CKD biomarkers associated with disease over time

# COMMAND ----------


def load_raw_data_fc(grand_df, group, timepoint):
    grand_query = grand_df.query(f'(`Treatment ID` == "{group}")').query(
        f'Visit == "{timepoint}"'
    )

    serum_cols = [col for col in grand_query.columns if 'serum' in col]

    grand_melt = grand_query.melt(
        id_vars=['animal', 'Visit', 'Treatment', 'Progressive'],
        value_vars=serum_cols,
        var_name="metabolite",
        value_name="level",
    )

    grouped = (
        grand_melt.groupby(['metabolite', 'Progressive'])['level']
        .median()
        .unstack()
        .reset_index()
        .rename(columns={0: 'non_progressive', 1: 'progressive'})
    )

    grouped["fold_change"] = grouped["progressive"] / grouped["non_progressive"]
    grouped["log2_fold_change"] = np.log2(grouped["fold_change"])
    return grouped


# COMMAND ----------
grand_df = (
    DataFileReader(
        file_path='s3://hypoxia-pilot-study/04_data_analysis/grand_table/grand_serum_raw.csv'
    )
    .load()
    .assign(Visit=lambda row: row['Visit'].str.replace(' ', ''))
)

load_raw_data_fc(grand_df, 'T03', 'Day168').display()
# COMMAND ----------

for timepoint in ['Day0', 'Day168', 'Day336']:
    res_path = f'/dbfs/mnt/client-153sap23p001-hypoxia-pilot-study/04_data_analysis/results/serum_placebo_ttest_{timepoint}.csv'

    res = DataFileReader(file_path=res_path).load(data_format="csv")
    res['ID'] = None
    res['p_bh'] = smt.multipletests(res["p.value"], method="fdr_bh")[1]  # noqa: F821

    raw_data_fc = load_raw_data_fc(grand_df, 'T01', timepoint)
    res_fc = res.merge(raw_data_fc, on='metabolite', how='outer')

    print(res_fc.shape)

    print('Threshold: p value < 1e-4')
    print(res_fc.query('`p.value` < 1e-4').shape)

    vplot = VolcanoPlot(res_fc, p_thres={"P": 1e-4})
    print(
        vplot.draw(
            x="log2_fold_change",
            point_size=5,
            x_axis_text_size=12,
            y_axis_text_size=12,
            fig_size=(1.6, 0.5),
        )
    )

    print('Threshold: p_bh < 0.1')
    print(res_fc.query('p_bh < 0.1').shape)
    vplot = VolcanoPlot(res_fc, p_thres={"p_bh": 0.1})
    print(
        vplot.draw(
            x="log2_fold_change",
            point_size=5,
            x_axis_text_size=12,
            y_axis_text_size=12,
            fig_size=(1.6, 0.5),
        )
    )


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## CKD biomarkers changing in response to treatment

# COMMAND ----------


def load_raw_data_fc_paired(grand_df, group, timepoint):
    grand_query = grand_df.query(f'(`Treatment ID` == "{group}")').query(
        f'Visit == "{timepoint}" | Visit == "Day0"'
    )

    serum_cols = [col for col in grand_query.columns if 'serum' in col]

    grand_melt = grand_query.melt(
        id_vars=['animal', 'Visit', 'Treatment', 'Progressive'],
        value_vars=serum_cols,
        var_name="metabolite",
        value_name="level",
    )

    grouped = (
        grand_melt.groupby(['metabolite', 'Visit'])['level']
        .median()
        .unstack()
        .reset_index()
    )

    grouped["fold_change"] = grouped[f"{timepoint}"] / grouped["Day0"]
    grouped["log2_fold_change"] = np.log2(grouped["fold_change"])
    return grouped


# COMMAND ----------

load_raw_data_fc_paired(grand_df, 'T03', 'Day168').display()

# COMMAND ----------
# paired t-test
for timepoint in ['Day168', 'Day336']:
    res_path = f'/dbfs/mnt/client-153sap23p001-hypoxia-pilot-study/04_data_analysis/results/serum_treatment_paired_ttest_Day0vs{timepoint}.csv'

    res = DataFileReader(file_path=res_path).load(data_format="csv")
    res['ID'] = None
    res['p_bh'] = smt.multipletests(res["p.value"], method="fdr_bh")[1]  # noqa: F821

    raw_data_fc = load_raw_data_fc_paired(grand_df, 'T03', timepoint)
    res_fc = res.merge(raw_data_fc, on='metabolite', how='outer')

    print(res_fc.shape)

    print('Threshold: p value < 1e-4')
    print(res_fc.query('`p.value` < 1e-4').shape)

    vplot = VolcanoPlot(res_fc, p_thres={"P": 1e-4})
    print(
        vplot.draw(
            x="log2_fold_change",
            point_size=5,
            x_axis_text_size=12,
            y_axis_text_size=12,
            fig_size=(1.6, 0.5),
        )
    )

    print('Threshold: p_bh < 0.1')
    print(res_fc.query('p_bh < 0.1').shape)
    vplot = VolcanoPlot(res_fc, p_thres={"p_bh": 0.1})
    print(
        vplot.draw(
            x="log2_fold_change",
            point_size=5,
            x_axis_text_size=12,
            y_axis_text_size=12,
            fig_size=(1.6, 0.5),
        )
    )


# COMMAND ----------
# Independent t-test
for timepoint in ['Day168', 'Day336']:
    res_path = f'/dbfs/mnt/client-153sap23p001-hypoxia-pilot-study/04_data_analysis/results/serum_treatment_independent_ttest_Day0vs{timepoint}.csv'

    res = DataFileReader(file_path=res_path).load(data_format="csv")
    res['ID'] = None
    res['p_bh'] = smt.multipletests(res["p.value"], method="fdr_bh")[1]  # noqa: F821

    raw_data_fc = load_raw_data_fc_paired(grand_df, 'T03', timepoint)
    res_fc = res.merge(raw_data_fc, on='metabolite', how='outer')

    print(res_fc.shape)

    print('Threshold: p value < 1e-4')
    print(res_fc.query('`p.value` < 1e-4').shape)

    vplot = VolcanoPlot(res_fc, p_thres={"P": 1e-4})
    print(
        vplot.draw(
            x="log2_fold_change",
            point_size=5,
            x_axis_text_size=12,
            y_axis_text_size=12,
            fig_size=(1.6, 0.5),
        )
    )

    print('Threshold: p_bh < 0.1')
    print(res_fc.query('p_bh < 0.1').shape)
    vplot = VolcanoPlot(res_fc, p_thres={"p_bh": 0.1})
    print(
        vplot.draw(
            x="log2_fold_change",
            point_size=5,
            x_axis_text_size=12,
            y_axis_text_size=12,
            fig_size=(1.6, 0.5),
        )
    )
