import pandas as pd
def get_prophet_result(df):
    df = df.sort_values(by='score', ascending=False, ignore_index=True)
    # df['decoy'] = np.where(df['label'] == 1, 0, 1)
    df['decoy'] = df['label']

    target_num = (df.decoy == 0).cumsum()
    decoy_num = (df.decoy == 1).cumsum()

    target_num[target_num == 0] = 1
    decoy_num[decoy_num == 0] = 1
    df['q_value'] = decoy_num / target_num
    df['q_value'] = df['q_value'][::-1].cummin()

    # log
    id_01 = ((df['q_value'] <= 0.01) & (df['decoy'] == 0)).sum()

    #  conservative FDR estimate
    filtered_df = df[(df['q_value'] <= 0.01)  & (df['decoy'] == 0)][
        ['precursor', 'label', 'score', 'rt_shift', 'irt', 'rt', 'delta_rt', 'q_value']]
    return filtered_df, id_01, df
data = pd.read_csv(r'D:\00.Westlake\00.Phd_candidate\00.project\00.DIA_BERT_TimsTOF\3D_code\666.result\20250902_OnlyTimsModel\CAD_score_data.csv')
filtered_df, id_01, df = get_prophet_result(data)
len(filtered_df)