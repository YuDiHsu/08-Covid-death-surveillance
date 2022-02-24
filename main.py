import datetime
import pandas as pd
import os
import multiprocessing as mp
import pyodbc
import glob
import matplotlib.pylab as plt
import numpy as np
import cx_Oracle
import requests
import xlsxwriter


def death_diag_code(path):
    df = pd.read_csv(path)
    df = df.fillna('')
    columns = list(df.columns)
    aesi_dict = {}
    for ae in df['aesi_diag_name'].values.tolist():
        if ae not in aesi_dict:
            aesi_dict[ae] = {}
            for col in columns[1:]:
                if col not in aesi_dict[ae]:
                    aesi_dict[ae][col] = ''

    for k, v in aesi_dict.items():
        for col in columns[1:]:
            for value in df[col].loc[(df['aesi_diag_name'] == k)]:
                if type(value) == str:
                    v[col] = str(value).replace('.', '').split(', ')
                else:
                    v[col] = value
    return aesi_dict


def select_func(x, v):
    flag = False
    # # determine the length of character to search adr_icd_code
    for n in list(set([len(l) for l in v['adr_icd_code']])):
        if x[0:n] in v['adr_icd_code']:
            flag = True
            break
    # # determine the length of character to search exception
    for m in list(set([len(j) for j in v['exception']])):
        if m:
            if x[0:m] in v['exception']:
                flag = False
                break
        else:
            break
    return flag


def date_convert(x, idno):
    if x:
        try:
            n_x = datetime.datetime.strptime(str(x).replace('-', ''), '%Y%m%d').date()
        except:
            print(f"Error case ID: {idno}, wrong date is {x}")
            x = int(x) - 1
            n_x = datetime.datetime.strptime(str(x).replace('-', ''), '%Y%m%d').date()
        return n_x
    else:
        return x


def exam_data():
    #     df = pd.read_csv(os.path.abspath(os.path.join('.', '2021-05-22 13_06 COVID_ID_LIST.csv')))
    file_path = sorted(glob.glob(os.path.join("/media/sf_Eic02/COVID-19_LAB", "*LIST.csv")), reverse=True)[0]
    df = pd.read_csv(file_path)
    for date in ['SICK_DATE', 'REPORT_DATE']:
        try:
            # df_.loc[:, date] = df_.loc[:, date].apply(lambda x: date_convert(x))
            df.loc[:, date] = df.apply(lambda x: date_convert(x[date], x['IDNO']), axis=1)
        except EOFError as e:
            print('EOFError', e)
        except Exception as Ex:
            print('Exception', Ex)
    df.loc[:, 'positive'] = 1
    return df


def load_raw_data(sftp_filename_list, division_name):
    df = pd.DataFrame()
    for f_n in sftp_filename_list:
        if '.ipynb' not in f_n:
            #             print(f'{f_n} is loading')
            df_ = pd.read_csv(os.path.join('/home/rserver/task/COVID_DEATH_SURVEILLANCE_LINE_NOTIFY/', 'sftp_download_folder', f'{division_name}_sftp', f_n), header=[0], low_memory=False)
            df_ = df_.fillna('')
            df_ = df_.astype(str)
            if division_name == 'ICPREG':
                df_.loc[:, '醫療院所代碼'] = ''
                df_.loc[:, '就醫類別'] = ''
                df_ = df_[['身分證號', '出生日期', '醫療院所代碼', '就醫日期', '就醫類別', '主診斷碼', '次診斷碼1', '次診斷碼2', '次診斷碼3', '次診斷碼4', '次診斷碼5']]
            #             print('Start to splice data to parts and process')

            df = df.append(df_)
    print("NHI data loading completed.")
    return df


def process_data(df_):
    for date in ['出生日期', '就醫日期']:
        try:
            # df_.loc[:, date] = df_.loc[:, date].apply(lambda x: date_convert(x))
            df_.loc[:, date] = df_.apply(lambda x: date_convert(x[date], x['身分證號']), axis=1)
        except EOFError as e:
            print('EOFError', e)
        except Exception as Ex:
            print('Exception', Ex)

    death_diag_dict = death_diag_code(os.path.join('/home/rserver/task/COVID_DEATH_SURVEILLANCE_LINE_NOTIFY/', 'covid_death_icd_code.csv'))
    df_select = pd.DataFrame()

    try:
        for col in ['主診斷碼', '次診斷碼1', '次診斷碼2', '次診斷碼3', '次診斷碼4', '次診斷碼5']:
            for k, v in death_diag_dict.items():
                temp_df = df_.loc[df_[col].apply(lambda x: select_func(x, v))]
                temp_df['syndrome'] = k
                df_select = df_select.append(temp_df)
    except EOFError as e:
        print(e)
    except Exception as Ex:
        print(Ex)
    df_select = df_select.reset_index(drop=True)

    return df_select


def get_raw_data(file_name, code):
    dsn = cx_Oracle.makedsn('192.168.170.52', '1561', service_name='DW')

    conn = cx_Oracle.connect(
        user='sas',
        password='ueCr5brAD6u4rAs62t9a',
        dsn=dsn,
        encoding='UTF8',
        nencoding='UTF8'
    )

    c = conn.cursor()

    c.execute(code)

    desc = c.description
    col_name_list = []
    for s in desc:
        col_name_list.append(s[0])

    data_list = c.fetchall()
    conn.close()

    temp_df = pd.DataFrame(data_list, columns=col_name_list)
    #     temp_df.to_csv(os.path.abspath(os.path.join('.', f'{file_name}.csv')), index=False)
    data_list.append(temp_df)

    return temp_df


def get_death_raw_data():
    server = '192.168.170.60\mssqlserver1'
    database = 'DOHDied'
    username = 'cdceic'
    password = '6&*cTNwg'
    string = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password};"
    cnxn = pyodbc.connect(string)
    df_death = pd.read_sql("select DhDeathID,DhDeathName,convert(datetime,DhDeathDay) as DeathDate from DohDied where convert(datetime,DhDeathDay) >= '2021-04-20'", cnxn)
    # df_death = pd.read_sql("select DhDeathName, DhDeathID, DhBirthDay_Day, DhDeathDay, DhHostName, DhDeathType, DhCauseNameA1,DhCauseNameB1,DhCauseNameC1,DhCauseNameD1,DhCityName,DhDeathCityName from DohDied where
    # DhDeathDay >= '2021/01/01' and DhDeathDay <='2021/12/31'",cnxn)
    df_death["DhDeathID"] = df_death["DhDeathID"].str.replace(" ", "")
    df_death["DhDeathName"] = df_death["DhDeathName"].str.replace(" ", "")

    return df_death


def get_covid_confirm_cases():
    files = glob.glob("/media/sf_Eic02/COVID-19_LAB/*_ID_LIST.csv")
    files.sort(key=os.path.getmtime)
    df_id_list = pd.read_csv(files[-1])

    return df_id_list


def render_mpl_table(data, col_width=10.0, row_height=0.625, font_size=12,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')
    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in mpl_table._cells.items():
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])
    return ax.get_figure(), ax


def write_xlsx(data_list: list, workbook_name, worksheet_name: list, col_len_width=None):
    all_col_len_list = []
    for single_data_pd in data_list:
        data_col_len_list = []
        for col_name in list(single_data_pd):
            if len(single_data_pd[col_name]):
                data_col_len_list.append(max(single_data_pd[col_name].apply(lambda x: len(str(x)))))
            else:
                data_col_len_list.append(1)
        all_col_len_list.append(data_col_len_list)

    workbook = xlsxwriter.Workbook(workbook_name)
    for idx, d in enumerate(zip(data_list, worksheet_name)):
        if type(d[0]) is pd.DataFrame:
            pd_list = d[0].fillna('').values.tolist()
            pd_col_list = list(d[0])
            col_param_list = []
            header = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'size': 12})
            for col_name in pd_col_list:
                col_param_list.append({'header': col_name, 'format': header})
            sheet = workbook.add_worksheet(d[1])
            if len(pd_list):
                sheet.add_table(0, 0, len(pd_list), len(pd_list[0]) - 1, {'data': pd_list, 'autofilter': True, 'columns': col_param_list})
            for _ in range(len(pd_list) + 1):
                sheet.set_row(_, 30, cell_format=header)
            for idxx, l in enumerate(zip(all_col_len_list[idx], pd_col_list)):
                sheet.set_column(idxx, idxx, max(l[0], len(l[1])) * 2)
    workbook.close()


if __name__ == "__main__":
    # # ------------------------ 健保資料 ------------------------
    mp.freeze_support()
    # # # Terminate pandas loc. Warning
    pd.options.mode.chained_assignment = None

    #     test_icdx_filename_list = [
    #         'ICDX_20210501.txt', 'ICDX_20210502.txt', 'ICDX_20210503.txt',
    #         'ICDX_20210504.txt', 'ICDX_20210505.txt', 'ICDX_20210506.txt', 'ICDX_20210507.txt', 'ICDX_20210508.txt', 'ICDX_20210509.txt',
    #         'ICDX_20210510.txt', 'ICDX_20210511.txt', 'ICDX_20210512.txt', 'ICDX_20210513.txt', 'ICDX_20210514.txt', 'ICDX_20210515.txt',
    #         'ICDX_20210516.txt', 'ICDX_20210517.txt', 'ICDX_20210518.txt',
    #         'ICDX_20210519.txt'
    #     ]
    #     test_icpreg_filename_list = [
    #         'ICPREG_20210501.txt', 'ICPREG_20210502.txt', 'ICPREG_20210503.txt', 'ICPREG_20210504.txt',
    #         'ICPREG_20210505.txt', 'ICPREG_20210506.txt', 'ICPREG_20210507.txt', 'ICPREG_20210508.txt', 'ICPREG_20210509.txt',
    #         'ICPREG_20210510.txt', 'ICPREG_20210511.txt', 'ICPREG_20210512.txt', 'ICPREG_20210513.txt', 'ICPREG_20210514.txt',
    #         'ICPREG_20210515.txt', 'ICPREG_20210516.txt', 'ICPREG_20210517.txt', 'ICPREG_20210518.txt',
    #         'ICPREG_20210519.txt'
    #     ]

    icdx_list = os.listdir(os.path.join('/home/rserver/task/COVID_DEATH_SURVEILLANCE_LINE_NOTIFY/', 'sftp_download_folder', 'ICDX_sftp'))
    icpreg_list = os.listdir(os.path.join('/home/rserver/task/COVID_DEATH_SURVEILLANCE_LINE_NOTIFY/', 'sftp_download_folder', 'ICPREG_sftp'))

    df_raw = pd.concat([load_raw_data(icdx_list, 'ICDX'), load_raw_data(icpreg_list, 'ICPREG')])
    df_raw = df_raw.reset_index(drop=True)
    num_processes = int(mp.cpu_count() / 2)

    chunk_size = int(df_raw.shape[0] / num_processes)
    chunks = [df_raw.loc[df_raw.index[i:i + chunk_size]] for i in range(0, df_raw.shape[0], chunk_size)]
    pool = mp.Pool(processes=num_processes)
    result_pd_list = pool.map(process_data, chunks)
    df_nhi = pd.concat(result_pd_list).reset_index(drop=True)

    df_exam = exam_data()
    df_exam = df_exam[['IDNO', 'SICK_DATE', 'positive', 'REPORT']]
    df_nhi = df_nhi.rename(columns={'身分證號': 'IDNO', '就醫日期': 'NHI_DEATH'})

    df_merge = df_nhi.merge(df_exam, how='left', on='IDNO')

    df_p = df_merge.loc[(df_merge['positive'] == 1) & (df_merge['syndrome'] == 'Death (sudden death)')].drop_duplicates(subset=['IDNO', 'NHI_DEATH']).reset_index(drop=True)
    df_p.loc[:, 'REPORT'] = df_p.loc[:, 'REPORT'].astype(int)
    df_p.loc[:, 'STATIS_DEATH'] = ''
    df_p.loc[:, 'NIDRS_DEATH'] = ''
    #     df_p = df_p[['IDNO', 'REPORT', 'NIDRS_DEATH', 'STATIS_DEATH', 'NHI_DEATH']]
    df_p = df_p[['IDNO', 'REPORT', 'NHI_DEATH']]

    # # ------------------------ 健保資料 ------------------------

    # # ------------------------ 部死亡檔 ------------------------
    df_death = get_death_raw_data().drop_duplicates()
    df_id_list = get_covid_confirm_cases()[['IDNO', 'NAME', 'REPORT']]
    covid_death_list = pd.merge(df_id_list, df_death, left_on="IDNO", right_on="DhDeathID").drop('DhDeathName', axis=1)
    covid_death_list.loc[:, 'NIDRS_DEATH'] = ''
    covid_death_list.loc[:, 'NHI_DEATH'] = ''
    covid_death_list = covid_death_list.rename(columns={'DeathDate': 'STATIS_DEATH'})
    #     df_statis = covid_death_list[['IDNO', 'REPORT', 'NIDRS_DEATH', 'STATIS_DEATH', 'NHI_DEATH']]
    df_statis = covid_death_list[['IDNO', 'REPORT', 'STATIS_DEATH']]
    df_statis.loc[:, 'STATIS_DEATH'] = df_statis.loc[:, 'STATIS_DEATH'].apply(lambda x: datetime.datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S').date())
    # covid_death_list.to_csv('/home/rserver/task/DAILY_COVID_COUNT_LINE_NOTIFY/covid_death_list.csv', index=False)
    # # ------------------------ 部死亡檔 ------------------------

    # # ------------------------ 法傳死亡檔 ------------------------
    sql_code = dict(tmp_data="select IDNO,NAME,REPORT,SICK_AGE,BIRTHDAY,GENDER,REPORT_DATE,IMMIGRATION,SICK_DATE,DEATH_DATE,RESULT_DESC,NATIONALITY from "
                             "(select t2.IDNO, t2.NAME, t3.report, t3.SICK_AGE, t3.BIRTHDAY, t2.GENDER, t3.REPORT_DATE, case t3.IMMIGRATION when 0 then 'domestic' when 1 then 'imported' "
                             "when 8 then 'unknown' end as IMMIGRATION, t3.SICK_DATE, t3.DEATH_DATE, t4.RESULT_DESC, t5.COUNTRY_NAME as NATIONALITY, ROW_NUMBER() Over (Partition By t3.IDNO Order By t3.REPORT_DATE Desc) "
                             "As Sort from CDCDW.DWS_SAMPLE_DETAIL t1 left join CDCDW.USV_INDIVIDUAL_SAS t2 on t1.INDIVIDUAL = t2.INDIVIDUAL left join CDCDW.USV_DWS_REPORT_DETAIL_EIC_UTF8 t3 "
                             "on t2.IDNO = t3.IDNO left join CDCDW.DIM_RESULT t4 on t1.RESULT = t4.RESULT left join CDCDW.DIM_COUNTRY t5 on t3.NATIONALITY = t5.COUNTRY "
                             "where t1.disease in ('19CoV','SICoV','SICV2') and t1.SAMPLE_DATE >= TO_DATE('2020/1/15', 'YYYY/MM/DD') and t1.RESULT = 5 "
                             "and t3.disease in ('19CoV','SICoV','SICV2') and t3.report_date >= TO_DATE('2021/4/20', 'YYYY/MM/DD') and t3.DEATH_DATE is not null) "
                             "where SORT = 1")

    df_re = get_raw_data('tmp_data', sql_code['tmp_data'])
    #     df_re.loc[:, 'STATIS_DEATH'] = ''
    #     df_re.loc[:, 'NHI_DEATH'] = ''
    df_re = df_re.rename(columns={'DEATH_DATE': 'NIDRS_DEATH'})
    df_re.loc[:, 'NIDRS_DEATH'] = df_re.loc[:, 'NIDRS_DEATH'].apply(lambda x: datetime.datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S').date())
    #     df_re = df_re[['IDNO', 'REPORT', 'NIDRS_DEATH', 'STATIS_DEATH', 'NHI_DEATH']]
    df_re = df_re[['IDNO', 'REPORT', 'NIDRS_DEATH']]
    # # ------------------------ 法傳死亡檔 ------------------------

    # # ------------------------ 出圖 ------------------------
    df_all = pd.concat([df_p, df_statis, df_re]).drop_duplicates(subset=['IDNO']).reset_index(drop=True)
    df_all = df_all[['IDNO', 'REPORT']]
    df_all = df_all.merge(df_re, how='left', on=['IDNO']).merge(df_statis, how='left', on=['IDNO']).merge(df_p, how='left', on=['IDNO']).reset_index(drop=True)
    df_all = df_all.loc[:, ~df_all.columns.duplicated()]
    df_all = df_all.drop('REPORT_y', axis=1).fillna('').rename(columns={'REPORT_x': 'REPORT'})

    fig, ax = render_mpl_table(df_all, header_columns=0, col_width=2.0)
    fig.savefig(os.path.abspath(os.path.join('/home/rserver/task/COVID_DEATH_SURVEILLANCE_LINE_NOTIFY/', "death_surveillace.png")))
    # # ------------------------ 出圖 ------------------------
    work_book_path = os.path.abspath(os.path.join('.', 'death_surveillance.xlsx'))

    write_xlsx([df_all], work_book_path, ['death_surveillance'])

    # # ------------------------ LINE NOTIFY ------------------------
    headers = {"Authorization": "Bearer " + "ivZDptYcHaf0t5fCVi2C8zmbIsOuAybDeRMi3P4Lchg",
               }
    message = f"\nDeath Surveillance"
    files = {'imageFile': open(os.path.abspath(os.path.join('/home/rserver/task/COVID_DEATH_SURVEILLANCE_LINE_NOTIFY/', "death_surveillace.png")), 'rb')}
    params = {"message": message}
    r = requests.post("https://notify-api.line.me/api/notify", headers=headers, params=params, files=files)


