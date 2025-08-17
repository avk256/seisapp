import git
from git import Repo
from git.exc import GitCommandError

repo_url = "https://github.com/avk256/seisproc.git"
clone_path = "./seisproc"

try:
    Repo.clone_from(repo_url, clone_path)
    print("Репозиторій успішно клоновано!")
except GitCommandError as e:
    print("❌ Помилка клонування репозиторію")

import seisproc.seisproc as ssp

import streamlit as st
import pandas as pd
import numpy as np

import statsmodels

from itertools import chain

@st.cache_data
def load_file(file_name):
    # st.write("Завантаження...")  # Це буде видно лише 1 раз
    df = pd.read_csv(file_name, header=None, sep=';')
    # ! Остання цифра позначає номер сейсмометра !!!
    df.columns = ['X1','Y11','Y21','Z1','X2','Y12','Y22','Z2','X3','Y13','Y23','Z3']
    return df

def drop_columns_with_suffixes(df, suffixes):
    """
    Видаляє всі колонки датафрейму, які закінчуються на будь-який із заданих суфіксів.
    Для колонок, що починаються з 'Y', видаляє ті, в яких передостанній символ входить до суфіксів.

    Параметри:
        df (pd.DataFrame): вхідний датафрейм
        suffixes (list of str): список суфіксів для видалення

    Повертає:
        pd.DataFrame: новий датафрейм без відповідних колонок
    """
    cols_to_drop = []

    for col in df.columns:
        if col.startswith('Y') and len(col) >= 2:
            if col[-1] in suffixes:
                cols_to_drop.append(col)
        elif any(col.endswith(suffix) for suffix in suffixes):
            cols_to_drop.append(col)

    return df.drop(columns=cols_to_drop)

def filter_columns_by_prefixes(column_names, prefixes):
    """
    Фільтрує назви колонок, залишаючи лише ті, що починаються з заданих префіксів.

    Параметри:
        column_names (list of str): список назв колонок
        prefixes (list of str): список префіксів, за якими здійснюється відбір

    Повертає:
        list of str: список колонок, що починаються з заданих префіксів
    """
    return [col for col in column_names if any(col.startswith(prefix) for prefix in prefixes)]


###############################################################################

st.set_page_config(page_title="SeisApp", layout="wide")
st.title("Аналіз даних сейсмометрів")

if "counter" not in st.session_state:
    st.session_state.count = 0

print(st.session_state.count)
# === Глобальні параметри, які впливають на інші вкладки ===
with st.sidebar:
    st.header("⚙️ Загальні налаштування")
    
    st.title("📥 Завантаження серій даних")

    
    # 1. Завантаження кількох CSV файлів
    uploaded_files = st.file_uploader(
        "Оберіть один або кілька CSV-файлів, які містять дані серій",
        type="csv",
        accept_multiple_files=True
    )

    dfs = {}  # ключ: ім'я файлу, значення: DataFrame
    
    # if st.button("Завантажити файли"):
    # Зчитування файлів у словник DataFrame'ів
    if "dfs" not in st.session_state:
        st.session_state.dfs = dfs
    
    n_seism = 0
    if "n_seism" not in st.session_state:
        st.session_state.n_seism = n_seism
        
    geoph_list = []
    if "geoph_list" not in st.session_state:
        st.session_state.geoph_list = geoph_list

    def_geoph_list = []
    if "def_geoph_list" not in st.session_state:
        st.session_state.def_geoph_list = def_geoph_list
        
    im_geoph_list = ['im_power']
    if "im_geoph_list" not in st.session_state:        
        st.session_state.im_geoph_list = im_geoph_list
        
    true_indices = []
    if "true_indices" not in st.session_state:      
        st.session_state.true_indices = true_indices


        
    if uploaded_files and len(st.session_state.dfs)==0:
        for file in uploaded_files:
            dfs[file.name] = load_file(file)
            st.session_state.n_seism = int(len(dfs[file.name].columns)/4)
        st.write("Завантаження файлів...")    
        st.session_state.dfs = dfs
        st.session_state.count = st.session_state.count + 1  
        

    
    if len(st.session_state.dfs)>0:
        # Створення чекбоксів та збереження їх значень
        siesm_states = []
        
        for i in range(st.session_state.n_seism):
            state = st.checkbox(f"Сейсмометр {i+1}", key=f"сейсмометр_{i}", value=True)
            siesm_states.append(state)

        # Показати результати
        st.write("Стан кожного сейсмометра:")
        st.write(siesm_states)
        
        
        st.session_state.true_indices = [str(i+1) for i, val in enumerate(siesm_states) if val is True]
        
        st.session_state.im_geoph_list = ['im_power' + str(i+1) for i, val in enumerate(siesm_states) if val is True] + ['im_power']
        
        if False in siesm_states:
            false_indices = [str(i+1) for i, val in enumerate(siesm_states) if val is False]
            for filename, df in st.session_state.dfs.items():
                st.session_state.dfs[filename] = drop_columns_with_suffixes(df, false_indices)
                st.session_state.geoph_list = list(df.columns)
        else:
            for filename, df in st.session_state.dfs.items():
                st.session_state.geoph_list = list(df.columns)
       
        st.session_state.def_geoph_list = filter_columns_by_prefixes(st.session_state.geoph_list, ['X', 'Z'])
                
            

            
            
            

    if len(st.session_state.dfs)>0:
    
        fs = st.number_input("🔻 Частота дискретизації", min_value=800.0, value=800.0, step=10.0, key='freq')
        min_freq = 0
        min_freq = 0

        # ================ Смуговий фільтр =====================================
        
        st.subheader("🎚️ Смуговий фільтр")
        # Поля для введення мінімальної та максимальної частоти
        col1, col2 = st.columns(2)
        with col1:
            min_freq = st.number_input("🔻 min частота", min_value=0.0, value=20.0, step=10.0, key='min_freq')
        with col2:
            max_freq = st.number_input("🔺 max частота", min_value=0.0, value=50.0, step=10.0, key='max_freq')
        
        # Кнопка для запуску фільтрації
        if st.button("⚙️ Фільтрувати"):
            st.success(f"Застосовується фільтр: від {min_freq} до {max_freq} Гц")
            # Тут можна викликати функцію фільтрації
            for key, data in st.session_state.dfs.items():
                st.session_state.dfs[key] = ssp.filter_dataframe(data, min_freq, max_freq, fs)
                
        

        # ================== Часове вікно =====================================
        
        st.subheader("🎚️ Часове вікно")
        # Поля для введення мінімальної та максимальної частоти
        col1, col2 = st.columns(2)
        with col1:
            min_time = st.number_input("🔻 min час", min_value=0.0, value=1.0, step=1.0, format="%.4f", key='min_time')
        with col2:
            max_time = st.number_input("🔺 max час", min_value=0.0, value=10.0, step=1.0, format="%.4f", key='max_time')
        
        # Кнопка для запуску фільтрації
        if st.button("⚙️ Застосувати вікно"):
            st.success(f"Застосовується часове вікно: від {min_time} до {max_time} с")
            # Тут можна викликати функцію фільтрації
            for key, data in st.session_state.dfs.items():
                dfs[key] = ssp.cut_dataframe_time_window(data, fs, min_time, max_time)
                
            st.session_state.dfs = dfs
            print(st.session_state.dfs[key].describe())
            # breakpoint()
        # ===================== Детренд =======================================
   
        st.subheader("🎚️ Операція віднімання тренду")
        # Кнопка для запуску детренду
        if st.button("⚙️ Застосувати детренд"):
            st.success("Застосовується детренд")
            # Тут можна викликати функцію фільтрації
            for key, data in st.session_state.dfs.items():
                print(data.describe())
                # breakpoint()

                st.session_state.dfs[key] = ssp.detrend_dataframe(data)
    
    
    
# dfs = dfs

# if st.button("⚙️ Розрахувати"):


# === Вкладки ===
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13 = st.tabs(["📈 Дані", 
                                                                                     "📊 Графіки серій", 
                                                                                     "📊 Графіки між серіями", 
                                                                                     "Спектр", 
                                                                                     "PSD", 
                                                                                     "RMS за PSD", 
                                                                                     "Затримки", 
                                                                                     "Когерентність",
                                                                                     "Когерентне підсумовування",
                                                                                     "Когерентне віднімання", 
                                                                                     "ВПФ", 
                                                                                     "Азимутальна когерентність",
                                                                                     "Математична модель сонара"])



# === Вкладка 1: Дані ===
with tab1:

        dfs_vpf = {}  # ключ: ім'я файлу, значення: DataFrame
        dfs_sum = {}  # ключ: ім'я файлу, значення: DataFrame
        dfs_sub = {}  # ключ: ім'я файлу, значення: DataFrame
        
        if "dfs_vpf" not in st.session_state:
            st.session_state.dfs_vpf = dfs_vpf
        if "dfs_sum" not in st.session_state:
            st.session_state.dfs_sum = dfs_sum
        if "dfs_sub" not in st.session_state:
            st.session_state.dfs_sub = dfs_sub


    
        # 3. Виведення результатів
        st.success(f"✅ Завантажено {len(dfs)} файлів")
        
        for filename, df in st.session_state.dfs.items():
            st.subheader(f"📄 Файл: {filename}")
            st.write(df.head())
    
            # Додатково: інформація

            st.text(f"Форма: {df.shape[0]} рядків × {df.shape[1]} колонок")
            st.write(df.describe())
            
            # Когерентне підсумовування
            
            df_sum_tmp = ssp.coherent_summation(df, fs=fs)
           
            # df_sum_tmp = ssp.duplicate_columns(df_sum_tmp)
            st.session_state.dfs_sum[filename+"_sum"] = df_sum_tmp

            df_vpf_tmp = ssp.vpf_df(df, fs)                        
            st.session_state.dfs_vpf[filename+"_vpf"] = df_vpf_tmp
        
        for filename, df in st.session_state.dfs_sum.items():   
            
            
            df_vpf_tmp = ssp.vpf_df(df, fs)                        
            st.session_state.dfs_vpf[filename+"_vpf"] = df_vpf_tmp
        
        for filename, df in st.session_state.dfs_sub.items():   
            
            
            df_vpf_tmp = ssp.vpf_df(df, fs)                        
            st.session_state.dfs_vpf[filename+"_vpf"] = df_vpf_tmp
            
        for filename, df in st.session_state.dfs_sub.items():   
                        
            df_sum_tmp = ssp.coherent_summation(df, fs=fs)
           
            # df_sum_tmp = ssp.duplicate_columns(df_sum_tmp)
            st.session_state.dfs_sum[filename+"_sum"] = df_sum_tmp
    


# === Вкладка 2: Графіки ===
with tab2:
    
    
    st.subheader("Графіки серій у домені амплітуда-час")
    n_cols_plots = int(st.number_input("Кількість колонок для відображення", min_value=0.0, value=3.0, step=1.0, key='n_col'))
    selected_plots = st.multiselect("Оберіть геофони для відображення зі списку:", st.session_state.geoph_list, default=st.session_state.def_geoph_list, key='sel_geoph')
    one_plot_plots = st.checkbox("Показати всі геофони на одному графіку", value=True, key='one_plot')
    
    
    if len(st.session_state.dfs)>0:
        for filename, data in st.session_state.dfs.items():
            st.subheader(filename)
            # st.write(filename)
            # st.caption(filename)
            
            if all(elem in list(data.columns) for elem in selected_plots):
            
                if one_plot_plots:
                    st.plotly_chart(ssp.plot_time_signals(data, fs, n_cols=n_cols_plots, threshold=0.5, columns=selected_plots, mode="plotly_one"), use_container_width=True, key='plot_one'+filename)
                else:
                    st.plotly_chart(ssp.plot_time_signals(data, fs, n_cols=n_cols_plots, threshold=0.5, columns=selected_plots, mode="plotly"), use_container_width=True, key='plot_many'+filename)
    
# === Вкладка 3: Графіки між серіями ===
with tab3:
    
    
    st.subheader("Графіки між серями у домені амплітуда-час")

    sel_ser_list = []
    sel_geoph_list = []

    min_time_list = []
    max_time_list = []

    df_plot = pd.DataFrame()
    df_plot_list = []


    if len(st.session_state.dfs)>0:

        n_ser = st.number_input("Кількість всіх серій для візуалізації", min_value=0, value=None, step=1, key='plots_ser_numb')

        if n_ser:
        
            with st.form("plots_window_form", clear_on_submit=False): 
                
                for i in range(1, n_ser+1):
                    sel_ser_plots = st.selectbox(f"Оберіть серію №{i} для відображення зі списку:", 
                                                      list(st.session_state.dfs.keys())+list(st.session_state.dfs_vpf.keys())+list(st.session_state.dfs_sum.keys())+list(st.session_state.dfs_sub.keys()),
                                                      key="sel_mul_ser_plots"+str(i))
                    sel_geoph_plots = st.multiselect("Оберіть геофони для відображення зі списку:", 
                                                        st.session_state.geoph_list+st.session_state.im_geoph_list+['X', 'Z', 'Y1', 'Y2'],
                                                        default=st.session_state.def_geoph_list,
                                                        key="sel_mul_geo_plots"+str(i))

                    sel_ser_list.append(sel_ser_plots)
                    sel_geoph_list.append(sel_geoph_plots)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        plots_min_time_s = st.number_input("🔻 Початок сигналу", format="%.4f", min_value=0.0, value=0.0, step=0.1, key='plots_min_time_sig'+str(i))
                    with col2:
                        plots_max_time_s = st.number_input("🔺 Кінець сигналу", format="%.4f", min_value=0.0, value=1.0, step=0.1, key='plots_max_time_sig'+str(i))
                   
                    min_time_list.append(plots_min_time_s)
                    max_time_list.append(plots_max_time_s)


                st.subheader("Вирівнювання сигналів")
                    
                sel_ser_align1 = st.selectbox("Оберіть базову серію до якої відбувається вирівнювання зі списку:", 
                                                  list(st.session_state.dfs.keys())+list(st.session_state.dfs_vpf.keys()),
                                                  key="sel_ser_align1")
                sel_geoph_align1 = st.selectbox("Оберіть геофони для вирівнювання зі списку:", 
                                                    st.session_state.geoph_list+st.session_state.im_geoph_list+['X', 'Z', 'Y1', 'Y2'],
                                                    key="sel_geo_align1")
                
                sel_ser_align2 = st.selectbox("Оберіть серію, яка буде вирівняна зі списку:", 
                                                  list(st.session_state.dfs.keys())+list(st.session_state.dfs_vpf.keys()),
                                                  key="sel_mul_ser_align2")
                sel_geoph_align2 = st.selectbox("Оберіть геофони для вирівнювання зі списку:", 
                                                    st.session_state.geoph_list+st.session_state.im_geoph_list+['X', 'Z', 'Y1', 'Y2'],
                                                    key="sel_mul_geo_align2")
                
                
                
                
                
                submitted = st.form_submit_button("⚙️ Побудувати графіки")
                
            if submitted:
                    
                for i, df_name in enumerate(sel_ser_list):
                    if 'vpf' in df_name: 
                        df = st.session_state.dfs_vpf[df_name][sel_geoph_list[i]]
                    elif 'sum' in df_name:
                        df = st.session_state.dfs_sum[df_name][sel_geoph_list[i]]
                    elif 'sub' in df_name:
                        df = st.session_state.dfs_sub[df_name][sel_geoph_list[i]]
                    else:
                        df = st.session_state.dfs[df_name][sel_geoph_list[i]]
                        
                    df = df.add_suffix("_"+df_name)
                    df = ssp.cut_dataframe_time_window(df, fs=fs, start_time=min_time_list[i], end_time=max_time_list[i])

                    df_plot_list.append(df)
                df_plot_list = ssp.align_dataframe_lengths(df_plot_list)
                df_plot = pd.concat(df_plot_list, axis=1)
                
                print(df_plot)
                print(min_time_list)
                print(max_time_list)
                
                st.plotly_chart(ssp.plot_time_signals(df_plot, 
                                                      fs, 
                                                      n_cols=1, 
                                                      threshold=0.5, columns=list(df_plot.columns), 
                                                      mode="plotly_one"), 
                                use_container_width=True, key='plots_multy')
                
                st.subheader("Затримка між сигналами")
                
                delay_matrix = ssp.compute_delay_matrix(df_plot, fs, method='gcc_phat')
                st.dataframe(delay_matrix)
                
                # st.write(delay_matrix['X2_25.07.20-2.csv']['X3_25.07.20-12.csv'])
                
                if 'vpf' in sel_ser_align1:
                    df1 = st.session_state.dfs_vpf[sel_ser_align1][sel_geoph_align1]
                elif 'sum' in sel_ser_align1:
                    df1 = st.session_state.dfs_sum[sel_ser_align1][sel_geoph_align1]
                elif 'sub' in sel_ser_align1:
                    df1 = st.session_state.dfs_sub[sel_ser_align1][sel_geoph_align1]
                else:
                    df1 = st.session_state.dfs[sel_ser_align1][sel_geoph_align1]

                if 'vpf' in sel_ser_align2:
                    df2 = st.session_state.dfs_vpf[sel_ser_align2][sel_geoph_align2]
                elif 'sum' in sel_ser_align2:
                    df2 = st.session_state.dfs_sum[sel_ser_align2][sel_geoph_align2]
                elif 'sub' in sel_ser_align2:
                    df2 = st.session_state.dfs_sub[sel_ser_align2][sel_geoph_align2]
                else:
                    df2 = st.session_state.dfs[sel_ser_align2][sel_geoph_align2]

 
                    
                df_align = ssp.align_two_signals(np.array(df1), np.array(df2), fs=fs, method='gcc_phat', max_lag_s=None)
                
                
                    
                if len(df_align)>0:
                    st.plotly_chart(ssp.plot_time_signals(df_align, 
                                                          fs, 
                                                          n_cols=1, 
                                                          threshold=0.5, columns=list(df_align.columns), 
                                                          mode="plotly_one"), 
                                    use_container_width=True, key='plots_align')



# === Вкладка 4: Спектр ===
with tab4:
    st.subheader("Спектрограми. Представлення у домені частота-час")
    
    with st.form("spectr_window_form", clear_on_submit=False):
        seg_len_s = st.number_input("Довжина одного сегмента спектрограми, с", min_value=0.0, value=None, step=0.1, format="%.4f", key='nperseg')
        overlap_s = st.number_input("Величина перекриття між сегментами, с", min_value=0.0, value=None, step=0.01, format="%.4f", key='noverlap')
        submitted = st.form_submit_button("⚙️ Застосувати параметри")
    
    
    
    if submitted and len(st.session_state.dfs)>0:
        for filename, data in st.session_state.dfs.items():
            st.subheader(filename)
            
            if all(elem in list(data.columns) for elem in selected_plots):
                st.pyplot(ssp.spectr_plot(data, fs, n_cols=n_cols_plots, columns=selected_plots, seg_len_s=seg_len_s, overlap_s=overlap_s), use_container_width=True)
    
# === Вкладка 5: PSD ===
with tab5:
    st.subheader("Спектральна щільність потужності (PSD)")
    db_scale = st.checkbox("Показати в шкалі децибел, дБ", value=True, key='db_scale1')
    
    if len(st.session_state.dfs)>0:
        for filename, data in st.session_state.dfs.items():
            st.subheader(filename)
            if all(elem in list(data.columns) for elem in selected_plots):
                if db_scale:
                    st.plotly_chart(ssp.psd_plot_df(data, fs=fs, n_cols=n_cols_plots, columns=selected_plots, mode='plotly',scale='db'), use_container_width=True, key='plot_psd1'+filename)
                else:
                    st.plotly_chart(ssp.psd_plot_df(data, fs=fs, n_cols=n_cols_plots, columns=selected_plots, mode='plotly',scale='energy'), use_container_width=True, key='plot_psd2'+filename)

# === Вкладка 6: RMS за PSD ===
with tab6:
    st.subheader("Графік PSD")
    
    if len(st.session_state.dfs)>0:
        
        selected_ser_psd = st.selectbox("Оберіть серію для відображення зі списку:", list(st.session_state.dfs.keys()),key="sel_rms_psd1")
        selected_geoph_psd = st.selectbox("Оберіть геофони для відображення зі списку:", st.session_state.geoph_list+st.session_state.im_geoph_list,key="sel_rms_psd2")
    
        db_scale_psd = st.checkbox("Показати в шкалі децибел, дБ", value=False, key='db_scale_rms_psd')
        if db_scale_psd:
            st.plotly_chart(ssp.psd_plot_df(st.session_state.dfs[selected_ser_psd], fs=fs, n_cols=1, columns=[selected_geoph_psd], mode='plotly', scale='db'), use_container_width=True,key="plot_rms_psd1")
            f, Pxx = ssp.psd_plot_df(st.session_state.dfs[selected_ser_psd], fs=fs, n_cols=1, columns=[selected_geoph_psd], mode='matrix', scale='db') 
        else:
            st.plotly_chart(ssp.psd_plot_df(st.session_state.dfs[selected_ser_psd], fs=fs, n_cols=1, columns=[selected_geoph_psd], mode='plotly', scale='energy'), use_container_width=True,key="plot_rms_psd2")
            f, Pxx = ssp.psd_plot_df(st.session_state.dfs[selected_ser_psd], fs=fs, n_cols=1, columns=[selected_geoph_psd], mode='matrix', scale='energy') 
    
        
        st.subheader("🎚️ Середнє квадратичне значення PSD в діапазоні")
    
        with st.form("rms_psd_window_form", clear_on_submit=False):

 





            # Поля для введення мінімальної та максимальної частоти
            col1, col2 = st.columns(2)
            with col1:
                min_freq_rms_psd = st.number_input("🔻 Мінімальна частота", min_value=0.0, value=20.0, step=1.0, key='min_freq_rms_psd')
                
            with col2:
                max_freq_rms_psd = st.number_input("🔺 Максимальна частота", min_value=0.0, value=50.0, step=1.0, key='max_freq_rms_psd')
            submitted = st.form_submit_button("⚙️ Розрахувати")
            
        if submitted:
            print(f[0])
            print(Pxx[0])
            rms_psd, range_freq_val = ssp.rms_in_band(f[0], Pxx[0], min_freq_rms_psd, max_freq_rms_psd)
            st.subheader(f"🎚️ RMS PSD дорівнює {rms_psd} в діапазоні від {range_freq_val[0]} Гц до {range_freq_val[1]} Гц")
    

# === Вкладка 7: Визначення затримок ===
with tab7:
    st.subheader("Затримки в сигналах геофонів")
    method_delay = st.selectbox("Оберіть метод обчислення затримок:", ['gcc_phat', 'envelope', 'cross_correlation', 'rms_envelope'],key="sel_method_delay")
    
    if len(st.session_state.dfs)>0:
        for filename, data in chain(st.session_state.dfs.items(), st.session_state.dfs_vpf.items()):
            st.subheader(filename)
            delay_matrix = ssp.compute_delay_matrix(data, fs, method=method_delay)
            st.dataframe(delay_matrix)


            
# === Вкладка 8: Когерентність ===
with tab8:
    st.subheader("Когерентність між двома сигналами")

    if len(st.session_state.dfs)>0:

        
        with st.form("coher_window_form", clear_on_submit=False): 
                        
            st.write("1й сигнал")
            selected_ser1_coher = st.selectbox("Оберіть серію для відображення зі списку:", list(st.session_state.dfs.keys())+list(st.session_state.dfs_vpf.keys()),key="coh_sel_ser1")
            selected_seism1_coher = st.selectbox("Оберіть типи геофонів для відображення зі списку:", st.session_state.geoph_list+st.session_state.im_geoph_list,key="coh_sel_seism1")

            col1, col2 = st.columns(2)
            with col1:
                coher_min_time_s1 = st.number_input("🔻 Початок сигналу", format="%.4f", min_value=0.0, value=0.0, step=0.1, key='coher_min_time_sig1')
            with col2:
                coher_max_time_s1 = st.number_input("🔺 Кінець сигналу", format="%.4f", min_value=0.0, value=1.0, step=0.1, key='coher_max_time_sig1')
           
            st.write("2й сигнал")
            selected_ser2_coher = st.selectbox("Оберіть серію для відображення зі списку:", list(st.session_state.dfs.keys())+list(st.session_state.dfs_vpf.keys()),key="coh_sel_ser2")
            selected_seism2_coher = st.selectbox("Оберіть типи геофонів для відображення зі списку:", st.session_state.geoph_list+st.session_state.im_geoph_list,key="coh_sel_seism2")

            col1, col2 = st.columns(2)
            with col1:
                coher_min_time_s2 = st.number_input("🔻 Початок сигналу", format="%.4f", min_value=0.0, value=0.0, step=0.1, key='coher_min_time_sig2')
            with col2:
                coher_max_time_s2 = st.number_input("🔺 Кінець сигналу", format="%.4f", min_value=0.0, value=1.0, step=0.1, key='coher_max_time_sig2')

                    
            type_plot = st.selectbox("Оберіть тип відображення зі списку:", ["linear", "log"], key="coh_type_plot")
            
            coher_seg_len_s = st.number_input("Довжина одного сегмента, с", min_value=0.0, value=0.02, step=0.1, format="%.4f", key='coher_nperseg')
            coher_overlap_s = st.number_input("Величина перекриття між сегментами, с", min_value=0.0, value=0.018, step=0.01, format="%.4f", key='coher_noverlap')

            st.write("Діапазон частот для обчислення метрики")

            col1, col2 = st.columns(2)
            with col1:
                min_freq_coher = st.number_input("🔻 Мінімальна частота", min_value=0.0, value=20.0, step=1.0,  key='min_freq_coher')
                
            with col2:
                max_freq_coher = st.number_input("🔺 Максимальна частота", min_value=0.0, value=50.0, step=1.0, key='max_freq_coher')
                



            submitted = st.form_submit_button("⚙️ Побудувати графіки")
    
            
        if submitted:
                
            coher_nperseg = int(coher_seg_len_s * fs)
            coher_noverlap = int(coher_overlap_s * fs)
            
            
            if 'vpf' in selected_ser1_coher:
                df1 = st.session_state.dfs_vpf[selected_ser1_coher][selected_seism1_coher]
            else:
                df1 = st.session_state.dfs[selected_ser1_coher][selected_seism1_coher]
                
            if 'vpf' in selected_ser2_coher:
                df2 = st.session_state.dfs_vpf[selected_ser2_coher][selected_seism2_coher]
            else:
                df2 = st.session_state.dfs[selected_ser2_coher][selected_seism2_coher]


            # df1 = st.session_state.dfs[selected_ser1_coher][selected_seism1_coher]
            # df2 = st.session_state.dfs[selected_ser2_coher][selected_seism2_coher]
            
            df2 = df2[:len(df1)]
                    
                
            df1 = ssp.cut_dataframe_time_window(df1, fs=fs, start_time=coher_min_time_s1, end_time=coher_max_time_s1)
            df2 = ssp.cut_dataframe_time_window(df2, fs=fs, start_time=coher_min_time_s2, end_time=coher_max_time_s2)
            
            sig1 = np.array(df1)
            sig2 = np.array(df2)
            
                
            st.plotly_chart(ssp.plot_coherence(sig1, 
                                               sig2, 
                                               fs, f"{selected_ser1_coher}, {selected_seism1_coher}", f"{selected_ser2_coher}, {selected_seism2_coher}", 
                                               mode='plotly', type_plot=type_plot,
                                               nperseg=coher_nperseg, noverlap=coher_noverlap,
                                               ), use_container_width=True, key='plot_coher')


            f, Cxy = ssp.plot_coherence(sig1, sig2, fs, f"{selected_ser1_coher}, {selected_seism1_coher}", f"{selected_ser2_coher}, {selected_seism2_coher}", 
                                               mode='matrix', type_plot="linear",
                                               nperseg=coher_nperseg, noverlap=coher_noverlap,
                                               )


            print(f)
            print(Cxy)
            rms_coher, range_freq_val_coher = ssp.rms_in_band(f, Cxy, min_freq_coher, max_freq_coher)
            st.subheader(f"🎚️ RMS когерентності дорівнює {rms_coher} в діапазоні від {range_freq_val_coher[0]} Гц до {range_freq_val_coher[1]} Гц")
            




# === Вкладка 9: Когерентне підсумовування сигналів сейсмометрів ===
with tab9:
    st.subheader("Когерентне підсумовування сигналів сейсмометрів")

    if len(st.session_state.dfs)>0:

        with st.form("coh_sum_window_form", clear_on_submit=False):
            # Поля для введення мінімальної та максимальної частоти
            st.write("Оберіть серію для підсумовування відповідних сигналів сейсмометрів")
            selected_ser_sum = st.selectbox("Оберіть серію для відображення зі списку:", list(st.session_state.dfs.keys()),key="coh_sum_sel_ser1")
            submitted = st.form_submit_button("⚙️ Розрахувати")
            
        if submitted:
            st.subheader("Обрана серія: "+selected_ser_sum)
            df_sum = ssp.coherent_summation(st.session_state.dfs[selected_ser_sum], fs=fs)
            print("coherent sum")
            print(df_sum)
            # breakpoint()
            # df_sum = ssp.duplicate_columns(df_sum)
            st.plotly_chart(ssp.plot_time_signals(df_sum, fs, n_cols=n_cols_plots, threshold=0.5, columns=list(df_sum.columns), mode="plotly_one"), use_container_width=True, key='plot_coher_sum')    
            st.session_state.dfs_sum[selected_ser_sum+"_sum"] = df_sum
        




        

# === Вкладка 10: Когерентне віднімання ===
with tab10:
    
    
    st.subheader("Когерентне віднімання шуму")
    # st.write(st.session_state.dfs.keys())
    
    subs_mode = st.radio(
    "Оберіть тип віднімання:",
    ["Одна серія", "Дві серії"],
    index=0  # за замовчуванням вибраний елемент
    )
    
    data_geoph = {}
    
    if "plot_flag" not in st.session_state:
        st.session_state.plot_flag = False

    if "noisy_sig_plot" not in st.session_state:
        st.session_state.noisy_sig_plot = []
        
    if "ref_noise_plot" not in st.session_state:
        st.session_state.ref_noise_plot = []
    
    if "res_signal" not in st.session_state:
        st.session_state.res_signal = []
        
        
    if "selected_ser1" not in st.session_state:
        st.session_state.selected_ser1 = 0
    

    if subs_mode == "Одна серія":
        
        st.write("Cигнал")
        selected_ser_sub = st.selectbox("Оберіть серію для відображення зі списку:", list(st.session_state.dfs.keys()),key="sel_sub_one")
        st.write("Геофони для аналізу")
        selected_geoph_sub = st.multiselect("Оберіть геофони для відображення зі списку:", st.session_state.geoph_list,key="sel_subs")
        
        noisy_sig_len_s = 0.0
        if len(st.session_state.dfs)>0 and selected_ser_sub and selected_geoph_sub:
            noisy_sig_df = st.session_state.dfs[selected_ser_sub][selected_geoph_sub]
            
            noisy_sig_len_s = len(noisy_sig_df)/fs
        


        
        st.subheader("🎚️ Часові вікна сигналу та шуму")
        
        with st.form("subs_window_form", clear_on_submit=False):
            
                      
 

            # print(selected_ser1)
            # breakpoint()


            if selected_ser_sub:
                st.subheader("Обрано серію "+selected_ser_sub)
            
            

            col1, col2 = st.columns(2)
            with col1:
                subs_min_time_s = st.number_input("🔻 Початок сигналу", min_value=0.0, value=2.53, step=0.1, format="%.4f" , key='subs_min_time_sig')
            with col2:
                subs_max_time_s = st.number_input("🔺 Кінець сигналу", min_value=0.0, value=noisy_sig_len_s, step=0.1, format="%.4f", key='subs_max_time_sig')
            col1, col2 = st.columns(2)
            with col1:
                subs_min_time_n = st.number_input("🔻 Початок шуму", min_value=0.0, value=noisy_sig_len_s, step=0.1, format="%.4f", key='subs_min_time_noise')
            with col2:
                subs_max_time_n = st.number_input("🔺 Кінець шуму", min_value=0.0, value=noisy_sig_len_s, step=0.1, format="%.4f", key='subs_max_time_noise')


            
            sig_len = subs_max_time_s-subs_min_time_s
            noise_len = subs_max_time_n-subs_min_time_n
            st.write(f"Довжина сигналу {sig_len:.4f} c")
            st.write(f"Довжина шуму {noise_len:.4f} c")
            
            
            # seg_len_s = st.number_input("Довжина одного сегмента спектрограми, с", min_value=0.0, value=0.02, step=0.1, format="%.4f", key='subs_nperseg')
            # overlap_s = st.number_input("Величина перекриття між сегментами, с", min_value=0.0, value=0.018, step=0.01, format="%.4f", key='subs_noverlap')


            st.write("Поріг чутливості маски когерентності") 
            st.write("Менше значення (~0.0) → маска чутливіша, більше частот вважаються когерентними → агресивніше приглушення.")
            st.write("Більше значення (~0.2–0.3) → менше частот перевищують поріг → м’якіше приглушення.")
            st.write("Значення >0.3 → дуже обережне приглушення")

            coherence_threshold = st.number_input("Поріг чутливості", min_value=0.0, value=0.9, step=0.1, key='subs_coher_thersh')
            
            st.write("Інтенсивність приглушення контролює, наскільки сильно заглушується спектр у точках, де когерентність висока.")
            st.write("1.0 (максимум) — повне приглушення когерентних компонент.")
            st.write("0.0 — взагалі не приглушується, лише оцінюється.")
            st.write("0.3–0.7 — м’яке згасання когерентних ділянок без агресивного вирізання.")
            
            suppression_strength = st.number_input("Інтенсивність приглушення", min_value=0.0, value=1.0, step=0.1, key='subs_suppr_strength')
            
            st.write("win_len. Довжина ковзного вікна (непарне число відліків)")
            st.write("Обирається за тривалістю події (імпульсу).")
            st.write("Має бути приблизно 1–2× довжини події, яку хочемо виявити/приглушити.")
            st.write("Типово: 101–301 для сигналів довжиною кілька тисяч відліків.")
            st.write("Занадто мале → шумні оцінки, стрибки;")
            st.write("занадто велике → втрата локалізації (ціль «змазується»).")
            
            win_len = int(st.number_input("Довжина ковзного вікна (непарне число відліків)", min_value=0, value=201, step=1, key='subs_win_len'))
            
            st.write("smooth_alpha. Згладжування коефіцієнта масштабу α")
            st.write("Використовується для стабілізації сили віднімання.")
            st.write("Має бути приблизно 1–2× довжини події, яку хочемо виявити/приглушити.")
            st.write("Типові значення: 0.1–0.3 × win_len.")
            st.write("Занадто мале → α «стрибає», виникає недо/перевіднімання;")
            st.write("занадто велике → віднімання розмазується на сусідні точки")
            
            smooth_alpha = int(st.number_input("згладжування коефіцієнта масштабу α (число відліків)", min_value=0, value=int(0.15*win_len), step=1, key='subs_smooth_alpha'))
            
            st.write("smooth_gamma. Згладжування показника когерентності γ²")
            st.write("Має бути меншим за smooth_alpha, щоб не втратити короткі події.")
            st.write("Типові значення: 0.05–0.2 × win_len.")
            st.write("Занадто мале → маска «мерехтить» (дірки в приглушенні);")
            st.write("занадто велике → короткі цілі теж придушуються.")
            st.write("занадто велике → віднімання розмазується на сусідні точки")
            
            smooth_gamma = int(st.number_input("Згладжування показника когерентності γ² (число відліків)", min_value=0, value=int(0.2*win_len), step=1, key='subs_smooth_gamma'))
            
            
            # method_delay_subs = st.selectbox("Оберіть метод обчислення затримок:", ['gcc_phat', 'envelope', 'cross_correlation', 'rms_envelope'],key="sel_method_delay_subs")
          
            plot_figs_s = st.checkbox("Побудувати графік амплітуда-час початкового сигналу", value=False, key='plot_figs_s')
            plot_figs_n = st.checkbox("Побудувати графік амплітуда-час шуму", value=False, key='plot_figs_n')
            plot_figs_r = st.checkbox("Побудувати графік амплітуда-час результату віднімання", value=False, key='plot_figs_r')
            
            plot_spectr_s = st.checkbox("Побудувати спектрограму початкового сигналу", value=False, key='plot_spectr_s')
            plot_spectr_n = st.checkbox("Побудувати спектрограму шуму", value=False, key='plot_spectr_n')
            plot_spectr_r = st.checkbox("Побудувати спектрограму результату віднімання", value=False, key='plot_spectr_r')
            
            plot_psd_s = st.checkbox("Побудувати PSD початкового сигналу", value=False, key='plot_psd_s')
            plot_psd_n = st.checkbox("Побудувати PSD шуму", value=False, key='plot_psd_n')
            plot_psd_r = st.checkbox("Побудувати PSD результату віднімання", value=False, key='plot_psd_r')
            
            plot_vpf_s = st.checkbox("Побудувати VPF початкового сигналу", value=False, key='plot_vpf_s')
            plot_vpf_n = st.checkbox("Побудувати VPF шуму", value=False, key='plot_vpf_n')
            plot_vpf_r = st.checkbox("Побудувати VPF результату віднімання", value=False, key='plot_vpf_r')
            
            # print(selected_ser1)
            # breakpoint()
            
            submitted = st.form_submit_button("⚙️ Застосувати вікно")
            # Кнопка для запуску фільтрації
            
            # print(selected_ser1)
            # print(st.session_state.selected_ser1)
            # breakpoint()            
        if submitted:
            
            st.session_state.selected_ser1 = selected_ser_sub
            st.write("Обрано серію "+selected_ser_sub)
            
            # print(selected_ser1)
            # print(st.session_state.selected_ser1)
            # breakpoint()       

            noisy_sig_df_cut = ssp.cut_dataframe_time_window(noisy_sig_df, fs=fs, start_time=subs_min_time_s, end_time=subs_max_time_s)
            ref_noise_df_cut = ssp.cut_dataframe_time_window(noisy_sig_df, fs=fs, start_time=subs_min_time_n, end_time=subs_max_time_n)
            ref_noise_df_cut = ref_noise_df_cut[0:len(noisy_sig_df_cut)]
            
        
            delay_list = []
            for geoph in selected_geoph_sub: #['X1','Y11','Y12','Z1','X2','Y21','Y22','Z2','X3','Y31','Y32','Z3']: 
                # signal, delay, _, _, _, _, _ = ssp.coherent_subtraction_adaptive(noisy_sig_df_cut[geoph], 
                #                                                              ref_noise_df_cut[geoph], 
                #                                                              seg_len_s=seg_len_s, 
                #                                                              overlap_s=overlap_s,
                #                                                              coherence_bias=coherence_threshold, 
                #                                                              suppression_strength=suppression_strength, 
                #                                                              delay_method=method_delay_subs, 
                #                                                              max_lag_s=None,
                #                                                              freq_limit=None)
                
                
                signal, delay, _, _  =  ssp.coherent_subtraction_adaptive_1d(noisy_sig_df_cut[geoph],
                                                                         ref_noise_df_cut[geoph],
                                                                         fs=fs,
                                                                         win_len=win_len,
                                                                         corr_threshold=coherence_threshold,
                                                                         suppression_strength=suppression_strength,
                                                                         smooth_alpha=smooth_alpha,
                                                                         smooth_gamma=smooth_gamma
                                                                         )
                
                
                
                signal = signal[:len(noisy_sig_df_cut)]
                data_geoph[geoph] = signal
                
                delay_list.append(delay)
                
            st.session_state.plot_flag = True
            
            cleaned_sig_df = pd.DataFrame(data_geoph)
            st.session_state.dfs_sub[st.session_state.selected_ser1+"_sub_one"] = cleaned_sig_df
            
  
            st.session_state.noisy_sig_plot = noisy_sig_df_cut
            st.session_state.ref_noise_plot = ref_noise_df_cut
            st.session_state.res_signal = cleaned_sig_df
            
            selected_ser1_sub = None
            
            delays = pd.DataFrame(delay_list, index=selected_geoph_sub)
            
            st.dataframe(delays)

                
    
    if subs_mode == "Дві серії":

        st.write("1й сигнал")
        selected_ser1_sub = st.selectbox("Оберіть серію для відображення зі списку:", list(st.session_state.dfs.keys()),key="sel_sub1")
        # selected_seism1 = st.selectbox("Оберіть типи геофонів для відображення зі списку:", ['X1','Y11','Y12','Z1','X2','Y21','Y22','Z2','X3','Y31','Y32','Z3'],key="select12")
        st.write("2й сигнал (шум)")
        selected_ser2_sub = st.selectbox("Оберіть серію для відображення зі списку:", list(st.session_state.dfs.keys()),key="sel_sub2")        
        st.write("Геофони для аналізу")
        selected_geoph_sub = st.multiselect("Оберіть геофони для відображення зі списку:", st.session_state.geoph_list+st.session_state.im_geoph_list,key="sel_subs")
        
        noisy_sig_len_s = 0.0
        ref_noise_len_s = 0.0 
        
        if len(st.session_state.dfs)>0:

            noisy_sig_df = st.session_state.dfs[selected_ser1_sub][selected_geoph_sub]
            ref_noise_df = st.session_state.dfs[selected_ser2_sub][selected_geoph_sub]
            
            noisy_sig_len_s = len(noisy_sig_df)/fs
            ref_noise_len_s = len(ref_noise_df)/fs

        with st.form("subs_window_form2", clear_on_submit=False):    
            
            
            if selected_ser1_sub:
                st.subheader("Обрано серію сигналу "+selected_ser1_sub)
            if selected_ser2_sub:
                st.subheader("Обрано серію шуму "+selected_ser2_sub)
            
            if len(st.session_state.dfs)>0:

                
                col1, col2 = st.columns(2)
                with col1:
                    subs_min_time_s = st.number_input("🔻 Початок сигналу", min_value=0.0, value=0.0, step=0.1, format="%.4f", key='subs_min_time_sig2')
                with col2:
                    subs_max_time_s = st.number_input("🔺 Кінець сигналу", min_value=0.0, value=noisy_sig_len_s, step=0.1, format="%.4f", key='subs_max_time_sig2')
                col1, col2 = st.columns(2)
                with col1:
                    subs_min_time_n = st.number_input("🔻 Початок шуму", min_value=0.0, value=0.0, step=0.1, format="%.4f", key='subs_min_time_noise2')
                with col2:
                    subs_max_time_n = st.number_input("🔺 Кінець шуму", min_value=0.0, value=ref_noise_len_s, step=0.1, format="%.4f", key='subs_max_time_noise2')
                
                sig_len = subs_max_time_s-subs_min_time_s
                noise_len = subs_max_time_n-subs_min_time_n
                sig_len_samples = sig_len * fs 
                noise_len_samples = noise_len * fs
                
                st.write(f"Частота дискретизації {fs:.4f} Гц")
                
                st.write(f"Довжина сигналу {sig_len:.4f} c")
                st.write(f"Довжина шуму {noise_len:.4f} c")
                
                st.write(f"Довжина сигналу у відліках (samples). {sig_len_samples:.0f}")
                st.write(f"Довжина шуму у відліках (samples). {noise_len_samples:.0f}")
                
                
                # seg_len_s = st.number_input("Довжина одного сегмента спектрограми, с", min_value=0.0, value=0.02, step=0.1, format="%.4f", key='subs_nperseg')
                # overlap_s = st.number_input("Величина перекриття між сегментами, с", min_value=0.0, value=0.018, step=0.01, format="%.4f", key='subs_noverlap')
    
    
                st.subheader("Поріг чутливості маски когерентності") 
                st.write("Менше значення (~0.0) → маска чутливіша, більше частот вважаються когерентними → агресивніше приглушення.")
                st.write("Більше значення (~0.2–0.3) → менше частот перевищують поріг → м’якіше приглушення.")
                st.write("Значення >0.3 → дуже обережне приглушення")
    
                coherence_threshold = st.number_input("Поріг чутливості", min_value=0.0, value=0.9, step=0.1, key='subs_coher_thersh2')
                
                st.subheader("Інтенсивність приглушення ")
                st.write("Контролює наскільки сильно заглушується спектр у точках, де когерентність висока.")
                st.write("1.0 (максимум) — повне приглушення когерентних компонент.")
                st.write("0.0 — взагалі не приглушується, лише оцінюється.")
                st.write("0.3–0.7 — м’яке згасання когерентних ділянок без агресивного вирізання.")
                
                suppression_strength = st.number_input("Інтенсивність приглушення", min_value=0.0, value=1.0, step=0.1, key='subs_suppr_strength2')
                
                st.subheader("win_len. Довжина ковзного вікна (непарне число відліків)")
                st.write("Обирається за тривалістю події (імпульсу).")
                st.write("Має бути приблизно 1–2× довжини події, яку хочемо виявити/приглушити.")
                st.write("Типово: 101–301 для сигналів довжиною кілька тисяч відліків.")
                st.write("Занадто мале → шумні оцінки, стрибки;")
                st.write("занадто велике → втрата локалізації (ціль «змазується»).")
                
                win_len = int(st.number_input("Довжина ковзного вікна (непарне число відліків)", min_value=0, value=201, step=1, key='subs_win_len2'))
                
                st.subheader("smooth_alpha. Згладжування коефіцієнта масштабу α")
                st.write("Використовується для стабілізації сили віднімання.")
                st.write("Має бути приблизно 1–2× довжини події, яку хочемо виявити/приглушити.")
                st.write("Типові значення: 0.1–0.3 × win_len.")
                st.write("Занадто мале → α «стрибає», виникає недо/перевіднімання;")
                st.write("занадто велике → віднімання розмазується на сусідні точки")
                
                smooth_alpha = int(st.number_input("згладжування коефіцієнта масштабу α (число відліків)", min_value=0, value=int(0.15*win_len), step=1, key='subs_smooth_alpha2'))
                
                st.subheader("smooth_gamma. Згладжування показника когерентності γ²")
                st.write("Має бути меншим за smooth_alpha, щоб не втратити короткі події.")
                st.write("Типові значення: 0.05–0.2 × win_len.")
                st.write("Занадто мале → маска «мерехтить» (дірки в приглушенні);")
                st.write("занадто велике → короткі цілі теж придушуються.")
                st.write("занадто велике → віднімання розмазується на сусідні точки")
                
                smooth_gamma = int(st.number_input("Згладжування показника когерентності γ² (число відліків)", min_value=0, value=int(0.2*win_len), step=1, key='subs_smooth_gamma2'))


                plot_figs_s = st.checkbox("Побудувати графік амплітуда-час початкового сигналу", value=False, key='plot_figs_s2')
                plot_figs_n = st.checkbox("Побудувати графік амплітуда-час шуму", value=False, key='plot_figs_n2')
                plot_figs_r = st.checkbox("Побудувати графік амплітуда-час результату віднімання", value=False, key='plot_figs_r2')
                
                plot_spectr_s = st.checkbox("Побудувати спектрограму початкового сигналу", value=False, key='plot_spectr_s2')
                plot_spectr_n = st.checkbox("Побудувати спектрограму шуму", value=False, key='plot_spectr_n2')
                plot_spectr_r = st.checkbox("Побудувати спектрограму результату віднімання", value=False, key='plot_spectr_r2')
                
                plot_psd_s = st.checkbox("Побудувати PSD початкового сигналу", value=False, key='plot_psd_s2')
                plot_psd_n = st.checkbox("Побудувати PSD шуму", value=False, key='plot_psd_n2')
                plot_psd_r = st.checkbox("Побудувати PSD результату віднімання", value=False, key='plot_psd_r2')
                
                plot_vpf_s = st.checkbox("Побудувати VPF початкового сигналу", value=False, key='plot_vpf_s2')
                plot_vpf_n = st.checkbox("Побудувати VPF шуму", value=False, key='plot_vpf_n2')
                plot_vpf_r = st.checkbox("Побудувати VPF результату віднімання", value=False, key='plot_vpf_r2')

                
                
                submitted = st.form_submit_button("⚙️ Застосувати параметри")
                
        if submitted:
            
            st.session_state.selected_ser1 = selected_ser1_sub
            # st.session_state.selected_ser2 = selected_ser2
            st.write("Обрано серію сигналу "+selected_ser1_sub)
            st.write("Обрано серію шуму "+selected_ser2_sub)
            
            # print(selected_ser1)
            # print(st.session_state.selected_ser1)
            # breakpoint()       
               
            
            noisy_sig_df_cut = ssp.cut_dataframe_time_window(st.session_state.dfs[selected_ser1_sub], fs=fs, start_time=subs_min_time_s, end_time=subs_max_time_s)
            ref_noise_df_cut = ssp.cut_dataframe_time_window(st.session_state.dfs[selected_ser2_sub], fs=fs, start_time=subs_min_time_n, end_time=subs_max_time_n)
            ref_noise_df_cut = ref_noise_df_cut[0:len(noisy_sig_df_cut)]

            delay_list = []
            
            for geoph in selected_geoph_sub: # ['X1','Y11','Y12','Z1','X2','Y21','Y22','Z2','X3','Y31','Y32','Z3']: 
                # signal, delay, _, _, _, _, _ = ssp.coherent_subtraction_aligned_with_mask(noisy_sig_df_cut[geoph], 
                #                                                                       ref_noise_df_cut[geoph], 
                #                                                                       seg_len_s=seg_len_s, 
                #                                                                       overlap_s=overlap_s,
                #                                                                       coherence_threshold=coherence_threshold)
                
                # signal, delay, _, _, _, _, _ = ssp.coherent_subtraction_adaptive(noisy_sig_df_cut[geoph], 
                #                                                                       ref_noise_df_cut[geoph], 
                #                                                                       seg_len_s=seg_len_s, 
                #                                                                       overlap_s=overlap_s,
                                                                                      
                #                                                                       coherence_bias=coherence_threshold, 
                #                                                                       suppression_strength=suppression_strength, 
                #                                                                       delay_method=method_delay_subs, 
                #                                                                       max_lag_s=None,
                #                                                                       freq_limit=None)

                signal, delay, _, _  =  ssp.coherent_subtraction_adaptive_1d(noisy_sig_df_cut[geoph],
                                                                         ref_noise_df_cut[geoph],
                                                                         fs=fs,
                                                                         win_len=win_len,
                                                                         corr_threshold=coherence_threshold,
                                                                         suppression_strength=suppression_strength,
                                                                         smooth_alpha=smooth_alpha,
                                                                         smooth_gamma=smooth_gamma
                                                                         )

                
                signal = signal[:len(st.session_state.dfs[selected_ser1_sub][geoph])]
                data_geoph[geoph] = signal
                
                delay_list.append(delay)
                
           
            st.session_state.plot_flag = True
            
            cleaned_sig_df = pd.DataFrame(data_geoph)
            st.session_state.dfs_sub[selected_ser1_sub+"_sub_two"] = cleaned_sig_df
            
            
            st.session_state.noisy_sig_plot = noisy_sig_df_cut
            st.session_state.ref_noise_plot = ref_noise_df_cut
            st.session_state.res_signal = cleaned_sig_df
            
            selected_ser1_sub = None
            
            delays = pd.DataFrame(delay_list, index=selected_geoph_sub)
            
            st.dataframe(delays)



    
    if st.session_state.plot_flag:
        
        # df = ssp.vpf_df(st.session_state.res_signal, fs)                        
        # if subs_mode == "Дві серії":
        #     st.session_state.dfs_vpf[st.session_state.selected_ser1+"_subtract__vpf"] = df
        # if subs_mode == "Одна серія":
        #     st.session_state.dfs_vpf[st.session_state.selected_ser1+"_subtract_cut__vpf"] = df
            
        signal_vpf = ssp.vpf_df(st.session_state.noisy_sig_plot, fs)
        noise_vpf = ssp.vpf_df(st.session_state.ref_noise_plot, fs)                         
        snr_df = ssp.compute_snr_df(signal_vpf, noise_vpf)
        st.subheader("Відношення SNR")
        st.dataframe(snr_df)

    
        st.subheader("Результат когерентного віднімання")
        if len(df):
            
            st.subheader("Серія сигналу: "+st.session_state.selected_ser1)
            if plot_figs_s or plot_figs_n or plot_figs_r:

                st.subheader("Графік амплітуда-час")
                one_plot_subs = st.checkbox("Показати всі геофони на одному графіку", value=True, key='one_plot_subs')
                
                if plot_figs_s: 
                    st.subheader("Початковий сигнал")
                    if one_plot_subs:
                        st.plotly_chart(ssp.plot_time_signals(st.session_state.noisy_sig_plot, fs, n_cols=n_cols_plots, threshold=0.5, columns=selected_geoph_sub, mode="plotly_one"), use_container_width=True, key='plot_one_subs_s'+filename)
                    else:
                        st.plotly_chart(ssp.plot_time_signals(st.session_state.noisy_sig_plot, fs, n_cols=n_cols_plots, threshold=0.5, columns=selected_geoph_sub, mode="plotly"), use_container_width=True, key='plot_many_subs_s'+filename)

                if plot_figs_n: 
                    st.subheader("Шум")
                    if one_plot_subs:
                        st.plotly_chart(ssp.plot_time_signals(st.session_state.ref_noise_plot, fs, n_cols=n_cols_plots, threshold=0.5, columns=selected_geoph_sub, mode="plotly_one"), use_container_width=True, key='plot_one_subs_n'+filename)
                    else:
                        st.plotly_chart(ssp.plot_time_signals(st.session_state.ref_noise_plot, fs, n_cols=n_cols_plots, threshold=0.5, columns=selected_geoph_sub, mode="plotly"), use_container_width=True, key='plot_many_subs_n'+filename)

                if plot_figs_r: 
                    st.subheader("Результат віднімання")
                    if one_plot_subs:
                        st.plotly_chart(ssp.plot_time_signals(st.session_state.res_signal, fs, n_cols=n_cols_plots, threshold=0.5, columns=selected_geoph_sub, mode="plotly_one"), use_container_width=True, key='plot_one_subs_r'+filename)
                    else:
                        st.plotly_chart(ssp.plot_time_signals(st.session_state.res_signal, fs, n_cols=n_cols_plots, threshold=0.5, columns=selected_geoph_sub, mode="plotly"), use_container_width=True, key='plot_many_subs_r'+filename)


            if plot_spectr_s or plot_spectr_n or plot_spectr_r:

                st.subheader("Спектрограмма")
                
                if plot_spectr_s: 
                    st.subheader("Початковий сигнал")
                   
                    st.pyplot(ssp.spectr_plot(st.session_state.noisy_sig_plot, fs, n_cols=n_cols_plots, columns=selected_geoph_sub), use_container_width=True)
                   

                if plot_spectr_n: 
                    st.subheader("Шум")
                    
                    st.pyplot(ssp.spectr_plot(st.session_state.ref_noise_plot, fs, n_cols=n_cols_plots, columns=selected_geoph_sub), use_container_width=True)
                    
                if plot_spectr_r: 
                    st.subheader("Результат віднімання")
                    st.pyplot(ssp.spectr_plot(st.session_state.res_signal, fs, n_cols=n_cols_plots, columns=selected_geoph_sub), use_container_width=True)
                    

            if plot_psd_s or plot_psd_n or plot_psd_r:

                st.subheader("Графік PSD")
                db_scale_subs = st.checkbox("Показати в шкалі децибел, дБ", value=True, key='db_scale_subs')
                
                if plot_psd_s: 
                    st.subheader("Початковий сигнал")
                    if db_scale_subs:
                        st.plotly_chart(ssp.psd_plot_df(st.session_state.noisy_sig_plot, fs, n_cols=n_cols_plots, columns=selected_geoph_sub, mode='plotly', scale='db'), use_container_width=True, key='plot_sub_psd1_s'+filename)
                    else:
                        st.plotly_chart(ssp.psd_plot_df(st.session_state.noisy_sig_plot, fs, n_cols=n_cols_plots, columns=selected_geoph_sub, mode='plotly', scale='energy'), use_container_width=True, key='plot_sub_psd2_s'+filename)

                if plot_psd_n: 
                    st.subheader("Шум")
                    if db_scale_subs:
                        st.plotly_chart(ssp.psd_plot_df(st.session_state.ref_noise_plot, fs, n_cols=n_cols_plots, columns=selected_geoph_sub, mode='plotly', scale='db'), use_container_width=True, key='plot_sub_psd1_n'+filename)
                    else:
                        st.plotly_chart(ssp.psd_plot_df(st.session_state.ref_noise_plot, fs, n_cols=n_cols_plots, columns=selected_geoph_sub, mode='plotly', scale='energy'), use_container_width=True, key='plot_sub_psd2_n'+filename)

                if plot_psd_r: 
                    st.subheader("Результат віднімання")
                    if db_scale_subs:
                        st.plotly_chart(ssp.psd_plot_df(st.session_state.res_signal, fs, n_cols=n_cols_plots, columns=selected_geoph_sub, mode='plotly', scale='db'), use_container_width=True, key='plot_sub_psd1_r'+filename)
                    else:
                        st.plotly_chart(ssp.psd_plot_df(st.session_state.res_signal, fs, n_cols=n_cols_plots, columns=selected_geoph_sub, mode='plotly', scale='energy'), use_container_width=True, key='plot_sub_psd2_r'+filename)

            if plot_vpf_s or plot_vpf_n or plot_vpf_r:

                st.subheader("VPF")
                
                print(st.session_state.im_geoph_list)
                print(list(set(st.session_state.im_geoph_list) - {'im_power'}))
                # breakpoint()
                
                if plot_vpf_s: 
                    st.subheader("Початковий сигнал")
                    df = ssp.vpf_df(st.session_state.noisy_sig_plot, fs)
                    st.plotly_chart(ssp.plot_time_signals(df, fs, n_cols=n_cols_plots, threshold=0.5, columns=list(set(st.session_state.im_geoph_list) - {'im_power'}), mode="plotly_one"), use_container_width=True, key='plot_one_vpf_subs_s'+filename)
                   

                if plot_vpf_n: 
                    st.subheader("Шум")
                    df = ssp.vpf_df(st.session_state.ref_noise_plot, fs)
                    st.plotly_chart(ssp.plot_time_signals(df, fs, n_cols=n_cols_plots, threshold=0.5, columns=list(set(st.session_state.im_geoph_list) - {'im_power'}), mode="plotly_one"), use_container_width=True, key='plot_one_vpf_subs_n'+filename)
                    
                if plot_vpf_r: 
                    st.subheader("Результат віднімання")
                    df = ssp.vpf_df(st.session_state.res_signal, fs)
                    st.plotly_chart(ssp.plot_time_signals(df, fs, n_cols=n_cols_plots, threshold=0.5, columns=list(set(st.session_state.im_geoph_list) - {'im_power'}), mode="plotly_one"), use_container_width=True, key='plot_one_vpf_subs_r'+filename)



# === Вкладка 11: Уявна енергія ===
with tab11:
    st.subheader("Обчислення векторної поляризаційної фільтрації")
    
    if len(st.session_state.dfs)>0:

        # angl1 = st.number_input("Напрям на джерело сейсмометра 1, градуси", min_value=0.0, value=0.0, step=10.0, key="energ_angl1")
        # angl2 = st.number_input("Напрям на джерело сейсмометра 2, градуси", min_value=0.0, value=0.0, step=10.0, key="energ_angl2")
        # angl3 = st.number_input("Напрям на джерело сейсмометра 3, градуси", min_value=0.0, value=0.0, step=10.0, key="energ_angl3")
        angl1 = 0
        angl2 = 0
        angl3 = 0
        series_vpf = st.selectbox("Оберіть серію зі списку:", list(st.session_state.dfs.keys()), key="energ_sel")
        
        seismometr_vpf = int(st.selectbox("Оберіть сейсмометр для подальшого аналізу:", st.session_state.true_indices, key="energ_sel_seism"))
        

        VrVz_dict = {}
        
        for i, (filename, data) in enumerate(st.session_state.dfs.items()):
            # Отримуємо колонки, що починаються з 'X' та 'Z'
            x_cols = sorted([col for col in data.columns if col.startswith('X')],
                            key=lambda x: x.lstrip('X'))
            z_cols = sorted([col for col in data.columns if col.startswith('Z')],
                            key=lambda x: x.lstrip('Z'))
            
            Vr = {}
            Vz = {}
        
            for col in x_cols:
                suffix = col[len('X'):]
                Vr[suffix] = [data[col]]  # обгортаємо у список 
        
            for col in z_cols:
                suffix = col[len('Z'):]
                Vz[suffix] = [-data[col]]  # інверсія значень
        
            VrVz_dict[filename + 'Vr'] = Vr
            VrVz_dict[filename + 'Vz'] = Vz                
                
                
       
        st.subheader("Графік Ганкеля")
        st.plotly_chart(ssp.plot_hankel(np.array(VrVz_dict[series_vpf+'Vr'][str(seismometr_vpf)])[0], np.array(VrVz_dict[series_vpf+'Vz'][str(seismometr_vpf)])[0], scale=1.0, mode = 'plotly'),use_container_width=True, key="plot_hackl"+str(seismometr_vpf))
        st.subheader("Графік уявної енергії")
        st.plotly_chart(ssp.vpf(np.array(VrVz_dict[series_vpf+'Vr'][str(seismometr_vpf)])[0], np.array(VrVz_dict[series_vpf+'Vz'][str(seismometr_vpf)])[0], fs, mode='plotly'), key="plot_imenrg"+str(seismometr_vpf))
        im_power = ssp.vpf(np.array(VrVz_dict[series_vpf+'Vr'][str(seismometr_vpf)])[0], np.array(VrVz_dict[series_vpf+'Vz'][str(seismometr_vpf)])[0], fs, mode='matrix') 
        im_power_df = pd.DataFrame({'im_power':im_power})
        # st.session_state.dfs[series+"_vpf"] = im_power_df
        
        ####### ------------- Розрахунок спектру та PSD на основі вікна з сигналом

        
        # Поля для введення мінімальної та максимальної частоти
        col1, col2 = st.columns(2)
        with col1:
            vpf_min_time_s = st.number_input("🔻 Початок сигналу", min_value=0.0, value=1.0, step=0.1, format="%.4f", key='min_time_sig1')
        with col2:
            vpf_max_time_s = st.number_input("🔺 Кінець сигналу", min_value=0.0, value=10.0, step=0.1, format="%.4f", key='max_time_sig1')
            
        vpf_cut = ssp.cut_dataframe_time_window(im_power_df, fs, vpf_min_time_s, vpf_max_time_s)
            
        st.subheader("Спектрограма")
        st.pyplot(ssp.spectr_plot(vpf_cut, fs, n_cols=1, columns=['im_power']), use_container_width=True)
        st.subheader("Графік PSD")
        db_scale_vpf = st.checkbox("Показати в шкалі децибел, дБ", value=False, key='db_scale_vpf')
        if db_scale_vpf:
            st.plotly_chart(ssp.psd_plot_df(vpf_cut, fs=fs, n_cols=1, columns=['im_power'], mode='plotly', scale='db'), use_container_width=True,key="plot_vpf_psd1")
            f_vpf, Pxx_vpf = ssp.psd_plot_df(vpf_cut, fs=fs, n_cols=1, columns=['im_power'], mode='matrix', scale='db') 
        else:
            st.plotly_chart(ssp.psd_plot_df(vpf_cut, fs=fs, n_cols=1, columns=['im_power'], mode='plotly', scale='energy'), use_container_width=True,key="plot_vpf_psd2")
            f_vpf, Pxx_vpf = ssp.psd_plot_df(vpf_cut, fs=fs, n_cols=1, columns=['im_power'], mode='matrix', scale='energy')     

        st.subheader("🎚️ Середнє квадратичне значення PSD в діапазоні")
    
        col1, col2 = st.columns(2)
        with col1:
            min_freq_psd = st.number_input("🔻 Мінімальна частота", min_value=0.0, value=20.0, step=1.0,  key='min_freq_psd')
            
        with col2:
            max_freq_psd = st.number_input("🔺 Максимальна частота", min_value=0.0, value=50.0, step=1.0, key='max_freq_psd')
            
            
        
        print(f[0])
        print(Pxx[0])
        rms_psd, range_freq_val = ssp.rms_in_band(f_vpf[0], Pxx_vpf[0], min_freq_psd, max_freq_psd)
        st.subheader(f"🎚️ RMS PSD дорівнює {rms_psd} в діапазоні від {range_freq_val[0]} Гц до {range_freq_val[1]} Гц")







        st.subheader("🎚️ Часові вікна сигналу та шуму")
        
        with st.form("vpf_window_form2", clear_on_submit=False):
            # Поля для введення мінімальної та максимальної частоти
            col1, col2 = st.columns(2)
            with col1:
                vpf_min_time_s = st.number_input("🔻 Початок сигналу", min_value=0.0, value=1.0, step=0.1, format="%.4f", key='min_time_sig2')
            with col2:
                vpf_max_time_s = st.number_input("🔺 Кінець сигналу", min_value=0.0, value=10.0, step=0.1, format="%.4f", key='max_time_sig2')
            col1, col2 = st.columns(2)
            with col1:
                vpf_min_time_n = st.number_input("🔻 Початок шуму", min_value=0.0, value=1.0, step=0.1, format="%.4f", key='min_time_noise2')
            with col2:
                vpf_max_time_n = st.number_input("🔺 Кінець шуму", min_value=0.0, value=10.0, step=0.1, format="%.4f", key='max_time_noise2')
            
            submitted = st.form_submit_button("⚙️ Застосувати вікно")
            # Кнопка для запуску фільтрації
        if submitted:
            signal = ssp.cut_dataframe_time_window(im_power_df, fs, vpf_min_time_s, vpf_max_time_s)
            noise = ssp.cut_dataframe_time_window(im_power_df, fs, vpf_min_time_n, vpf_max_time_n)
            
            signal_db = 10*np.log10((np.mean(signal**2))**(1/2)+10**(-12))
            noise_db = 10*np.log10((np.mean(noise**2))**(1/2)+10**(-12))
            
            # snr = (np.mean(signal**2))**(1/2)/(np.mean(noise**2))**(1/2)
            snr = ssp.compute_snr_df(signal, noise)
            
            
            st.subheader("RMS сигналу = " + str(signal_db) + " дБ")
            st.subheader("RMS шуму = " + str(noise_db ) + " дБ")
            st.subheader("Відношення SNR")
            st.dataframe(snr)
            st.subheader("Графік Ганкеля виділенного вікном сигналу")
            st.plotly_chart(ssp.plot_hankel(np.array(VrVz_dict[series_vpf+'Vr'][str(seismometr_vpf)])[0], 
                                            np.array(VrVz_dict[series_vpf+'Vz'][str(seismometr_vpf)])[0], 
                                            scale=1.0, mode = 'plotly', 
                                            start_time=vpf_min_time_s, end_time=vpf_max_time_s),
                            use_container_width=True, key="plot_hackl_w"+str(seismometr_vpf))
            

# === Вкладка 12: Азимутальна когерентність ===


with tab12:
    st.subheader("Обчислення азимутальної когерентності")
    st.subheader("Задайте схему розрахунку")

    corr_arcs = {}
    sel_sig1_list = []
    sel_sig2_list = []
    sel_geo1_list = []
    sel_geo2_list = []

    sel_arc_list = []
    sel_angl_list = []

    if len(st.session_state.dfs)>0:

        n_arc = int(st.number_input("Кількість дуг з сейсмометрами", min_value=0, value=3, step=1, key='arc_numb'))
        seism_list_txt = st.text_area("Введіть кількість сейсмометрів на кожній дузі (через кому):", "7, 9, 11")
        seism_list = None
        if seism_list_txt:
            try:
                seism_list = [float(x.strip()) for x in seism_list_txt.split(',')]
                st.write("Список:", seism_list)
            except ValueError:
                st.error("Неможливо перетворити деякі елементи на числа.")

        if n_arc and seism_list:
        
            # with st.form("azmt_window_form", clear_on_submit=False): 
            col1, col2, col3 = st.columns(3)
            for i in range(1, int(sum(seism_list))-n_arc+1):
                with col1:
                    sel_ser1 = st.selectbox(f"Оберіть першу серію пари №{i} для відображення зі списку:", 
                                                      list(st.session_state.dfs.keys())+list(st.session_state.dfs_vpf.keys())+list(st.session_state.dfs_sum.keys())+list(st.session_state.dfs_sub.keys()),
                                                      key="azmt_sel_ser1"+str(i))
                    sel_geoph1 = st.multiselect("Оберіть геофони для відображення зі списку:", 
                                                        st.session_state.geoph_list+st.session_state.im_geoph_list+['X', 'Z', 'Y1', 'Y2'],
                                                        default=st.session_state.def_geoph_list,
                                                        key="azmt_sel_geo1"+str(i))
                with col2:
                    sel_ser2 = st.selectbox(f"Оберіть другу серію пари №{i} для відображення зі списку:", 
                                                      list(st.session_state.dfs.keys())+list(st.session_state.dfs_vpf.keys())+list(st.session_state.dfs_sum.keys())+list(st.session_state.dfs_sub.keys()),
                                                      key="azmt_sel_ser2"+str(i))
                    sel_geoph2 = st.multiselect("Оберіть геофони для відображення зі списку:", 
                                                        st.session_state.geoph_list+st.session_state.im_geoph_list+['X', 'Z', 'Y1', 'Y2'],
                                                        default=st.session_state.def_geoph_list,
                                                        key="azmt_sel_geo2"+str(i))
                with col3:
                    sel_angl = float(st.number_input("Кут другого в парі сейсмометра", min_value=0, value=30, step=1, key='arc_angl'+str(i)))
                    sel_arc = int(st.selectbox("Оберіть номер дуги:", list(range(1, n_arc + 1)), key="arc_numb"+str(i)))
                    
                
                sel_geo1_list.append(sel_geoph1)
                sel_geo2_list.append(sel_geoph2)
                sel_sig1_list.append(sel_ser1)
                sel_sig2_list.append(sel_ser2)
                sel_arc_list.append(sel_arc)
                sel_angl_list.append(sel_angl)
                
            st.write(sel_sig1_list)
            st.write(sel_sig2_list)
            st.write(sel_geo1_list)
            st.write(sel_geo2_list)
            st.write(sel_arc_list)
            st.write(sel_angl_list)
            
            if st.button("⚙️ Застосувати схему розрахунку"):
                st.success("Застосовується схема")
                
                # corr_list = []
                
                    
                # for sig1, sig2, geo1, geo2, arc, angl in zip(sel_sig1_list, sel_sig2_list, sel_geo1_list, sel_geo2_list, sel_arc_list, sel_angl_list):
                    
                #     df1 = st.session_state.dfs[sig1][geo1]
                #     df2 = st.session_state.dfs[sig2][geo2]
                    
                #     st.write(df1)
                #     st.write(df2)
                
                #     corr_list.append(np.corrcoef(df1, df2))
                    
                # st.write(corr_list)
                
                
                        
                #     df = df.add_suffix("_"+df_name)
                #     df = ssp.cut_dataframe_time_window(df, fs=fs, start_time=min_time_list[i], end_time=max_time_list[i])

                #     df_plot_list.append(df)
                # df_plot_list = ssp.align_dataframe_lengths(df_plot_list)
                # df_plot = pd.concat(df_plot_list, axis=1)
                
                # print(df_plot)
                # print(min_time_list)
                # print(max_time_list)
                
                # st.plotly_chart(ssp.plot_time_signals(df_plot, 
                #                                       fs, 
                #                                       n_cols=1, 
                #                                       threshold=0.5, columns=list(df_plot.columns), 
                #                                       mode="plotly_one"), 
                #                 use_container_width=True, key='plots_multy')
                
                # st.subheader("Затримка між сигналами")
                
                # delay_matrix = ssp.compute_delay_matrix(df_plot, fs, method='gcc_phat')
                # st.dataframe(delay_matrix)



# === Вкладка 13: Математична модель сонара ===
with tab13:
    st.subheader("Математична модель затухання")
    st.subheader("Виконайте когерентне віднімання. Побудуйте діаграму VPF")

    if len(st.session_state.dfs_vpf)>0:
        
        with st.form("model_window_form", clear_on_submit=False):

            st.write("Перша частина серії. Оберіть серію з суффіксом _vpf")
            selected_ser1_mat = st.selectbox("Оберіть серію №1 для відображення зі списку:", list(st.session_state.dfs_vpf.keys()), key="mat_ser1")        
            st.write("Друга частина серії. Оберіть серію з суффіксом _vpf")
            selected_ser2_mat = st.selectbox("Оберіть серію для відображення зі списку:", list(st.session_state.dfs_vpf.keys()), key="mat_ser2")        
            st.write("Вкажіть відстані до сейсмометрів серій")
            seismometr1 = float(st.number_input("Вкажіть відстань №1", min_value=1.0, max_value=20.0, value=3.0, step=1.0, key="mat_seism1"))
            seismometr2 = float(st.number_input("Вкажіть відстань №2", min_value=1.0, max_value=20.0, value=5.0, step=1.0, key="mat_seism2"))
            seismometr3 = float(st.number_input("Вкажіть відстань №3", min_value=1.0, max_value=20.0, value=7.5, step=1.0, key="mat_seism3"))
            seismometr4 = float(st.number_input("Вкажіть відстань №4", min_value=1.0, max_value=20.0, value=10.0, step=1.0, key="mat_seism4"))
            seismometr5 = float(st.number_input("Вкажіть відстань №5", min_value=1.0, max_value=20.0, value=12.5, step=1.0, key="mat_seism5"))
            seismometr6 = float(st.number_input("Вкажіть відстань №6", min_value=1.0, max_value=20.0, value=15.0, step=1.0, key="mat_seism6"))
            
            db_scale_model = st.checkbox("Показати в шкалі децибел, дБ", value=False, key='db_scale_model')
            
            submitted = st.form_submit_button("⚙️ Розрахувати")
            
        if submitted:
            
            
            
            min_freq_list = [30, 50, 70, 90, 110, 130, 150, 170, 190, 210, 230, 250, 270]
            max_freq_list = [50, 70, 90, 110, 130, 150, 170, 190, 210, 230, 250, 270, 290]
            
            index_labels = [f"{min_}-{max_}" for min_, max_ in zip(min_freq_list, max_freq_list)]

            # st.session_state.dfs[selected_ser1] - вибрана серія, що містить 3 колонки значень після VPF
            
            df_vpf_dict = {}
            
            
            seismometr_list = [seismometr1, seismometr2, seismometr3,
                               seismometr4, seismometr5, seismometr6]
            i = 0
            for col in ['im_power1', 'im_power2', 'im_power3']:
                seism_freq_vals1 = []
                seism_freq_vals2 = []
                for min_freq, max_freq in zip(min_freq_list, max_freq_list):
                    
                    f1, Pxx1 = ssp.psd_plot_df(st.session_state.dfs_vpf[selected_ser1_mat], fs=fs, n_cols=1, columns=[col], mode='matrix', scale='energy')                
                    rms1, range_freq_val1 = ssp.rms_in_band(f1[0], Pxx1[0], min_freq, max_freq)        
                    seism_freq_vals1.append(float(rms1))
                    
                    f2, Pxx2 = ssp.psd_plot_df(st.session_state.dfs_vpf[selected_ser2_mat], fs=fs, n_cols=1, columns=[col], mode='matrix', scale='energy')                
                    rms2, range_freq_val2 = ssp.rms_in_band(f2[0], Pxx2[0], min_freq, max_freq)        
                    seism_freq_vals2.append(float(rms2))
                    
                    print(st.session_state.dfs_vpf[selected_ser1_mat])
                    print(st.session_state.dfs_vpf[selected_ser2_mat])
                    
                print(seism_freq_vals1)
                print(seism_freq_vals2)
                    
                df_vpf_dict[seismometr_list[i]] = seism_freq_vals1         
                df_vpf_dict[seismometr_list[i+3]] = seism_freq_vals2         
                i = i + 1
            
            
            df_vpf = pd.DataFrame({k: df_vpf_dict[k] for k in sorted(df_vpf_dict)})
            df_vpf.index = index_labels
            
            print(df_vpf_dict)
            print(df_vpf)
            
            
            
            st.subheader("RMS від PSD сигналу на фіксованих відстанях після VPF на фіксованій частоті")
            if db_scale_model:
                st.dataframe(ssp.to_decibels(df_vpf), height=495)
            else:
                st.dataframe(df_vpf.style.format("{:.4e}"), height=495)
            
            st.subheader("Питомі втрати на фіксованих відстанях після VPF на фіксованої частоті")
            
            selected_col = st.selectbox("Оберіть відстань як базову для визначення затухання (зазвичай найменша)", list(df_vpf.columns), key="mat_sel_col")        
            
            # Відняти значення стовпця від усіх інших стовпців (покроково)
            # df_vpf_sub = df_vpf.drop(columns=selected_col).subtract(df_vpf[selected_col], axis=0)
            
            df_vpf_sub = df_vpf.drop(columns=selected_col).divide(df_vpf[selected_col], axis=0)
            
            if db_scale_model:
                st.dataframe(ssp.to_decibels(df_vpf_sub), height=495)
            else:
                st.dataframe(df_vpf_sub.style.format("{:.4e}"), height=495)
            
            st.subheader("Теоретичні коефіцієнти затухання амплітуд в залежності від відстані. Втрати на розповсюдження")
            
            seismometr_dist = [x - seismometr_list[0] for x in seismometr_list]
            
            df_wave_diminish_cols = [x for x in seismometr_dist if x>0]
            
            wave_diminish = [1/x for x in df_wave_diminish_cols]
            
            print(seismometr_dist)
            print(df_wave_diminish_cols)
            print(wave_diminish)
            
            df_wave_diminish = pd.DataFrame(wave_diminish)
            df_wave_diminish = df_wave_diminish.T
            df_wave_diminish.columns = df_wave_diminish_cols
            
            
            if db_scale_model:
                st.dataframe(ssp.to_decibels(df_wave_diminish))
            else:
                st.dataframe(df_wave_diminish)
            
            st.subheader("Емпіричні коефіцієнти затухання амплітуд в залежності від відстані. Втрати на розповсюдження")
            
            df_wave_diminish_emp = pd.DataFrame(df_vpf_sub.mean())
            df_wave_diminish_emp = df_wave_diminish_emp.T
            df_wave_diminish_emp.columns = df_wave_diminish_cols
            
            
            if db_scale_model:
                st.dataframe(ssp.to_decibels(df_wave_diminish_emp))
            else:
                st.dataframe(df_wave_diminish_emp)


            
            
            
            
