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

@st.cache_data
def load_file(file_name):
    # st.write("Завантаження...")  # Це буде видно лише 1 раз
    df = pd.read_csv(file_name, header=None, sep=';')
    df.columns = ['X1','Y11','Y12','Z1','X2','Y21','Y22','Z2','X3','Y31','Y32','Z3']
    return df

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
    if uploaded_files and len(st.session_state.dfs)==0:
        for file in uploaded_files:
            dfs[file.name] = load_file(file)
        st.write("Завантаження файлів...")    
        st.session_state.dfs = dfs
        st.session_state.count = st.session_state.count + 1  

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
            min_time = st.number_input("🔻 min час", min_value=0.0, value=1.0, step=1.0, key='min_time')
        with col2:
            max_time = st.number_input("🔺 max час", min_value=0.0, value=10.0, step=1.0, key='max_time')
        
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
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs(["📈 Дані", "📊 Графіки", "Спектр", "PSD", "RMS за PSD",  "Крос-кореляція", "Когерентність", "Когерентне віднімання", "Уявна частина комплексної потужності", "Математична модель сонара"])



# === Вкладка 1: Дані ===
with tab1:
    
        # 3. Виведення результатів
        st.success(f"✅ Завантажено {len(dfs)} файлів")
        
        for filename, df in st.session_state.dfs.items():
            st.subheader(f"📄 Файл: {filename}")
            st.write(df.head())
    
            # Додатково: інформація
            st.text(f"Форма: {df.shape[0]} рядків × {df.shape[1]} колонок")
            st.write(df.describe())
            
    


# === Вкладка 2: Графіки ===
with tab2:
    
    
    st.subheader("Графіки у домені амплітуда-час")
    n_cols = int(st.number_input("Кількість колонок для відображення", min_value=0.0, value=3.0, step=1.0, key='n_col'))
    selected = st.multiselect("Оберіть геофони для відображення зі списку:", ['X1','Y11','Y12','Z1','X2','Y21','Y22','Z2','X3','Y31','Y32','Z3'], default=['X1', 'X2', 'X3', 'Z1', 'Z2', 'Z3'], key='sel_geoph')
    one_plot = st.checkbox("Показати всі геофони на одному графіку", value=True, key='one_plot')
    
    
    if len(st.session_state.dfs)>0:
        for filename, data in st.session_state.dfs.items():
            st.subheader(filename)
            # st.write(filename)
            # st.caption(filename)
            
            if all(elem in list(data.columns) for elem in selected):
            
                if one_plot:
                    st.plotly_chart(ssp.plot_time_signals(data, fs, n_cols=n_cols, threshold=0.5, columns=selected, mode="plotly_one"), use_container_width=True, key='plot_one'+filename)
                else:
                    st.plotly_chart(ssp.plot_time_signals(data, fs, n_cols=n_cols, threshold=0.5, columns=selected, mode="plotly"), use_container_width=True, key='plot_many'+filename)
    

# === Вкладка 3: Спектр ===
with tab3:
    st.subheader("Спектрограми. Представлення у домені частота-час")
    
    with st.form("spectr_window_form", clear_on_submit=False):
        seg_len_s = st.number_input("Довжина одного сегмента спектрограми, с", min_value=0.0, value=None, step=0.1, key='nperseg')
        overlap_s = st.number_input("Величина перекриття між сегментами, с", min_value=0.0, value=None, step=0.01, key='noverlap')
        submitted = st.form_submit_button("⚙️ Застосувати параметри")
    
    
    
    if submitted and len(st.session_state.dfs)>0:
        for filename, data in st.session_state.dfs.items():
            st.subheader(filename)
            
            if all(elem in list(data.columns) for elem in selected):
                st.pyplot(ssp.spectr_plot(data, fs, n_cols=n_cols, columns=selected,seg_len_s=seg_len_s, overlap_s=overlap_s), use_container_width=True)
    
# === Вкладка 4: PSD ===
with tab4:
    st.subheader("Спектральна щільність потужності (PSD)")
    db_scale = st.checkbox("Показати в шкалі децибел, дБ", value=True, key='db_scale1')
    
    if len(st.session_state.dfs)>0:
        for filename, data in st.session_state.dfs.items():
            st.subheader(filename)
            if all(elem in list(data.columns) for elem in selected):
                if db_scale:
                    st.plotly_chart(ssp.psd_plot_df(data, fs=fs, n_cols=n_cols, columns=selected, mode='plotly',scale='db'), use_container_width=True, key='plot_psd1'+filename)
                else:
                    st.plotly_chart(ssp.psd_plot_df(data, fs=fs, n_cols=n_cols, columns=selected, mode='plotly',scale='energy'), use_container_width=True, key='plot_psd2'+filename)

# === Вкладка 5: RMS за PSD ===
with tab5:
    st.subheader("Графік PSD")
    
    if len(st.session_state.dfs)>0:
        selected_ser_psd = st.selectbox("Оберіть серію для відображення зі списку:", list(st.session_state.dfs.keys()),key="sel_rms_psd1")
        selected_geoph_psd = st.selectbox("Оберіть геофони для відображення зі списку:", ['X1','Y11','Y12','Z1','X2','Y21','Y22','Z2','X3','Y31','Y32','Z3'],key="sel_rms_psd2")
    
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
    

# === Вкладка 6: Крос-кореляція ===
with tab6:
    st.subheader("Затримки в сигналах геофонів, обчислені за методом крос-кореляції")
    # n_min = st.number_input("Мінімальне негативне значення діапазону", min_value=-100.0, value=-0.07, step=0.01, key='n_min')
    # n_max = st.number_input("Максимальне негативне значення діапазону", min_value=-100.0, value=-0.01, step=0.01, key='n_max')
    # p_min = st.number_input("Мінімальне позитивне значення діапазону", min_value=0.0, value=0.01, step=0.01, key='p_min')
    # p_max = st.number_input("Максимальне позитивне значення діапазону", min_value=0.0, value=0.07, step=0.01, key='p_max')
    selected = st.multiselect("Оберіть типи геофонів для відображення зі списку:", ['X', 'Y', 'Z'], default=['X', 'Z'], key='sel_geoph_cros')
    # delays_dict = {key: None for key in selected}
    
    if len(st.session_state.dfs)>0:
        for filename, data in st.session_state.dfs.items():
            st.subheader(filename)
            
            # if all(elem in list(data.columns) for elem in selected):
            # X, Y, Z = ssp.cross_corr_crossval_from_df(data, fs, verbose=False, allowed_lag_ranges_s=[(n_min, n_max),(p_min, p_max)])
            X, Y, Z = ssp.cross_corr_crossval_from_df(data, fs, verbose=False)
            delays_dict = {name: globals()[name] for name in selected}
            st.pyplot(ssp.plot_multiple_delay_matrices(delays_dict))
            
# === Вкладка 7: Когерентність ===
with tab7:
    st.subheader("Когерентність між двома сигналами")

    if len(st.session_state.dfs)>0:

        st.write("1й сигнал")
        selected_ser1 = st.selectbox("Оберіть серію для відображення зі списку:", list(st.session_state.dfs.keys()),key="coh_sel_ser1")
        selected_seism1 = st.selectbox("Оберіть типи геофонів для відображення зі списку:", ['X1','Y11','Y12','Z1','X2','Y21','Y22','Z2','X3','Y31','Y32','Z3'],key="coh_sel_seism1")
        st.write("2й сигнал")
        selected_ser2 = st.selectbox("Оберіть серію для відображення зі списку:", list(st.session_state.dfs.keys()),key="coh_sel_ser2")
        selected_seism2 = st.selectbox("Оберіть типи геофонів для відображення зі списку:", ['X1','Y11','Y12','Z1','X2','Y21','Y22','Z2','X3','Y31','Y32','Z3'],key="coh_sel_seism2")
        st.plotly_chart(ssp.plot_coherence(st.session_state.dfs[selected_ser1][selected_seism1], st.session_state.dfs[selected_ser2][selected_seism2], fs, f"{selected_ser1}, {selected_seism1}", f"{selected_ser2}, {selected_seism2}", mode='plotly'), use_container_width=True, key='plot_coher')
        
        

# === Вкладка 8: Когерентне віднімання ===
with tab8:
    
    
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
        
    dfs_vpf = {}  # ключ: ім'я файлу, значення: DataFrame
    
    # if st.button("Завантажити файли"):
    # Зчитування файлів у словник DataFrame'ів
    if "dfs_vpf" not in st.session_state:
        st.session_state.dfs_vpf = dfs_vpf
        
    if "selected_ser1" not in st.session_state:
        st.session_state.selected_ser1 = 0
    

    if subs_mode == "Одна серія":
        
        st.write("Cигнал")
        selected_ser1 = st.selectbox("Оберіть серію для відображення зі списку:", list(st.session_state.dfs.keys()),key="sel_sub_one")
        st.write("Геофони для аналізу")
        selected_geoph = st.multiselect("Оберіть геофони для відображення зі списку:", ['X1','Y11','Y12','Z1','X2','Y21','Y22','Z2','X3','Y31','Y32','Z3'], default=['X1', 'X2', 'X3', 'Z1', 'Z2', 'Z3'],key="sel_subs")
        
        st.subheader("🎚️ Часові вікна сигналу та шуму")
        
        with st.form("subs_window_form", clear_on_submit=False):
            
                      
 

            # print(selected_ser1)
            # breakpoint()


            if selected_ser1:
                st.subheader("Обрано серію "+selected_ser1)
            
            

            # Поля для введення мінімальної та максимальної частоти
            col1, col2 = st.columns(2)
            with col1:
                subs_min_time_s = st.number_input("🔻 Початок сигналу", min_value=0.0, value=2.53, step=0.1, key='subs_min_time_sig')
            with col2:
                subs_max_time_s = st.number_input("🔺 Кінець сигналу", min_value=0.0, value=2.63, step=0.1, key='subs_max_time_sig')
            col1, col2 = st.columns(2)
            with col1:
                subs_min_time_n = st.number_input("🔻 Початок шуму", min_value=0.0, value=2.63, step=0.1, key='subs_min_time_noise')
            with col2:
                subs_max_time_n = st.number_input("🔺 Кінець шуму", min_value=0.0, value=2.74, step=0.1, key='subs_max_time_noise')
            
            sig_len = subs_max_time_s-subs_min_time_s
            noise_len = subs_max_time_n-subs_min_time_n
            st.write(f"Довжина сигналу {sig_len:.4f} c")
            st.write(f"Довжина шуму {noise_len:.4f} c")
            
            
            seg_len_s = st.number_input("Довжина одного сегмента спектрограми, с", min_value=0.0, value=0.02, step=0.1, key='subs_nperseg')
            overlap_s = st.number_input("Величина перекриття між сегментами, с", min_value=0.0, value=0.018, step=0.01, key='subs_noverlap')
            coherence_threshold = st.number_input("Поріг когерентності", min_value=0.0, value=0.8, step=0.1, key='subs_coher_thersh')
            
            
          

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
            
            st.session_state.selected_ser1 = selected_ser1
            st.write("Обрано серію "+selected_ser1)
            
            # print(selected_ser1)
            # print(st.session_state.selected_ser1)
            # breakpoint()       
            
            noisy_sig_df = st.session_state.dfs[selected_ser1][selected_geoph]
            noisy_sig_df_cut = ssp.cut_dataframe_time_window(noisy_sig_df, fs=fs, start_time=subs_min_time_s, end_time=subs_max_time_s)
            ref_noise_df_cut = ssp.cut_dataframe_time_window(noisy_sig_df, fs=fs, start_time=subs_min_time_n, end_time=subs_max_time_n)
            ref_noise_df_cut = ref_noise_df_cut[0:len(noisy_sig_df_cut)]
        
        
            for geoph in selected_geoph: #['X1','Y11','Y12','Z1','X2','Y21','Y22','Z2','X3','Y31','Y32','Z3']: 
                signal, _, _, _, _, _, _ = ssp.coherent_subtraction_aligned_with_mask(noisy_sig_df_cut[geoph], 
                                                                             ref_noise_df_cut[geoph], 
                                                                             seg_len_s=seg_len_s, 
                                                                             overlap_s=overlap_s,
                                                                             coherence_threshold=coherence_threshold)
                signal = signal[:len(noisy_sig_df_cut)]
                data_geoph[geoph] = signal
                
            st.session_state.plot_flag = True
            
            cleaned_sig_df = pd.DataFrame(data_geoph)
            st.session_state.dfs[st.session_state.selected_ser1+"_subtract_cut"] = cleaned_sig_df
  
            st.session_state.noisy_sig_plot = noisy_sig_df_cut
            st.session_state.ref_noise_plot = ref_noise_df_cut
            st.session_state.res_signal = cleaned_sig_df
            
            selected_ser1 = None

                
    
    if subs_mode == "Дві серії":

        st.write("1й сигнал")
        selected_ser1 = st.selectbox("Оберіть серію для відображення зі списку:", list(st.session_state.dfs.keys()),key="sel_sub1")
        # selected_seism1 = st.selectbox("Оберіть типи геофонів для відображення зі списку:", ['X1','Y11','Y12','Z1','X2','Y21','Y22','Z2','X3','Y31','Y32','Z3'],key="select12")
        st.write("2й сигнал (шум)")
        selected_ser2 = st.selectbox("Оберіть серію для відображення зі списку:", list(st.session_state.dfs.keys()),key="sel_sub2")        
        st.write("Геофони для аналізу")
        selected_geoph = st.multiselect("Оберіть геофони для відображення зі списку:", ['X1','Y11','Y12','Z1','X2','Y21','Y22','Z2','X3','Y31','Y32','Z3'], default=['X1', 'X2', 'X3', 'Z1', 'Z2', 'Z3'],key="sel_subs")
        




        with st.form("subs_window_form2", clear_on_submit=False):    
            
            
            if selected_ser1:
                st.subheader("Обрано серію сигналу "+selected_ser1)
            if selected_ser2:
                st.subheader("Обрано серію шуму "+selected_ser2)
            
            if len(st.session_state.dfs)>0:
        
                
                
                
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
            
            st.session_state.selected_ser1 = selected_ser1
            # st.session_state.selected_ser2 = selected_ser2
            st.write("Обрано серію сигналу "+selected_ser1)
            st.write("Обрано серію шуму "+selected_ser2)
            
            # print(selected_ser1)
            # print(st.session_state.selected_ser1)
            # breakpoint()       
               
            
            for geoph in selected_geoph: # ['X1','Y11','Y12','Z1','X2','Y21','Y22','Z2','X3','Y31','Y32','Z3']: 
                signal, _, _, _, _, _, _ = ssp.coherent_subtraction_aligned_with_mask(st.session_state.dfs[selected_ser1][geoph], st.session_state.dfs[selected_ser2][geoph], seg_len_s=None, overlap_s=None,coherence_threshold=0.7)
                signal = signal[:len(st.session_state.dfs[selected_ser1][geoph])]
                data_geoph[geoph] = signal
                
           
            st.session_state.plot_flag = True
            
            cleaned_sig_df = pd.DataFrame(data_geoph)
            st.session_state.dfs[selected_ser1+"_subtract"] = df
            
            st.session_state.noisy_sig_plot = st.session_state.dfs[selected_ser1]
            st.session_state.ref_noise_plot = st.session_state.dfs[selected_ser2]
            st.session_state.res_signal = cleaned_sig_df


    
    if st.session_state.plot_flag:
        
        df = ssp.vpf_df(st.session_state.res_signal, fs)                        
        if subs_mode == "Дві серії":
            st.session_state.dfs_vpf[st.session_state.selected_ser1+"_subtract_vpf"] = df
        if subs_mode == "Одна серія":
            st.session_state.dfs_vpf[st.session_state.selected_ser1+"_subtract_cut_vpf"] = df
            
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
                        st.plotly_chart(ssp.plot_time_signals(st.session_state.noisy_sig_plot, fs, n_cols=n_cols, threshold=0.5, columns=selected_geoph, mode="plotly_one"), use_container_width=True, key='plot_one_subs_s'+filename)
                    else:
                        st.plotly_chart(ssp.plot_time_signals(st.session_state.noisy_sig_plot, fs, n_cols=n_cols, threshold=0.5, columns=selected_geoph, mode="plotly"), use_container_width=True, key='plot_many_subs_s'+filename)

                if plot_figs_n: 
                    st.subheader("Шум")
                    if one_plot_subs:
                        st.plotly_chart(ssp.plot_time_signals(st.session_state.ref_noise_plot, fs, n_cols=n_cols, threshold=0.5, columns=selected_geoph, mode="plotly_one"), use_container_width=True, key='plot_one_subs_n'+filename)
                    else:
                        st.plotly_chart(ssp.plot_time_signals(st.session_state.ref_noise_plot, fs, n_cols=n_cols, threshold=0.5, columns=selected_geoph, mode="plotly"), use_container_width=True, key='plot_many_subs_n'+filename)

                if plot_figs_r: 
                    st.subheader("Результат віднімання")
                    if one_plot_subs:
                        st.plotly_chart(ssp.plot_time_signals(st.session_state.res_signal, fs, n_cols=n_cols, threshold=0.5, columns=selected_geoph, mode="plotly_one"), use_container_width=True, key='plot_one_subs_r'+filename)
                    else:
                        st.plotly_chart(ssp.plot_time_signals(st.session_state.res_signal, fs, n_cols=n_cols, threshold=0.5, columns=selected_geoph, mode="plotly"), use_container_width=True, key='plot_many_subs_r'+filename)


            if plot_spectr_s or plot_spectr_n or plot_spectr_r:

                st.subheader("Спектрограмма")
                
                if plot_spectr_s: 
                    st.subheader("Початковий сигнал")
                   
                    st.pyplot(ssp.spectr_plot(st.session_state.noisy_sig_plot, fs, n_cols=n_cols, columns=selected_geoph), use_container_width=True)
                   

                if plot_spectr_n: 
                    st.subheader("Шум")
                    
                    st.pyplot(ssp.spectr_plot(st.session_state.ref_noise_plot, fs, n_cols=n_cols, columns=selected_geoph), use_container_width=True)
                    
                if plot_spectr_r: 
                    st.subheader("Результат віднімання")
                    st.pyplot(ssp.spectr_plot(st.session_state.res_signal, fs, n_cols=n_cols, columns=selected_geoph), use_container_width=True)
                    

            if plot_psd_s or plot_psd_n or plot_psd_r:

                st.subheader("Графік PSD")
                db_scale_subs = st.checkbox("Показати в шкалі децибел, дБ", value=True, key='db_scale_subs')
                
                if plot_psd_s: 
                    st.subheader("Початковий сигнал")
                    if db_scale_subs:
                        st.plotly_chart(ssp.psd_plot_df(st.session_state.noisy_sig_plot, fs, n_cols=n_cols, columns=selected_geoph, mode='plotly', scale='db'), use_container_width=True, key='plot_sub_psd1_s'+filename)
                    else:
                        st.plotly_chart(ssp.psd_plot_df(st.session_state.noisy_sig_plot, fs, n_cols=n_cols, columns=selected_geoph, mode='plotly', scale='energy'), use_container_width=True, key='plot_sub_psd2_s'+filename)

                if plot_psd_n: 
                    st.subheader("Шум")
                    if db_scale_subs:
                        st.plotly_chart(ssp.psd_plot_df(st.session_state.ref_noise_plot, fs, n_cols=n_cols, columns=selected_geoph, mode='plotly', scale='db'), use_container_width=True, key='plot_sub_psd1_n'+filename)
                    else:
                        st.plotly_chart(ssp.psd_plot_df(st.session_state.ref_noise_plot, fs, n_cols=n_cols, columns=selected_geoph, mode='plotly', scale='energy'), use_container_width=True, key='plot_sub_psd2_n'+filename)

                if plot_psd_r: 
                    st.subheader("Результат віднімання")
                    if db_scale_subs:
                        st.plotly_chart(ssp.psd_plot_df(st.session_state.res_signal, fs, n_cols=n_cols, columns=selected_geoph, mode='plotly', scale='db'), use_container_width=True, key='plot_sub_psd1_r'+filename)
                    else:
                        st.plotly_chart(ssp.psd_plot_df(st.session_state.res_signal, fs, n_cols=n_cols, columns=selected_geoph, mode='plotly', scale='energy'), use_container_width=True, key='plot_sub_psd2_r'+filename)

            if plot_vpf_s or plot_vpf_n or plot_vpf_r:

                st.subheader("VPF")
                
                
                
                if plot_vpf_s: 
                    st.subheader("Початковий сигнал")
                    df = ssp.vpf_df(st.session_state.noisy_sig_plot, fs)
                    st.plotly_chart(ssp.plot_time_signals(df, fs, n_cols=n_cols, threshold=0.5, columns=['im_power1', 'im_power2', 'im_power3'], mode="plotly_one"), use_container_width=True, key='plot_one_vpf_subs_s'+filename)
                   

                if plot_vpf_n: 
                    st.subheader("Шум")
                    df = ssp.vpf_df(st.session_state.ref_noise_plot, fs)
                    st.plotly_chart(ssp.plot_time_signals(df, fs, n_cols=n_cols, threshold=0.5, columns=['im_power1', 'im_power2', 'im_power3'], mode="plotly_one"), use_container_width=True, key='plot_one_vpf_subs_n'+filename)
                    
                if plot_vpf_r: 
                    st.subheader("Результат віднімання")
                    df = ssp.vpf_df(st.session_state.res_signal, fs)
                    st.plotly_chart(ssp.plot_time_signals(df, fs, n_cols=n_cols, threshold=0.5, columns=['im_power1', 'im_power2', 'im_power3'], mode="plotly_one"), use_container_width=True, key='plot_one_vpf_subs_r'+filename)



# === Вкладка 9: Уявна енергія ===
with tab9:
    st.subheader("Обчислення векторної поляризаційної фільтрації")
    
    if len(st.session_state.dfs)>0:

        # angl1 = st.number_input("Напрям на джерело сейсмометра 1, градуси", min_value=0.0, value=0.0, step=10.0, key="energ_angl1")
        # angl2 = st.number_input("Напрям на джерело сейсмометра 2, градуси", min_value=0.0, value=0.0, step=10.0, key="energ_angl2")
        # angl3 = st.number_input("Напрям на джерело сейсмометра 3, градуси", min_value=0.0, value=0.0, step=10.0, key="energ_angl3")
        angl1 = 0
        angl2 = 0
        angl3 = 0
        series = st.selectbox("Оберіть серію зі списку:", list(st.session_state.dfs.keys()), key="energ_sel")
        seismometr = int(st.number_input("Оберіть сейсмометр для подальшого аналізу", min_value=1.0, max_value=3.0, value=1.0, step=1.0, key="energ_seism"))

        VrVz_dict = {}

        for i, (filename, data) in enumerate(st.session_state.dfs.items()):
            # st.write("Файл ", filename, " індекс серії  ", str(i+1))
            # st.subheader(str(i))
            if 'X1' in list(data.columns):
    
                Vr1 = []
                Vr2 = []
                Vr3 = []
                Vz1 = []
                Vz2 = []
                Vz3 = []
                # st.write(i)
                # st.write(filename)
                # Vr1.append(ssp.compute_radial(data['X1'], data['Y11'], data['Y12'], angl1))
                # Vr2.append(ssp.compute_radial(data['X2'], data['Y21'], data['Y22'], angl2))
                # Vr3.append(ssp.compute_radial(data['X3'], data['Y31'], data['Y32'], angl2))
                
                Vr1.append(data['X1'])
                Vr2.append(data['X2'])
                Vr3.append(data['X3'])
                Vz1.append(-data['Z1'])
                Vz2.append(-data['Z2'])
                Vz3.append(-data['Z3'])
                Vr = {'1':Vr1, '2':Vr2, '3':Vr3}
                Vz = {'1':Vz1, '2':Vz2, '3':Vz3}
                VrVz_dict[filename+'Vr'] = Vr
                VrVz_dict[filename+'Vz'] = Vz
                
                # st.write(filename+'Vr')
                # st.write(filename+'Vz')
                
       
        st.subheader("Графік Ганкеля")
        st.plotly_chart(ssp.plot_hankel(np.array(VrVz_dict[series+'Vr'][str(seismometr)])[0], np.array(VrVz_dict[series+'Vz'][str(seismometr)])[0], scale=1.0, mode = 'plotly'),use_container_width=True, key="plot_hackl"+str(seismometr))
        st.subheader("Графік уявної енергії")
        st.plotly_chart(ssp.vpf(np.array(VrVz_dict[series+'Vr'][str(seismometr)])[0], np.array(VrVz_dict[series+'Vz'][str(seismometr)])[0], fs, mode='plotly'), key="plot_imenrg"+str(seismometr))
        im_power = ssp.vpf(np.array(VrVz_dict[series+'Vr'][str(seismometr)])[0], np.array(VrVz_dict[series+'Vz'][str(seismometr)])[0], fs, mode='matrix') 
        im_power_df = pd.DataFrame({'im_power':im_power})
        # st.session_state.dfs[series+"_vpf"] = im_power_df
        
        st.subheader("Спектрограма")
        st.pyplot(ssp.spectr_plot(im_power_df, fs, n_cols=1, columns=['im_power']), use_container_width=True)
        st.subheader("Графік PSD")
        db_scale_vpf = st.checkbox("Показати в шкалі децибел, дБ", value=False, key='db_scale_vpf')
        if db_scale_vpf:
            st.plotly_chart(ssp.psd_plot_df(im_power_df, fs=fs, n_cols=1, columns=['im_power'], mode='plotly', scale='db'), use_container_width=True,key="plot_vpf_psd1")
            f, Pxx = ssp.psd_plot_df(im_power_df, fs=fs, n_cols=1, columns=['im_power'], mode='matrix', scale='db') 
        else:
            st.plotly_chart(ssp.psd_plot_df(im_power_df, fs=fs, n_cols=1, columns=['im_power'], mode='plotly', scale='energy'), use_container_width=True,key="plot_vpf_psd2")
            f, Pxx = ssp.psd_plot_df(im_power_df, fs=fs, n_cols=1, columns=['im_power'], mode='matrix', scale='energy') 

        st.subheader("🎚️ Середнє квадратичне значення PSD в діапазоні")

        with st.form("psd_window_form", clear_on_submit=False):
            # Поля для введення мінімальної та максимальної частоти
            col1, col2 = st.columns(2)
            with col1:
                min_freq_psd = st.number_input("🔻 Мінімальна частота", min_value=0.0, value=20.0, step=1.0, key='min_freq_psd')
                
            with col2:
                max_freq_psd = st.number_input("🔺 Максимальна частота", min_value=0.0, value=50.0, step=1.0, key='max_freq_psd')
            submitted = st.form_submit_button("⚙️ Розрахувати")
            
            if submitted:
                print(f[0])
                print(Pxx[0])
                rms_psd, range_freq_val = ssp.rms_in_band(f[0], Pxx[0], min_freq_psd, max_freq_psd)
                st.subheader(f"🎚️ RMS PSD дорівнює {rms_psd} в діапазоні від {range_freq_val[0]} Гц до {range_freq_val[1]} Гц")



        st.subheader("🎚️ Часові вікна сигналу та шуму")
        
        with st.form("vpf_window_form", clear_on_submit=False):
            # Поля для введення мінімальної та максимальної частоти
            col1, col2 = st.columns(2)
            with col1:
                min_time_s = st.number_input("🔻 Початок сигналу", min_value=0.0, value=1.0, step=0.1, key='min_time_sig')
            with col2:
                max_time_s = st.number_input("🔺 Кінець сигналу", min_value=0.0, value=10.0, step=0.1, key='max_time_sig')
            col1, col2 = st.columns(2)
            with col1:
                min_time_n = st.number_input("🔻 Початок шуму", min_value=0.0, value=1.0, step=0.1, key='min_time_noise')
            with col2:
                max_time_n = st.number_input("🔺 Кінець шуму", min_value=0.0, value=10.0, step=0.1, key='max_time_noise')
            
            submitted = st.form_submit_button("⚙️ Застосувати вікно")
            # Кнопка для запуску фільтрації
        if submitted:
            signal = ssp.cut_dataframe_time_window(im_power_df, fs, min_time_s, max_time_s)
            noise = ssp.cut_dataframe_time_window(im_power_df, fs, min_time_n, max_time_n)
            
            signal_db = 10*np.log10((np.mean(signal**2))**(1/2)+10**(-12))
            noise_db = 10*np.log10((np.mean(noise**2))**(1/2)+10**(-12))
            
            # snr = (np.mean(signal**2))**(1/2)/(np.mean(noise**2))**(1/2)
            snr = ssp.compute_snr_df(signal, noise)
            
            
            st.subheader("RMS сигналу = " + str(signal_db) + " дБ")
            st.subheader("RMS шуму = " + str(noise_db ) + " дБ")
            st.subheader("Відношення SNR")
            st.dataframe(snr)
            st.subheader("Графік Ганкеля виділенного вікном сигналу")
            st.plotly_chart(ssp.plot_hankel(np.array(VrVz_dict[series+'Vr'][str(seismometr)])[0], 
                                            np.array(VrVz_dict[series+'Vz'][str(seismometr)])[0], 
                                            scale=1.0, mode = 'plotly', 
                                            start_time=min_time_s, end_time=max_time_s),
                            use_container_width=True, key="plot_hackl_w"+str(seismometr))
            


# === Вкладка 10: Математична модель сонара ===
with tab10:
    st.subheader("Математична модель затухання")
    st.subheader("Виконайте когерентне віднімання. Побудуйте діаграму VPF")

    if len(st.session_state.dfs_vpf)>0:
        
        with st.form("model_window_form", clear_on_submit=False):

            st.write("Перша частина серії. Оберіть серію з суффіксом _vpf")
            selected_ser1 = st.selectbox("Оберіть серію №1 для відображення зі списку:", list(st.session_state.dfs_vpf.keys()), key="mat_ser1")        
            st.write("Друга частина серії. Оберіть серію з суффіксом _vpf")
            selected_ser2 = st.selectbox("Оберіть серію для відображення зі списку:", list(st.session_state.dfs_vpf.keys()), key="mat_ser2")        
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
                    
                    f1, Pxx1 = ssp.psd_plot_df(st.session_state.dfs_vpf[selected_ser1], fs=fs, n_cols=1, columns=[col], mode='matrix', scale='energy')                
                    rms1, range_freq_val1 = ssp.rms_in_band(f1[0], Pxx1[0], min_freq, max_freq)        
                    seism_freq_vals1.append(float(rms1))
                    
                    f2, Pxx2 = ssp.psd_plot_df(st.session_state.dfs_vpf[selected_ser2], fs=fs, n_cols=1, columns=[col], mode='matrix', scale='energy')                
                    rms2, range_freq_val2 = ssp.rms_in_band(f2[0], Pxx2[0], min_freq, max_freq)        
                    seism_freq_vals2.append(float(rms2))
                    
                    print(st.session_state.dfs_vpf[selected_ser1])
                    print(st.session_state.dfs_vpf[selected_ser2])
                    
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


            
            
            
            
