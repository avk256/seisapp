import git
from git import Repo
from git.exc import GitCommandError

repo_url = "https://github.com/avk256/seisproc.git"
clone_path = "./seisproc"

try:
    Repo.clone_from(repo_url, clone_path)
    print("Репозиторій успішно клоновано!")
except GitCommandError as e:
    print("❌ Помилка клонування репозиторію:")

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
    
    
    # Зчитування файлів у словник DataFrame'ів
    dfs = {}  # ключ: ім'я файлу, значення: DataFrame
    
    if uploaded_files:
        for file in uploaded_files:
            dfs[file.name] = load_file(file)

    if len(dfs)>0:
        
        
        fs = st.number_input("🔻 Частота дискретизації", min_value=800.0, value=800.0, step=10.0)
        min_freq = 0
        min_freq = 0

        # ================ Смуговий фільтр =====================================
        
        st.subheader("🎚️ Смуговий фільтр")
        # Поля для введення мінімальної та максимальної частоти
        col1, col2 = st.columns(2)
        with col1:
            min_freq = st.number_input("🔻 min частота", min_value=0.0, value=20.0, step=10.0)
        with col2:
            max_freq = st.number_input("🔺 max частота", min_value=0.0, value=50.0, step=10.0)
        
        # Кнопка для запуску фільтрації
        if st.button("⚙️ Фільтрувати"):
            st.success(f"Застосовується фільтр: від {min_freq} до {max_freq} Гц")
            # Тут можна викликати функцію фільтрації
            for key, data in zip(dfs.keys(), dfs.values()):
                dfs[key] = ssp.filter_dataframe(data, min_freq, max_freq, fs)

        # ================== Часове вікно =====================================
        
        st.subheader("🎚️ Часове вікно")
        # Поля для введення мінімальної та максимальної частоти
        col1, col2 = st.columns(2)
        with col1:
            min_time = st.number_input("🔻 min час", min_value=0.0, value=1.0, step=1.0)
        with col2:
            max_time = st.number_input("🔺 max час", min_value=0.0, value=10.0, step=1.0)
        
        # Кнопка для запуску фільтрації
        if st.button("⚙️ Застосувати вікно"):
            st.success(f"Застосовується часове вікно: від {min_time} до {max_time} с")
            # Тут можна викликати функцію фільтрації
            for key, data in zip(dfs.keys(), dfs.values()):
                dfs[key] = ssp.cut_dataframe_time_window(data, fs, min_time, max_time)

        # ===================== Детренд =======================================
   
        st.subheader("🎚️ Операція віднімання тренду")
        # Кнопка для запуску детренду
        if st.button("⚙️ Застосувати детренд"):
            st.success("Застосовується детренд")
            # Тут можна викликати функцію фільтрації
            for key, data in zip(dfs.keys(), dfs.values()):
                dfs[key] = ssp.detrend_dataframe(data)
        
    
    
dfs = dfs

# === Вкладки ===
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(["📈 Дані", "📊 Графіки", "Спектр", "PSD", "Крос-кореляція", "Когерентність", "Когерентне віднімання", "Уявна енергія", "Математичне модель сонара"])

# === Вкладка 1: Дані ===
with tab1:
    
        # 3. Виведення результатів
        st.success(f"✅ Завантажено {len(dfs)} файлів")
        
        for filename, df in dfs.items():
            st.subheader(f"📄 Файл: {filename}")
            st.write(df.head())
    
            # Додатково: інформація
            st.text(f"Форма: {df.shape[0]} рядків × {df.shape[1]} колонок")
            st.write(df.describe())
            
    


# === Вкладка 2: Графіки ===
with tab2:
    
    
    st.subheader("Графіки у домені амплітуда-час")
    n_cols = int(st.number_input("Кількість колонок для відображення", min_value=0.0, value=3.0, step=1.0))
    selected = st.multiselect("Оберіть геофони для відображення зі списку:", ['X1','Y11','Y12','Z1','X2','Y21','Y22','Z2','X3','Y31','Y32','Z3'], default=['X1', 'X2', 'X3', 'Z1', 'Z2', 'Z3'])
    one_plot = st.checkbox("Показати всі геофони на одному графіку", value=True)
    
    
    if len(dfs)>0:
        for filename, data in dfs.items():
            st.subheader(filename)
            # st.write(filename)
            # st.caption(filename)
            
            if all(elem in list(data.columns) for elem in selected):
            
                if one_plot:
                    st.plotly_chart(ssp.plot_time_signals(data, fs, n_cols=n_cols, threshold=0.5, columns=selected, mode="plotly_one"), use_container_width=True)
                else:
                    st.plotly_chart(ssp.plot_time_signals(data, fs, n_cols=n_cols, threshold=0.5, columns=selected, mode="plotly"), use_container_width=True)
    

# === Вкладка 3: Спектр ===
with tab3:
    st.subheader("Спектрограми. Представлення у домені частота-час")
    
    if len(dfs)>0:
        for filename, data in dfs.items():
            st.subheader(filename)
            
            if all(elem in list(data.columns) for elem in selected):
                st.pyplot(ssp.spectr_plot(data, fs, n_cols=n_cols, columns=selected), use_container_width=True)
    
# === Вкладка 4: PSD ===
with tab4:
    st.subheader("Спектральна щільність потужності (PSD)")
    
    if len(dfs)>0:
        for filename, data in dfs.items():
            st.subheader(filename)
            if all(elem in list(data.columns) for elem in selected):
                st.plotly_chart(ssp.psd_plot_df(data, fs=fs, n_cols=n_cols, columns=selected, mode='plotly'), use_container_width=True)

# === Вкладка 5: Крос-кореляція ===
with tab5:
    st.subheader("Затримки в сигналах геофонів, обчислені за методом крос-кореляції")
    n_min = st.number_input("Мінімальне негативне значення діапазону", min_value=-100.0, value=-0.07, step=0.01)
    n_max = st.number_input("Максимальне негативне значення діапазону", min_value=-100.0, value=-0.01, step=0.01)
    p_min = st.number_input("Мінімальне позитивне значення діапазону", min_value=0.0, value=0.01, step=0.01)
    p_max = st.number_input("Максимальне позитивне значення діапазону", min_value=0.0, value=0.07, step=0.01)
    selected = st.multiselect("Оберіть типи геофонів для відображення зі списку:", ['X', 'Y', 'Z'], default=['X', 'Z'])
    # delays_dict = {key: None for key in selected}
    
    if len(dfs)>0:
        for filename, data in dfs.items():
            st.subheader(filename)
            
            # if all(elem in list(data.columns) for elem in selected):
            X, Y, Z = ssp.cross_corr_crossval_from_df(data, fs, verbose=False, allowed_lag_ranges_s=[(n_min, n_max),(p_min, p_max)])
            delays_dict = {name: globals()[name] for name in selected}
            st.pyplot(ssp.plot_multiple_delay_matrices(delays_dict))
            
# === Вкладка 6: Когерентність ===
with tab6:
    st.subheader("Когерентність між двома сигналами")

    if len(dfs)>0:

        st.write("1й сигнал")
        selected_ser1 = st.selectbox("Оберіть серію для відображення зі списку:", list(dfs.keys()),key="select1")
        selected_seism1 = st.selectbox("Оберіть типи геофонів для відображення зі списку:", ['X1','Y11','Y12','Z1','X2','Y21','Y22','Z2','X3','Y31','Y32','Z3'],key="select2")
        st.write("2й сигнал")
        selected_ser2 = st.selectbox("Оберіть серію для відображення зі списку:", list(dfs.keys()),key="select3")
        selected_seism2 = st.selectbox("Оберіть типи геофонів для відображення зі списку:", ['X1','Y11','Y12','Z1','X2','Y21','Y22','Z2','X3','Y31','Y32','Z3'],key="select4")
        st.plotly_chart(ssp.plot_coherence(dfs[selected_ser1][selected_seism1], dfs[selected_ser2][selected_seism2], fs, f"{selected_ser1}, {selected_seism1}", f"{selected_ser2}, {selected_seism2}", mode='plotly'), use_container_width=True)
        
        

# === Вкладка 7: Когерентне віднімання ===
with tab7:
    
    st.session_state.dfs = dfs
    
    st.subheader("Когерентне віднімання шуму")
    # st.write(st.session_state.dfs.keys())
    if len(dfs)>0:

        st.write("1й сигнал")
        selected_ser1 = st.selectbox("Оберіть серію для відображення зі списку:", list(dfs.keys()),key="select11")
        # selected_seism1 = st.selectbox("Оберіть типи геофонів для відображення зі списку:", ['X1','Y11','Y12','Z1','X2','Y21','Y22','Z2','X3','Y31','Y32','Z3'],key="select12")
        st.write("2й сигнал (шум)")
        selected_ser2 = st.selectbox("Оберіть серію для відображення зі списку:", list(dfs.keys()),key="select13")        
        st.write("Геофони для аналізу")
        selected_geoph = st.multiselect("Оберіть геофони для відображення зі списку:", ['X1','Y11','Y12','Z1','X2','Y21','Y22','Z2','X3','Y31','Y32','Z3'], default=['X1', 'X2', 'X3', 'Z1', 'Z2', 'Z3'],key="select14")
        
        data_geoph = {}
        for geoph in ['X1','Y11','Y12','Z1','X2','Y21','Y22','Z2','X3','Y31','Y32','Z3']: 
            signal, _, _, _ = ssp.coherent_subtraction_aligned_with_mask(dfs[selected_ser1][geoph], dfs[selected_ser2][geoph], nperseg=2048, noverlap=1792,coherence_threshold=0.7)
            signal = signal[:len(dfs[selected_ser1][geoph])]
            data_geoph[geoph] = signal
            
       


        df = pd.DataFrame(data_geoph)
        st.session_state.dfs[selected_ser1+"_subtract"] = df
        st.subheader("Результат когерентного віднімання")
        if len(df):
            # dfs[selected_ser1+"_subtract"] = df 
            st.subheader("Графік амплітуда-час")
            st.plotly_chart(ssp.plot_time_signals(df, fs, n_cols=n_cols, columns=selected_geoph, threshold=0.5), use_container_width=True)
            st.subheader("Спектрограма")
            st.pyplot(ssp.spectr_plot(df, fs, n_cols=n_cols, columns=selected_geoph), use_container_width=True)
            st.subheader("Графік PSD")
            st.plotly_chart(ssp.psd_plot_df(df, fs=fs, n_cols=n_cols, columns=selected_geoph, mode='plotly'), use_container_width=True)
            
        # st.write(st.session_state.dfs.keys())
        # st.plotly_chart(ssp.plot_coherence(dfs[selected_ser1][selected_seism1], dfs[selected_ser2][selected_seism2], fs, f"{selected_ser1}, {selected_seism1}", f"{selected_ser2}, {selected_seism2}", mode='plotly'), use_container_width=True)


# === Вкладка 8: Уявна енергія ===
with tab8:
    st.subheader("Обчислення векторной поляризаційної фільтрації")

    
    
    if len(dfs)>0:

        angl1 = st.number_input("Напрям на джерело сейсмометра 1, градуси", min_value=0.0, value=0.0, step=10.0)
        angl2 = st.number_input("Напрям на джерело сейсмометра 2, градуси", min_value=0.0, value=0.0, step=10.0)
        angl3 = st.number_input("Напрям на джерело сейсмометра 3, градуси", min_value=0.0, value=0.0, step=10.0)
        series = st.selectbox("Оберіть серію зі списку:", list(st.session_state.dfs.keys()),key="select23")
        seismometr = int(st.number_input("Оберіть сейсмометр для подальшого аналізу", min_value=1.0, max_value=3.0, value=1.0, step=1.0))

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
                Vr1.append(ssp.compute_radial(data['X1'], data['Y11'], data['Y12'], angl1))
                Vr2.append(ssp.compute_radial(data['X2'], data['Y21'], data['Y22'], angl2))
                Vr3.append(ssp.compute_radial(data['X3'], data['Y31'], data['Y32'], angl2))
                Vz1.append(data['Z1'])
                Vz2.append(data['Z2'])
                Vz3.append(data['Z3'])
                Vr = {'1':Vr1, '2':Vr2, '3':Vr3}
                Vz = {'1':Vz1, '2':Vz2, '3':Vz3}
                VrVz_dict[filename+'Vr'] = Vr
                VrVz_dict[filename+'Vz'] = Vz
                
                # st.write(filename+'Vr')
                # st.write(filename+'Vz')
                
       
        st.subheader("Графік Ганкеля")
        st.plotly_chart(ssp.plot_hankel(np.array(VrVz_dict[series+'Vr'][str(seismometr)])[0], np.array(VrVz_dict[series+'Vz'][str(seismometr)])[0], scale=1.0, mode = 'plotly'),use_container_width=True)
        st.subheader("Графік уявної енергії")
        st.plotly_chart(ssp.vpf(np.array(VrVz_dict[series+'Vr'][str(seismometr)])[0], np.array(VrVz_dict[series+'Vz'][str(seismometr)])[0], fs, mode='plotly'))

# === Вкладка 9: Математична модель сонара ===
with tab9:
    st.subheader("Математична модель сонара")

    if len(dfs)>0:

        st.write("Cигнал")
        selected_ser1 = st.selectbox("Оберіть серію для відображення зі списку:", list(st.session_state.dfs.keys()),key="item91")
        #selected_seism1 = st.selectbox("Оберіть типи геофонів для відображення зі списку:", ['X1','Y11','Y12','Z1','X2','Y21','Y22','Z2','X3','Y31','Y32','Z3'],key="select12")
        st.write("Шум")
        selected_ser2 = st.selectbox("Оберіть серію для відображення зі списку:", list(st.session_state.dfs.keys()),key="item92")        
        st.write("Сейсмометр для якого буде виконано аналіз")
        seismometr = int(st.number_input("Оберіть сейсмометр для подальшого аналізу", min_value=1.0, max_value=3.0, value=1.0, step=1.0, key="item93"))

        rho = float(st.number_input("Густина грунта, кг/м³", min_value=0.0, value=2500.0, step=10.0, key="item94"))

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
                Vr1.append(ssp.compute_radial(data['X1'], data['Y11'], data['Y12'], angl1))
                Vr2.append(ssp.compute_radial(data['X2'], data['Y21'], data['Y22'], angl2))
                Vr3.append(ssp.compute_radial(data['X3'], data['Y31'], data['Y32'], angl2))
                Vz1.append(data['Z1'])
                Vz2.append(data['Z2'])
                Vz3.append(data['Z3'])
                Vr = {'1':Vr1, '2':Vr2, '3':Vr3}
                Vz = {'1':Vz1, '2':Vz2, '3':Vz3}
                VrVz_dict[filename+'Vr'] = Vr
                VrVz_dict[filename+'Vz'] = Vz
                
        
        print('tab9')
        print(np.array(VrVz_dict[selected_ser1+'Vr'][str(seismometr)])[0])
        print(np.array(VrVz_dict[selected_ser1+'Vz'][str(seismometr)])[0])
        ls_signal = ssp.energy_density(np.array(VrVz_dict[selected_ser1+'Vr'][str(seismometr)])[0], np.array(VrVz_dict[selected_ser1+'Vz'][str(seismometr)])[0], rho)
        ls_noise = ssp.energy_density(np.array(VrVz_dict[selected_ser2+'Vr'][str(seismometr)])[0], np.array(VrVz_dict[selected_ser2+'Vz'][str(seismometr)])[0], rho)
        
        st.subheader("Щільність енергії джерела, ДБ")
        st.subheader(ls_signal)
        st.subheader("Щільність енергії шуму, ДБ")
        st.subheader(ls_noise)
        st.subheader("SNR")
        st.subheader(ls_signal/ls_noise)
    
    