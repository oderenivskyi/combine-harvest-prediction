import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import preprocess_for_inference

st.set_page_config(
    page_title="Прогноз врожайності", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model():
    model = joblib.load('xgb_reg_model.joblib')
    return model

try:
    model = load_model()
except FileNotFoundError:
    st.error("Файл моделі 'xgb_reg_model.joblib' не знайдено!")
    st.stop()

st.title("🌾 Прогнозування продуктивності збору врожаю")

st.markdown(
    """
    Цей додаток прогнозує **погодинну продуктивність** комбайнів. 
    Завантажте Excel-файл з даними про погоду та техніку, щоб отримати розрахунок.
    """
)

st.sidebar.header("Вхідні дані")
uploaded_file = st.sidebar.file_uploader("Завантажте Excel файл (.xlsx)", type=["xlsx"])


if uploaded_file is not None:
    try:
        df_raw = pd.read_excel(uploaded_file)
        
        st.subheader("1. Огляд завантажених даних")
        st.dataframe(df_raw.head())

        if st.button("Зробити прогноз", type="primary"):
            with st.spinner('Йде розрахунок прогнозу...'):
                try:
                    X_input, df_processed = preprocess_for_inference(df_raw, 'encoder.joblib')
                    
                    predictions = model.predict(X_input)
                    
                    df_processed['Прогноз_га'] = predictions
                    
                    if 'date_time' in df_processed.columns:
                        df_processed.rename(columns={'date_time': 'Дата_час'}, inplace=True)

                    st.success("Розрахунок завершено успішно!")
                    
                    st.divider()
                    st.subheader("2. Результати прогнозу")
                    
                    cols_to_show = ['Дата_час', 'Модель', 'Прогноз_га']
                    if 'Гаражний номер' in df_processed.columns:
                        cols_to_show.insert(1, 'Гаражний номер')
                    
                    st.dataframe(df_processed[cols_to_show], use_container_width=True)
                    
                    # --- Візуалізація ---
                    st.subheader("3. Графік погодинної продуктивності")
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    if 'Гаражний номер' in df_processed.columns:
                        sns.lineplot(data=df_processed, x='Дата_час', y='Прогноз_га', hue='Гаражний номер', marker='o', ax=ax)
                    else:
                        sns.lineplot(data=df_processed, x='Дата_час', y='Прогноз_га', marker='o', color='#1f77b4', ax=ax)
                    
                    plt.title("Динаміка прогнозу збору (по годинах)", fontsize=14)
                    plt.xlabel("Дата та час", fontsize=12)
                    plt.ylabel("Прогноз (га)", fontsize=12)
                    plt.grid(True, linestyle='--', alpha=0.6)
                    plt.xticks(rotation=45)
                    
                    st.pyplot(fig)
                    
                    st.info(f"Всього за цей період прогнозується зібрати: **{df_processed['Прогноз_га'].sum():.2f} га**")

                except Exception as e:
                    st.error(f"Помилка при обробці даних: {e}")
                    st.warning("Перевірте, чи файл 'encoder.joblib' знаходиться в папці з проектом.")

    except Exception as e:
        st.error(f"Не вдалося прочитати файл: {e}")

else:
    # Повідомлення, якщо файл не завантажено (займає основний простір, поки пусто)
    st.info("👈 Для початку роботи завантажте файл у панелі зліва.")