import streamlit as st
import joblib
import numpy as np
import os

# Настройка страницы
st.set_page_config(page_title="Калькулятор риска ВМД", layout="centered")
st.title("🔬 Калькулятор риска низкого зрения и фенотипирование ВМД")
st.markdown("Введите значения признаков для пациента с ВМД:")

# Функция загрузки моделей (кешируется для ускорения)
@st.cache_resource
def load_models():
    models_dir = "models"
    # Модели логистической регрессии
    logreg_default = joblib.load(os.path.join(models_dir, "logreg_model.pkl"))
    scaler_default = joblib.load(os.path.join(models_dir, "scaler_lr.pkl"))
    logreg_balanced = joblib.load(os.path.join(models_dir, "logreg_balanced_model.pkl"))
    scaler_balanced = joblib.load(os.path.join(models_dir, "scaler_balanced.pkl"))
    # Модель кластеризации
    kmeans = joblib.load(os.path.join(models_dir, "kmeans_model.pkl"))
    scaler_kmeans = joblib.load(os.path.join(models_dir, "scaler_kmeans.pkl"))
    # Список признаков (порядок важен)
    with open(os.path.join(models_dir, "feature_names.txt"), 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    return (logreg_default, scaler_default, logreg_balanced, scaler_balanced,
            kmeans, scaler_kmeans, feature_names)

# Загрузка моделей
logreg_default, scaler_default, logreg_balanced, scaler_balanced, kmeans, scaler_kmeans, feature_names = load_models()

# Интерфейс ввода данных (9 признаков)
col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Возраст (лет)", min_value=40, max_value=100, value=75, step=1)
    cts_foveola = st.number_input("ЦТС foveola (мкм)", min_value=100.0, max_value=600.0, value=274.0, step=1.0)
    cme = st.selectbox("Кистозный отёк (CME)", options=[0, 1], format_func=lambda x: "Нет (0)" if x == 0 else "Есть (1)")
with col2:
    cts_inner = st.number_input("ЦТС sup.inner fovea (мкм)", min_value=100.0, max_value=600.0, value=299.0, step=1.0)
    cts_outer = st.number_input("ЦТС sup.out. fovea (мкм)", min_value=100.0, max_value=600.0, value=303.0, step=1.0)
    fibrovask = st.selectbox("Фиброваскулярная отслойка", options=[0, 1], format_func=lambda x: "Нет (0)" if x == 0 else "Есть (1)")
with col3:
    avg_vol = st.number_input("Средний объём (average volume)", min_value=200.0, max_value=400.0, value=275.0, step=1.0)
    druses = st.selectbox("Друзы", options=[0, 1], format_func=lambda x: "Нет (0)" if x == 0 else "Есть (1)")
    hm = st.selectbox("Гиперрефлективный материал", options=[0, 1], format_func=lambda x: "Нет (0)" if x == 0 else "Есть (1)")

# Выбор режима прогнозирования
mode = st.radio(
    "Выберите режим прогнозирования:",
    ["Стандартный (высокая специфичность)", "Чувствительный (поиск всех случаев)"],
    help="Стандартный – реже ошибается, но может пропустить болезнь. Чувствительный – находит почти всех больных, но чаще даёт ложные тревоги."
)

# Кнопка расчёта
if st.button("🚀 Рассчитать риск"):
    # Формируем массив признаков в правильном порядке
    input_features = np.array([age, cts_foveola, cts_inner, cts_outer, avg_vol,
                               cme, fibrovask, druses, hm]).reshape(1, -1)

    # Выбор модели и масштабировщика в зависимости от режима
    if mode == "Стандартный (высокая специфичность)":
        scaler = scaler_default
        logreg = logreg_default
        model_note = " (стандартный режим)"
    else:
        scaler = scaler_balanced
        logreg = logreg_balanced
        model_note = " (чувствительный режим)"

    # Масштабирование и предсказание вероятности низкого зрения
    input_scaled = scaler.transform(input_features)
    risk_probability = logreg.predict_proba(input_scaled)[0, 1]

    # Кластеризация (общая для обоих режимов)
    input_scaled_kmeans = scaler_kmeans.transform(input_features)
    cluster_num = kmeans.predict(input_scaled_kmeans)[0]

    # Названия кластеров
    cluster_names = {
        0: "Кластер 0: Активная неоваскуляризация",
        1: "Кластер 1: Отёчный/фиброзный (наиболее тяжёлый)",
        2: "Кластер 2: Друзный/ранний"
    }
    cluster_meaning = cluster_names.get(cluster_num, "Неизвестный кластер")

    # Рекомендации по кластеру
    cluster_recommendations = {
        0: "🔵 **Рекомендация:** активная anti-VEGF терапия. Ежемесячный мониторинг ОКТ и остроты зрения.",
        1: "🔴 **Рекомендация:** интенсивная anti-VEGF терапия (режим treat-and-extend). Рассмотреть кортикостероиды при отёке. Контроль каждые 4–6 недель.",
        2: "🟢 **Рекомендация:** динамическое наблюдение каждые 6–12 месяцев. Коррекция факторов риска. При появлении активности – начало терапии."
    }

    # Вывод результатов
    st.subheader("📊 Результаты")
    col_res1, col_res2 = st.columns(2)
    with col_res1:
        st.metric("Риск низкого зрения", f"{risk_probability:.1%}")
    with col_res2:
        st.metric("Фенотип", cluster_meaning.split(':')[0], help=cluster_meaning)

    # Отображение рекомендации
    st.info(cluster_recommendations[cluster_num])

    # Интерпретация риска в зависимости от режима
    if mode == "Стандартный (высокая специфичность)":
        if risk_probability < 0.2:
            st.success(f"**Низкий риск** (вероятность {risk_probability:.1%}). С высокой уверенностью можно исключить низкое зрение.")
        elif risk_probability < 0.5:
            st.warning(f"**Средний риск** ({risk_probability:.1%}). Рекомендуется дополнительное обследование.")
        else:
            st.error(f"**Высокий риск** ({risk_probability:.1%}). Высокая вероятность низкого зрения, требуется срочная консультация.")
    else:  # чувствительный режим
        if risk_probability < 0.1:
            st.success(f"**Очень низкая вероятность** ({risk_probability:.1%}). Низкое зрение маловероятно.")
        elif risk_probability < 0.3:
            st.info(f"**Умеренная вероятность** ({risk_probability:.1%}). Рекомендуется наблюдение.")
        elif risk_probability < 0.5:
            st.warning(f"**Повышенная вероятность** ({risk_probability:.1%}). Желательно углублённое обследование.")
        else:
            st.error(f"**Высокая вероятность** ({risk_probability:.1%}). Пациент в группе высокого риска, требуется активное вмешательство.")

    # Примечания
    st.caption(f"*Кластер интерпретируется как: {cluster_meaning}*")
    st.caption(f"*Использована модель{model_note}*")
    st.caption("⚠️ Рекомендации носят справочный характер и не заменяют клинического решения врача.")