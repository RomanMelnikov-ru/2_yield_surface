import numpy as np
import streamlit as st
import plotly.graph_objects as go

# Глобальные переменные для хранения состояния
epsilon_ref = 0.1  # Параметр для нелинейного закона

# Инициализация состояния сессии
if "max_pc" not in st.session_state:
    st.session_state.max_pc = 0
if "max_epsilon_p" not in st.session_state:
    st.session_state.max_epsilon_p = 0
if "initial_data" not in st.session_state:
    st.session_state.initial_data = None

# Функция для расчета предельного девиаторного напряжения qf
def q_f(p, phi, c):
    M = (6 * np.sin(np.radians(phi))) / (3 - np.sin(np.radians(phi)))
    return M * p + c

# Функция для расчета поверхностей текучести (нелинейный закон)
def yield_surface(p, q_ult, epsilon_p, epsilon_ref):
    return q_ult * (1 - np.exp(-epsilon_p / epsilon_ref))

# Функция для расчета пластической деформации epsilon_p
def calculate_epsilon_p(p_natural, q_natural, phi, c, epsilon_ref):
    q_ult = q_f(p_natural, phi, c)
    if q_natural >= q_ult:
        return np.inf  # Если точка на линии разрушения, epsilon_p стремится к бесконечности
    return -epsilon_ref * np.log(1 - q_natural / q_ult)

# Функция для расчета точки пересечения Cap Hardening Surface и yield_surface
def find_intersection(p, q_cap, q_yield):
    for i in range(len(p) - 1):
        if (q_cap[i] - q_yield[i]) * (q_cap[i + 1] - q_yield[i + 1]) <= 0:
            return p[i], q_cap[i]
    return None, None

# Функция для расчета и построения графика
def plot_graph(gamma, h, c, phi, pc_input, modify_state, pc_slider, epsilon_p_slider):
    # Расчет параметров
    K0 = 1 - np.sin(np.radians(phi))
    M = (6 * np.sin(np.radians(phi))) / (3 - np.sin(np.radians(phi)))

    # Бытовое давление грунта (природное состояние)
    u = 0  # Поровое давление (кПа), если есть
    sigma_v = gamma * h - u
    sigma_h = K0 * sigma_v
    p_point_natural = (sigma_v + 2 * sigma_h) / 3
    q_point_natural = sigma_v - sigma_h

    # Инициализация давления упрочнения и пластической деформации
    if not modify_state:
        pc = max(pc_input, np.sqrt(p_point_natural**2 + q_point_natural**2))
        epsilon_p = calculate_epsilon_p(p_point_natural, q_point_natural, phi, c, epsilon_ref)
        st.session_state.max_pc = max(st.session_state.max_pc, pc)  # Обновляем максимальное значение pc
        st.session_state.max_epsilon_p = max(st.session_state.max_epsilon_p, epsilon_p)  # Обновляем максимальное значение epsilon_p
    else:
        # Используем максимальные значения, если слайдер уменьшается
        pc = max(pc_slider, st.session_state.max_pc)
        epsilon_p = max(epsilon_p_slider, st.session_state.max_epsilon_p)
        st.session_state.max_pc = max(st.session_state.max_pc, pc)  # Обновляем максимальное значение pc
        st.session_state.max_epsilon_p = max(st.session_state.max_epsilon_p, epsilon_p)  # Обновляем максимальное значение epsilon_p

    # Диапазон значений p (среднее эффективное напряжение)
    p_min = -c / M  # Пересечение линии Mohr-Coulomb с осью p
    p = np.linspace(p_min, 250, 50)

    # Линия Mohr-Coulomb Failure Criterion
    q_mc = M * p + c

    # Cap Hardening Surface (эллиптическая поверхность)
    p_cap = np.linspace(0, pc, 100)
    q_cap = np.sqrt(pc**2 - p_cap**2)

    # Поверхность текучести для рассчитанного epsilon_p
    q_ult = q_f(p, phi, c)
    q_yield = yield_surface(p, q_ult, epsilon_p, epsilon_ref)

    # Находим точку пересечения Cap Hardening Surface и yield_surface
    p_intersect, q_intersect = find_intersection(p_cap, q_cap, yield_surface(p_cap, q_f(p_cap, phi, c), epsilon_p, epsilon_ref))

    # Создание графика
    fig = go.Figure()

    # Закрашиваем область упругого поведения (до точки пересечения и под эллипсом)
    if p_intersect is not None:
        # Заливка до точки пересечения (под yield_surface)
        p_fill_left = np.linspace(0, p_intersect, 50)
        q_fill_left = np.minimum(yield_surface(p_fill_left, q_f(p_fill_left, phi, c), epsilon_p, epsilon_ref),
                                 np.sqrt(pc ** 2 - p_fill_left ** 2))

        # Заливка от точки пересечения до правой стороны эллипса (под Cap Hardening Surface)
        p_fill_right = np.linspace(p_intersect, pc, 50)
        q_fill_right = np.sqrt(pc ** 2 - p_fill_right ** 2)

        # Объединяем данные для заливки
        p_fill = np.concatenate([p_fill_left, p_fill_right])
        q_fill = np.concatenate([q_fill_left, q_fill_right])

        linear_elastic_group = "Область упругого поведения"

        # Добавляем заливку как один объект
        fig.add_trace(go.Scatter(
            x=p_fill,
            y=q_fill,
            fill='tozeroy',
            fillcolor='lightblue',
            legendgroup=linear_elastic_group,
            name=linear_elastic_group,
            line=dict(color='lightblue'),
            showlegend=False  # Показываем только одну запись в легенде
        ))

    # Заливка области упругого поведения до начала координат (если c > 0)
    if c > 0:
        p_fill_zero = np.linspace(p_min, 0, 50)
        q_fill_zero = yield_surface(p_fill_zero, q_f(p_fill_zero, phi, c), epsilon_p, epsilon_ref)
        fig.add_trace(go.Scatter(
            x=p_fill_zero,
            y=q_fill_zero,
            fill='tozeroy',
            fillcolor='lightblue',
            legendgroup=linear_elastic_group,
            name=linear_elastic_group,
            line=dict(color='lightblue'),
            showlegend=True  # Скрываем эту запись в легенде
        ))

    # Линии
    fig.add_trace(go.Scatter(x=p, y=q_mc, mode='lines', name='Mohr-Coulomb Failure Criterion', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=p_cap, y=q_cap, mode='lines', name='Объемная поверхность текучести', line=dict(color='navy', dash='dot')))
    fig.add_trace(go.Scatter(x=p, y=q_yield, mode='lines', name='Сдвиговая поверхность текучести', line=dict(color='blue', dash='dot')))

    # Добавление осей координат
    fig.add_shape(type="line", x0=0, y0=0, x1=250, y1=0, line=dict(color="black", width=2))  # Ось p
    fig.add_shape(type="line", x0=0, y0=0, x1=0, y1=250, line=dict(color="black", width=2))  # Ось q

    # Настройки графика
    fig.update_layout(
        xaxis_title="p (кПа)",
        yaxis_title="q (кПа)",
        xaxis_range=[p_min, 250],  # Фиксированный масштаб оси p
        yaxis_range=[0, 250],  # Фиксированный масштаб оси q
        showlegend=True
    )

    return fig

# Интерфейс Streamlit
st.title("Модель с двойным упрочнением на плоскости p-q")

# Боковая панель для ввода данных
with st.sidebar:
    st.header("Ввод данных")
    modify_state = st.checkbox("Изменить напряженное состояние")

    # Ввод данных пользователем
    gamma = st.number_input("Удельный вес грунта, γ (кН/м³):", value=18.0)
    h = st.number_input("Глубина, h (м):", value=5.0)
    c = st.number_input("Удельное сцепление, c (кПа):", value=20.0)
    phi = st.number_input("Угол внутреннего трения, φ (°):", value=20.0)
    pc_input = st.number_input("Давление предупрочнения, pc (кПа):", value=0)

    # Сохраняем исходные данные при первом включении чекбокса
    if modify_state and st.session_state.initial_data is None:
        st.session_state.initial_data = {
            "gamma": gamma,
            "h": h,
            "c": c,
            "phi": phi,
            "pc_input": pc_input
        }

    # Если чекбокс активен, используем сохраненные данные и блокируем редактирование
    if modify_state:
        gamma = st.session_state.initial_data["gamma"]
        h = st.session_state.initial_data["h"]
        c = st.session_state.initial_data["c"]
        phi = st.session_state.initial_data["phi"]
        pc_input = st.session_state.initial_data["pc_input"]

        st.write("Исходные данные (заблокированы):")
        st.write(f"Удельный вес грунта, γ (кН/м³): {gamma}")
        st.write(f"Глубина, h (м): {h}")
        st.write(f"Удельное сцепление, c (кПа): {c}")
        st.write(f"Угол внутреннего трения, φ (°): {phi}")
        st.write(f"Давление предупрочнения, pc (кПа): {pc_input}")

        # Рассчитываем начальное значение pc для слайдера
        K0 = 1 - np.sin(np.radians(phi))
        sigma_v = gamma * h
        sigma_h = K0 * sigma_v
        p_point_natural = (sigma_v + 2 * sigma_h) / 3
        q_point_natural = sigma_v - sigma_h
        initial_pc = max(pc_input, np.sqrt(p_point_natural**2 + q_point_natural**2))

        pc_slider = st.slider("Изменение объемной поверхности текучести", 0.0, 300.0, float(initial_pc))
        epsilon_p_slider = st.slider("Изменение сдвиговой поверхности текучести", 0.0, 0.3, float(st.session_state.max_epsilon_p), step=0.005)
    else:
        # Если чекбокс не активен, разрешаем редактирование
        pc_slider = pc_input
        epsilon_p_slider = 0.1

# Построение графика
fig = plot_graph(gamma, h, c, phi, pc_input, modify_state, pc_slider, epsilon_p_slider)
st.plotly_chart(fig)
