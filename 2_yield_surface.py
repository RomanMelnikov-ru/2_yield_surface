import numpy as np
import streamlit as st
import plotly.graph_objects as go

# Константы
EPSILON_REF = 0.1  # Параметр для нелинейного закона

# Инициализация состояния сессии
if "initial_data" not in st.session_state:
    st.session_state.initial_data = None
if "modified_pc" not in st.session_state:
    st.session_state.modified_pc = None
if "modified_epsilon_p" not in st.session_state:
    st.session_state.modified_epsilon_p = None


# Функции для расчетов
def q_f(p, phi, c):
    M = (6 * np.sin(np.radians(phi))) / (3 - np.sin(np.radians(phi)))
    return M * p + c


def yield_surface(p, q_ult, epsilon_p, epsilon_ref):
    return q_ult * (1 - np.exp(-epsilon_p / epsilon_ref))


def calculate_epsilon_p(p_natural, q_natural, phi, c, epsilon_ref):
    q_ult = q_f(p_natural, phi, c)
    if q_natural >= q_ult:
        return np.inf
    return -epsilon_ref * np.log(1 - q_natural / q_ult)


def calculate_natural_state(gamma, h, phi):
    K0 = 1 - np.sin(np.radians(phi))
    sigma_v = gamma * h
    sigma_h = K0 * sigma_v
    p = (sigma_v + 2 * sigma_h) / 3
    q = sigma_v - sigma_h
    return p, q


def calculate_initial_pc(p_natural, q_natural, pc_input):
    return max(pc_input, np.sqrt(p_natural ** 2 + q_natural ** 2))


# Интерфейс Streamlit
st.title("Модель с двойным упрочнением на плоскости p-q")

# Боковая панель для ввода данных
with st.sidebar:
    st.header("Управление")
    modify_state = st.checkbox("Изменить напряженное состояние")

    # Ввод исходных данных (показывается только когда чекбокс неактивен)
    if not modify_state:
        st.header("Исходные параметры")
        gamma = st.number_input("Удельный вес грунта, γ (кН/м³):", value=18.0, min_value=0.1)
        h = st.number_input("Глубина, h (м):", value=5.0, min_value=0.1)
        c = st.number_input("Удельное сцепление, c (кПа):", value=20.0, min_value=0.0)
        phi = st.number_input("Угол внутреннего трения, φ (°):", value=20.0, min_value=0.0, max_value=45.0)
        pc_input = st.number_input("Давление предуплотнения, pc (кПа):", value=0.0, min_value=0.0)

        # Рассчитываем природное состояние
        p_natural, q_natural = calculate_natural_state(gamma, h, phi)
        initial_pc = calculate_initial_pc(p_natural, q_natural, pc_input)
        initial_epsilon_p = calculate_epsilon_p(p_natural, q_natural, phi, c, EPSILON_REF)

        # Сохраняем текущие параметры как исходные
        st.session_state.initial_data = {
            "gamma": gamma,
            "h": h,
            "c": c,
            "phi": phi,
            "pc_input": pc_input,
            "initial_pc": initial_pc,
            "initial_epsilon_p": initial_epsilon_p
        }

        # Сбрасываем модифицированные параметры
        st.session_state.modified_pc = None
        st.session_state.modified_epsilon_p = None
    else:
        # Используем сохраненные исходные данные
        if st.session_state.initial_data is None:
            st.warning("Сначала введите исходные параметры")
            st.stop()

        data = st.session_state.initial_data
        st.header("Текущие параметры")
        st.write(f"Удельный вес: {data['gamma']} кН/м³")
        st.write(f"Глубина: {data['h']} м")
        st.write(f"Удельное сцепление: {data['c']} кПа")
        st.write(f"Угол внутреннего трения: {data['phi']}°")
        st.write(f"Давление предуплотнения: {data['pc_input']} кПа")

        # Управление упругой областью
        st.header("Изменение напряженного состояния")

        # Инициализация модифицированных параметров
        if st.session_state.modified_pc is None:
            st.session_state.modified_pc = data['initial_pc']
        if st.session_state.modified_epsilon_p is None:
            st.session_state.modified_epsilon_p = data['initial_epsilon_p']

        # Слайдеры для управления (сразу изменяют значения)
        st.session_state.modified_pc = st.slider(
            "Среднее напряжение р (кПа)",
            min_value=float(data['initial_pc']),
            max_value=300.0,
            value=float(st.session_state.modified_pc),
            step=0.1
        )

        st.session_state.modified_epsilon_p = st.slider(
            "Сдвиговая деформация (εₚ)",
            min_value=float(data['initial_epsilon_p']),
            max_value=0.20,
            value=float(st.session_state.modified_epsilon_p),
            step=0.001
        )


# Функция построения графика
def plot_yield_surfaces(gamma, h, c, phi, pc_input, modify_state):
    # Получаем актуальные данные
    if modify_state:
        data = st.session_state.initial_data
        pc = st.session_state.modified_pc
        epsilon_p = st.session_state.modified_epsilon_p
    else:
        p_natural, q_natural = calculate_natural_state(gamma, h, phi)
        pc = calculate_initial_pc(p_natural, q_natural, pc_input)
        epsilon_p = calculate_epsilon_p(p_natural, q_natural, phi, c, EPSILON_REF)
        data = st.session_state.initial_data

    # Расчет параметров
    M = (6 * np.sin(np.radians(phi))) / (3 - np.sin(np.radians(phi)))
    p_min = -c / M if c > 0 else 0
    p = np.linspace(p_min, 250, 100)

    # Линия разрушения Мора-Кулона
    q_mc = M * p + c

    # Cap Hardening Surface (эллиптическая поверхность)
    p_cap = np.linspace(0, pc, 100)
    q_cap = np.sqrt(pc ** 2 - p_cap ** 2)

    # Поверхность текучести
    q_yield = yield_surface(p, q_f(p, phi, c), epsilon_p, EPSILON_REF)

    # Создание графика
    fig = go.Figure()

    # Область упругого поведения (включая отрицательные значения p)
    if pc > 0:
        # Положительная часть p
        p_fill_pos = np.linspace(0, pc, 100)
        q_fill_upper_pos = np.minimum(
            yield_surface(p_fill_pos, q_f(p_fill_pos, phi, c), epsilon_p, EPSILON_REF),
            np.sqrt(pc ** 2 - p_fill_pos ** 2)
        )

        # Отрицательная часть p (если есть)
        if p_min < 0:
            p_fill_neg = np.linspace(p_min, 0, 100)
            q_fill_upper_neg = yield_surface(p_fill_neg, q_f(p_fill_neg, phi, c), epsilon_p, EPSILON_REF)

            # Объединяем области
            p_fill = np.concatenate([p_fill_neg, p_fill_pos])
            q_fill_upper = np.concatenate([q_fill_upper_neg, q_fill_upper_pos])
        else:
            p_fill = p_fill_pos
            q_fill_upper = q_fill_upper_pos

        fig.add_trace(go.Scatter(
            x=p_fill, y=q_fill_upper,
            fill='tozeroy',
            fillcolor='rgba(173, 216, 230, 0.5)',
            line=dict(color='rgba(0,0,0,0)'),
            name="Область упругого поведения",
            showlegend=True
        ))

    # Линии на графике
    fig.add_trace(go.Scatter(x=p, y=q_mc, mode='lines', name='Линия прочности Мора-Кулона', line=dict(color='red')))
    fig.add_trace(
        go.Scatter(x=p_cap, y=q_cap, mode='lines', name='Объемная поверхность текучести', line=dict(color='navy', dash='dot')))
    fig.add_trace(
        go.Scatter(x=p, y=q_yield, mode='lines', name='Сдвиговая поверхность текучести', line=dict(color='blue', dash='dot')))

    # Оси и оформление
    fig.update_layout(
        xaxis_title="p (кПа)",
        yaxis_title="q (кПа)",
        xaxis_range=[p_min - 5 if p_min < 0 else 0, 250],
        yaxis_range=[-5, 250],
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.add_shape(type="line", x0=0, y0=0, x1=250, y1=0, line=dict(color="black", width=2))
    fig.add_shape(type="line", x0=0, y0=0, x1=0, y1=250, line=dict(color="black", width=2))

    return fig


# Построение графика
if st.session_state.initial_data is not None or not modify_state:
    if modify_state:
        data = st.session_state.initial_data
        fig = plot_yield_surfaces(
            data['gamma'], data['h'], data['c'], data['phi'],
            data['pc_input'], modify_state
        )
    else:
        fig = plot_yield_surfaces(gamma, h, c, phi, pc_input, modify_state)
    st.plotly_chart(fig, use_container_width=True)
