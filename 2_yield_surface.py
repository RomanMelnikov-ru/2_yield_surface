import numpy as np
import streamlit as st
import plotly.graph_objects as go


# Инициализация состояния сессии
def init_session_state():
    session_vars = {
        'p_point_natural': None,
        'q_point_natural': None,
        'epsilon_p': None,
        'pc': None,
        'max_pc': None,
        'max_epsilon_p': None,
        'gamma': 18.0,
        'h': 5.0,
        'c': 20.0,
        'phi': 20.0,
        'pc_input': 0.0,
        'modify_state': False,
        'show_sliders': False
    }
    for key, value in session_vars.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()
epsilon_ref = 0.1  # Параметр для нелинейного закона


# Расчетные функции
def q_f(p, phi, c):
    M = (6 * np.sin(np.radians(phi))) / (3 - np.sin(np.radians(phi))) if phi != 0 else 0
    return M * p + c


def yield_surface(p, q_ult, epsilon_p, epsilon_ref):
    return q_ult * (1 - np.exp(-epsilon_p / epsilon_ref)) if epsilon_p != 0 else 0


def calculate_epsilon_p(p_natural, q_natural, phi, c, epsilon_ref):
    if c == 0 and phi == 0:
        return 0
    q_ult = q_f(p_natural, phi, c)
    if q_natural >= q_ult:
        return np.inf
    return -epsilon_ref * np.log(1 - q_natural / q_ult) if q_ult != 0 else 0


def find_intersection(p, q_cap, q_yield):
    for i in range(len(p) - 1):
        if (q_cap[i] - q_yield[i]) * (q_cap[i + 1] - q_yield[i + 1]) <= 0:
            return p[i], q_cap[i]
    return None, None


# Основная функция построения графика
def plot_hardening_soil():
    try:
        # Получаем параметры
        gamma = st.session_state.gamma
        h = st.session_state.h
        c = st.session_state.c
        phi = st.session_state.phi
        pc_input = st.session_state.pc_input

        # Расчет параметров
        K0 = 1 - np.sin(np.radians(phi)) if phi != 0 else 1.0
        M = (6 * np.sin(np.radians(phi))) / (3 - np.sin(np.radians(phi))) if phi != 0 else 0

        # Бытовое давление грунта
        sigma_v = gamma * h
        sigma_h = K0 * sigma_v
        st.session_state.p_point_natural = (sigma_v + 2 * sigma_h) / 3
        st.session_state.q_point_natural = sigma_v - sigma_h

        # Инициализация параметров
        if not st.session_state.modify_state:
            st.session_state.pc = max(pc_input, np.sqrt(
                st.session_state.p_point_natural ** 2 + st.session_state.q_point_natural ** 2))
            st.session_state.epsilon_p = calculate_epsilon_p(
                st.session_state.p_point_natural,
                st.session_state.q_point_natural,
                phi, c, epsilon_ref
            )
            st.session_state.max_pc = st.session_state.pc
            st.session_state.max_epsilon_p = st.session_state.epsilon_p
            st.session_state.slider_pc = st.session_state.pc
            st.session_state.slider_epsilon_p = st.session_state.epsilon_p

        # Диапазон значений p с учетом отрицательной области при c > 0
        p_min = -c / M if M != 0 and c > 0 else 0
        p_max = max(250, st.session_state.pc * 1.2) if st.session_state.pc > 0 else 250
        p = np.linspace(p_min, p_max, 200)
        q_mc = q_f(p, phi, c)

        # Cap Hardening Surface
        p_cap = np.linspace(0, st.session_state.pc, 100) if st.session_state.pc > 0 else np.array([0])
        q_cap = np.sqrt(st.session_state.pc ** 2 - p_cap ** 2) if st.session_state.pc > 0 else np.array([0])

        # Поверхность текучести
        q_ult = q_f(p, phi, c)
        q_yield = yield_surface(p, q_ult, st.session_state.epsilon_p, epsilon_ref)

        # Точка пересечения
        p_intersect, q_intersect = find_intersection(
            p_cap, q_cap,
            yield_surface(p_cap, q_f(p_cap, phi, c), st.session_state.epsilon_p, epsilon_ref)
        ) if st.session_state.pc > 0 else (None, None)

        # Создание графика
        fig = go.Figure()

        # Добавление осей координат
        fig.add_shape(type="line", x0=min(p_min, 0), y0=0, x1=p_max, y1=0, line=dict(color="black", width=1))
        fig.add_shape(type="line", x0=0, y0=0, x1=0, y1=max(200, np.max(q_mc) * 1.1), line=dict(color="black", width=1))

        # Закрашивание всей упругой области
        if st.session_state.pc > 0:
            # Область под эллипсом и поверхностью текучести
            p_fill = np.linspace(0, st.session_state.pc, 100)
            q_cap_fill = np.sqrt(st.session_state.pc ** 2 - p_fill ** 2)
            q_yield_fill = yield_surface(p_fill, q_f(p_fill, phi, c), st.session_state.epsilon_p, epsilon_ref)
            q_fill_upper = np.minimum(q_cap_fill, q_yield_fill)

            fig.add_trace(go.Scatter(
                x=np.concatenate([p_fill, p_fill[::-1]]),
                y=np.concatenate([q_fill_upper, np.zeros_like(p_fill)[::-1]]),
                fill='toself',
                fillcolor='rgba(173, 216, 230, 0.5)',
                line=dict(color='rgba(173, 216, 230, 0.5)'),
                name="Упругая область"
            ))

        # Область в отрицательных p при c > 0
        if c > 0 and M != 0:
            p_fill_neg = np.linspace(p_min, 0, 100)
            q_fill_neg = yield_surface(p_fill_neg, q_f(p_fill_neg, phi, c), st.session_state.epsilon_p, epsilon_ref)

            fig.add_trace(go.Scatter(
                x=np.concatenate([p_fill_neg, p_fill_neg[::-1]]),
                y=np.concatenate([q_fill_neg, np.zeros_like(p_fill_neg)[::-1]]),
                fill='toself',
                fillcolor='rgba(173, 216, 230, 0.5)',
                line=dict(color='rgba(173, 216, 230, 0.5)'),
                showlegend=False
            ))

        # Линии графика
        fig.add_trace(go.Scatter(
            x=p, y=q_mc,
            mode='lines',
            name="Mohr-Coulomb Failure Criterion",
            line=dict(color='red', width=2)
        ))

        if st.session_state.pc > 0:
            fig.add_trace(go.Scatter(
                x=p_cap, y=q_cap,
                mode='lines',
                name="Объемная поверхность текучести",
                line=dict(color='blue', dash='dot', width=2)
            ))

        fig.add_trace(go.Scatter(
            x=p, y=q_yield,
            mode='lines',
            name="Сдвиговая поверхность текучести",
            line=dict(color='green', dash='dot', width=2)
        ))

        # Точки на графике
        #fig.add_trace(go.Scatter(
        #    x=[st.session_state.p_point_natural],
        #    y=[st.session_state.q_point_natural],
        #    mode='markers',
        #    marker=dict(color='black', size=10),
        #    name=f"Природное состояние (p={st.session_state.p_point_natural:.1f}, q={st.session_state.q_point_natural:.1f})"
        #))

        # Настройки графика
        fig.update_layout(
            xaxis_title="p (кПа)",
            yaxis_title="q (кПа)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(range=[min(p_min, -10), p_max]),
            yaxis=dict(range=[0, max(200, np.max(q_mc) * 1.1)]),
            height=600,
            margin=dict(l=50, r=50, t=80, b=50),
            plot_bgcolor='white'
        )

        # Равный масштаб осей
        fig.update_yaxes(
            scaleanchor="x",
            scaleratio=1,
            constrain="domain"
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Ошибка при построении графика: {str(e)}")


# Функции обновления
def update_pc():
    new_pc = st.session_state.slider_pc
    if new_pc > st.session_state.max_pc:
        st.session_state.pc = new_pc
        st.session_state.max_pc = new_pc


def update_epsilon_p():
    new_epsilon_p = st.session_state.slider_epsilon_p
    if new_epsilon_p > st.session_state.max_epsilon_p:
        st.session_state.epsilon_p = new_epsilon_p
        st.session_state.max_epsilon_p = new_epsilon_p


# Интерфейс приложения
st.set_page_config(layout="wide", page_title="Двойное упрочнение в модели Hardening Soil")

# Сайдбар для ввода параметров
with st.sidebar:
    st.header("Параметры грунта")

    st.number_input(
        "Удельный вес грунта, γ (кН/м³):",
        min_value=0.1,
        max_value=20.0,
        value=st.session_state.gamma,
        step=0.1,
        key="gamma"
    )

    st.number_input(
        "Глубина, h (м):",
        min_value=0.1,
        max_value=10.0,
        value=st.session_state.h,
        step=0.5,
        key="h"
    )

    st.number_input(
        "Сцепление, c (кПа):",
        min_value=0.0,
        max_value=40.0,
        value=st.session_state.c,
        step=1.0,
        key="c"
    )

    st.number_input(
        "Угол внутреннего трения, φ (°):",
        min_value=0.0,
        max_value=40.0,
        value=st.session_state.phi,
        step=1.0,
        key="phi"
    )

    st.number_input(
        "Давление предупрочнения, pc (кПа):",
        min_value=0.0,
        max_value=300.0,
        value=st.session_state.pc_input,
        step=10.0,
        key="pc_input"
    )

    st.checkbox(
        "Изменить напряженное состояние",
        value=st.session_state.modify_state,
        key="modify_state",
        on_change=lambda: st.session_state.update({
            'show_sliders': st.session_state.modify_state
        })
    )

    if st.session_state.show_sliders:
        st.slider(
            "Изменение объемной поверхности текучести",
            min_value=0.0,
            max_value=300.0,
            value=st.session_state.get('slider_pc', st.session_state.pc_input),
            step=1.0,
            key="slider_pc",
            on_change=update_pc
        )

        st.slider(
            "Изменение сдвиговой поверхности текучести",
            min_value=0.0,
            max_value=0.35,
            value=st.session_state.get('slider_epsilon_p', 0.1),
            step=0.005,
            key="slider_epsilon_p",
            on_change=update_epsilon_p
        )

# Основная область с графиком
st.title("Двойное упрочнение в модели Hardening Soil")
plot_hardening_soil()
