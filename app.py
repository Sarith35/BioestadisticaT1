import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

st.set_page_config(
    page_title="Calculadora de Razon de Prevalencias",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.header-box {
    background: linear-gradient(135deg, #1a5276 0%, #154360 100%);
    border-radius: 12px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    color: white;
}
.header-box h1 {
    color: white;
    font-size: 1.8rem;
    font-weight: 700;
    margin: 0 0 0.3rem 0;
}
.header-box p {
    color: rgba(255,255,255,0.82);
    font-size: 0.95rem;
    margin: 0;
}

.metric-card {
    background: #f8f9fa;
    border: 1px solid #e3e6ea;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    text-align: center;
    height: 100%;
}
.metric-card .label {
    font-size: 0.78rem;
    font-weight: 600;
    color: #6c757d;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.4rem;
}
.metric-card .value {
    font-size: 1.7rem;
    font-weight: 700;
    color: #1a5276;
    line-height: 1.1;
}
.metric-card .sub {
    font-size: 0.8rem;
    color: #868e96;
    margin-top: 0.3rem;
}

.conclusion-riesgo {
    background: #fdf2f2;
    border-left: 4px solid #c0392b;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.25rem;
    margin: 0.75rem 0;
}
.conclusion-protector {
    background: #f0fdf4;
    border-left: 4px solid #1e8449;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.25rem;
    margin: 0.75rem 0;
}
.conclusion-neutro {
    background: #f0f4ff;
    border-left: 4px solid #2e86c1;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.25rem;
    margin: 0.75rem 0;
}

.paso {
    background: #fff;
    border: 1px solid #e3e6ea;
    border-radius: 10px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 0.8rem;
}
.paso .paso-titulo {
    font-size: 0.82rem;
    font-weight: 700;
    color: #1a5276;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.5rem;
}

.tabla-2x2 table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.92rem;
}
.tabla-2x2 th {
    background: #1a5276;
    color: white;
    padding: 0.6rem 1rem;
    text-align: center;
    font-weight: 600;
}
.tabla-2x2 td {
    padding: 0.6rem 1rem;
    text-align: center;
    border-bottom: 1px solid #e3e6ea;
}
.tabla-2x2 tr:last-child td {
    background: #f0f4ff;
    font-weight: 600;
}
.tabla-2x2 td:first-child {
    text-align: left;
    font-weight: 500;
    color: #495057;
}

.advertencia {
    background: #fff8e1;
    border: 1px solid #f0c040;
    border-radius: 8px;
    padding: 0.85rem 1rem;
    font-size: 0.9rem;
    color: #7d5a00;
}

.formula-box {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 0.85rem 1.1rem;
    font-family: 'Courier New', monospace;
    font-size: 0.9rem;
    color: #212529;
    margin: 0.5rem 0;
    border: 1px solid #dee2e6;
}

.seccion-titulo {
    font-size: 1.05rem;
    font-weight: 700;
    color: #1a5276;
    border-bottom: 2px solid #d6eaf8;
    padding-bottom: 0.4rem;
    margin: 1.5rem 0 1rem 0;
}
</style>
""", unsafe_allow_html=True)


def calcular_rp(a, b, c, d, nivel=0.95):
    n_exp    = a + b
    n_no_exp = c + d

    pe = a / n_exp
    pn = c / n_no_exp
    rp = pe / pn

    z     = stats.norm.ppf(1 - (1 - nivel) / 2)
    ln_rp = np.log(rp)
    se    = np.sqrt(b / (a * n_exp) + d / (c * n_no_exp))

    ic_inf = np.exp(ln_rp - z * se)
    ic_sup = np.exp(ln_rp + z * se)

    return {
        "pe": pe, "pn": pn, "rp": rp,
        "ln_rp": ln_rp, "se": se, "z": z,
        "ic_inf": ic_inf, "ic_sup": ic_sup,
        "ic_incluye_1": ic_inf <= 1.0 <= ic_sup,
        "n_exp": n_exp, "n_no_exp": n_no_exp,
    }


def validar_datos(a, b, c, d):
    errores = []
    for nombre, v in [("a", a), ("b", b), ("c", c), ("d", d)]:
        if v < 0:
            errores.append(f"**{nombre}** no puede ser negativo.")
    if (a + b) == 0:
        errores.append("a + b = 0: no hay personas expuestas.")
    if (c + d) == 0:
        errores.append("c + d = 0: no hay personas no expuestas.")
    if a == 0:
        errores.append("**a = 0**: ningún expuesto tiene la enfermedad. RP = 0, interpretación muy limitada.")
    if c == 0:
        errores.append("**c = 0**: ningún no-expuesto tiene la enfermedad. RP no definida.")
    return errores


def forest_plot(rp, ic_inf, ic_sup, nom_exp, nom_enf):
    ic_incluye_1 = ic_inf <= 1.0 <= ic_sup
    color_pt = "#c0392b" if not ic_incluye_1 else "#2e86c1"
    color_ln = "#922b21" if not ic_incluye_1 else "#1a5276"

    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f8f9fa")

    x_min = min(ic_inf * 0.75, 0.4)
    x_max = max(ic_sup * 1.25, 2.5)

    ax.axvline(1, color="#7f8c8d", linewidth=1.4, linestyle="--", alpha=0.85,
               label="Linea de nulidad (RP = 1)")
    ax.axvspan(x_min, 1,   alpha=0.07, color="#e74c3c")
    ax.axvspan(1,   x_max, alpha=0.07, color="#27ae60")

    ax.plot([ic_inf, ic_sup], [1, 1], color=color_ln, linewidth=4,
            solid_capstyle="round", zorder=3)
    for x in [ic_inf, ic_sup]:
        ax.plot(x, 1, "|", color=color_ln, markersize=18, markeredgewidth=2.5, zorder=4)
    ax.plot(rp, 1, "o", color=color_pt, markersize=13, zorder=5)

    ax.text(rp,     1.28, f"RP = {rp:.4f}", ha="center", va="bottom",
            fontsize=10, fontweight="bold", color=color_pt)
    ax.text(ic_inf, 0.72, f"{ic_inf:.4f}", ha="center", va="top",
            fontsize=9, color=color_ln)
    ax.text(ic_sup, 0.72, f"{ic_sup:.4f}", ha="center", va="top",
            fontsize=9, color=color_ln)

    ax.text(x_min + 0.02, 1.55, "Zona protectora\n(RP < 1)",
            ha="left", va="top", fontsize=8, color="#c0392b", alpha=0.7)
    ax.text(x_max - 0.02, 1.55, "Zona de riesgo\n(RP > 1)",
            ha="right", va="top", fontsize=8, color="#1e8449", alpha=0.7)

    estado = "IC NO incluye 1 — Asociacion significativa" if not ic_incluye_1 \
             else "IC incluye 1 — No significativo"
    ax.text(0.98, 0.05, estado, transform=ax.transAxes, fontsize=8.5,
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffcc",
                      edgecolor="#aaa", alpha=0.95))

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0.5, 1.8)
    ax.set_yticks([])
    ax.set_xlabel("Razon de Prevalencias (RP)", fontsize=10)
    ax.set_title(f"Forest plot — {nom_exp} vs {nom_enf}", fontsize=10, pad=8)
    ax.legend(fontsize=8, loc="upper left")
    ax.spines[["top", "left", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def barras_prevalencias(pe, pn, nom_exp):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f8f9fa")

    grupos  = [f"Expuestos\n({nom_exp})", "No expuestos"]
    valores = [pe * 100, pn * 100]
    colores = ["#c0392b", "#2e86c1"]

    bars = ax.bar(grupos, valores, color=colores, width=0.5,
                  edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, valores):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}%",
                ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_ylabel("Prevalencia (%)", fontsize=10)
    ax.set_title("Comparacion de prevalencias", fontsize=10, pad=8)
    ax.set_ylim(0, max(valores) * 1.35)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


DEFAULTS = dict(inp_a=80, inp_b=120, inp_c=40, inp_d=260,
                inp_exp="Tabaquismo", inp_enf="Bronquitis cronica", inp_nivel=0.95)

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

EJEMPLOS = {
    "Tabaquismo y bronquitis":       dict(inp_a=80,  inp_b=120, inp_c=40,  inp_d=260,
                                          inp_exp="Tabaquismo",        inp_enf="Bronquitis cronica"),
    "Sedentarismo y diabetes":       dict(inp_a=150, inp_b=100, inp_c=50,  inp_d=300,
                                          inp_exp="Sedentarismo",      inp_enf="Diabetes tipo 2"),
    "Lactancia y otitis (protector)":dict(inp_a=30,  inp_b=170, inp_c=60,  inp_d=140,
                                          inp_exp="Lactancia materna", inp_enf="Otitis media"),
    "Sin asociacion (cafe/HTA)":     dict(inp_a=45,  inp_b=155, inp_c=40,  inp_d=160,
                                          inp_exp="Consumo de cafe",   inp_enf="Hipertension"),
}

with st.sidebar:
    st.markdown("### Ejercicios de practica")
    st.caption("Carga un ejemplo para ver como funciona el calculo.")

    for nombre_ej, vals in EJEMPLOS.items():
        if st.button(nombre_ej, use_container_width=True):
            for k, v in vals.items():
                st.session_state[k] = v

    st.markdown("---")

    st.markdown("### Datos del estudio")
    st.caption("Ingresa los valores de tu tabla 2x2.")

    st.text_input("Exposicion", key="inp_exp")
    st.text_input("Enfermedad", key="inp_enf")

    st.markdown("**Tabla 2x2**")

    col1, col2 = st.columns(2)
    with col1:
        st.number_input("a  (exp. CON)", min_value=0, step=1, key="inp_a")
        st.number_input("c  (no exp. CON)", min_value=0, step=1, key="inp_c")
    with col2:
        st.number_input("b  (exp. SIN)", min_value=0, step=1, key="inp_b")
        st.number_input("d  (no exp. SIN)", min_value=0, step=1, key="inp_d")

    st.markdown("---")

    st.markdown("### Nivel de confianza")
    st.select_slider(
        "Selecciona",
        options=[0.90, 0.95, 0.99],
        format_func=lambda x: f"{int(x*100)}%",
        key="inp_nivel",
    )

a       = int(st.session_state["inp_a"])
b       = int(st.session_state["inp_b"])
c       = int(st.session_state["inp_c"])
d       = int(st.session_state["inp_d"])
nom_exp = st.session_state["inp_exp"].strip() or "Exposicion"
nom_enf = st.session_state["inp_enf"].strip() or "Enfermedad"
nivel   = st.session_state["inp_nivel"]

st.markdown("""
<div class="header-box">
  <h1>Razon de Prevalencias con Intervalo de Confianza</h1>
  <p>Software pedagogico — Bioestadistica — Estudios transversales — Tabla 2x2</p>
</div>
""", unsafe_allow_html=True)

errores = validar_datos(a, b, c, d)

if errores:
    st.markdown('<div class="advertencia">Revisa los datos antes de continuar:<br>' +
                "<br>".join(f"• {e}" for e in errores) + "</div>", unsafe_allow_html=True)
    st.stop()

res = calcular_rp(a, b, c, d, nivel)
pe, pn, rp       = res["pe"], res["pn"], res["rp"]
ic_inf, ic_sup   = res["ic_inf"], res["ic_sup"]
ic1              = res["ic_incluye_1"]
ln_rp, se, z     = res["ln_rp"], res["se"], res["z"]

st.markdown('<div class="seccion-titulo">1. Tabla 2x2 del estudio</div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="tabla-2x2">
<table>
  <thead>
    <tr>
      <th></th>
      <th>Con {nom_enf}</th>
      <th>Sin {nom_enf}</th>
      <th>Total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Expuestos ({nom_exp})</td>
      <td><strong>a = {a}</strong></td>
      <td><strong>b = {b}</strong></td>
      <td>{a+b}</td>
    </tr>
    <tr>
      <td>No expuestos</td>
      <td><strong>c = {c}</strong></td>
      <td><strong>d = {d}</strong></td>
      <td>{c+d}</td>
    </tr>
    <tr>
      <td>Total</td>
      <td>{a+c}</td>
      <td>{b+d}</td>
      <td><strong>{a+b+c+d}</strong></td>
    </tr>
  </tbody>
</table>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="seccion-titulo">2. Calculo paso a paso</div>', unsafe_allow_html=True)

col_l, col_r = st.columns([1, 1], gap="large")

with col_l:
    st.markdown(f"""
    <div class="paso">
      <div class="paso-titulo">Paso 1 — Prevalencia en expuestos</div>
      <div class="formula-box">P_exp = a / (a+b) = {a} / {a+b} = <strong>{pe:.4f}</strong> ({pe*100:.2f}%)</div>
      De cada 100 {nom_exp.lower()}, {pe*100:.1f} tienen {nom_enf.lower()}.
    </div>
    <div class="paso">
      <div class="paso-titulo">Paso 2 — Prevalencia en no expuestos</div>
      <div class="formula-box">P_no = c / (c+d) = {c} / {c+d} = <strong>{pn:.4f}</strong> ({pn*100:.2f}%)</div>
      De cada 100 no expuestos, {pn*100:.1f} tienen {nom_enf.lower()}.
    </div>
    <div class="paso">
      <div class="paso-titulo">Paso 3 — Razon de Prevalencias</div>
      <div class="formula-box">RP = P_exp / P_no = {pe:.4f} / {pn:.4f} = <strong>{rp:.4f}</strong></div>
      Los expuestos tienen {rp:.2f}x la prevalencia de los no expuestos.
    </div>
    """, unsafe_allow_html=True)

with col_r:
    term1       = f"{b/(a*(a+b)):.6f}"
    term2       = f"{d/(c*(c+d)):.6f}"
    exp_inf     = f"{ln_rp - z*se:.4f}"
    exp_sup     = f"{ln_rp + z*se:.4f}"

    paso5 = (
        f"SE = sqrt[ b/(a·n_exp) + d/(c·n_no) ]<br>"
        f"SE = sqrt[ {b}/{a}·{a+b} + {d}/{c}·{c+d} ]<br>"
        f"SE = sqrt[ {term1} + {term2} ]<br>"
        f"SE = <strong>{se:.4f}</strong>"
    )
    paso6 = (
        f"z = {z:.4f}<br>"
        f"IC_inf = e^(ln(RP) - z·SE) = e^{exp_inf} = <strong>{ic_inf:.4f}</strong><br>"
        f"IC_sup = e^(ln(RP) + z·SE) = e^{exp_sup} = <strong>{ic_sup:.4f}</strong>"
    )

    st.markdown(f"""
    <div class="paso">
      <div class="paso-titulo">Paso 4 — Logaritmo de la RP</div>
      <div class="formula-box">ln(RP) = ln({rp:.4f}) = <strong>{ln_rp:.4f}</strong></div>
      Se usa escala logaritmica porque ln(RP) se distribuye aproximadamente normal, lo que permite construir el IC de forma estable.
    </div>
    <div class="paso">
      <div class="paso-titulo">Paso 5 — Error estandar del ln(RP)</div>
      <div class="formula-box">{paso5}</div>
    </div>
    <div class="paso">
      <div class="paso-titulo">Paso 6 — Intervalo de Confianza al {int(nivel*100)}%</div>
      <div class="formula-box">{paso6}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="seccion-titulo">3. Resultados</div>', unsafe_allow_html=True)

m1, m2, m3, m4, m5 = st.columns(5)
metricas = [
    (m1, "Prev. expuestos",            f"{pe*100:.2f}%",   f"{a} / {a+b} personas"),
    (m2, "Prev. no expuestos",         f"{pn*100:.2f}%",   f"{c} / {c+d} personas"),
    (m3, "RP estimada",                f"{rp:.4f}",         "Punto estimado"),
    (m4, f"IC {int(nivel*100)}% inf.", f"{ic_inf:.4f}",    "Limite inferior"),
    (m5, f"IC {int(nivel*100)}% sup.", f"{ic_sup:.4f}",    "Limite superior"),
]
for col, label, val, sub in metricas:
    with col:
        st.markdown(f"""
        <div class="metric-card">
          <div class="label">{label}</div>
          <div class="value">{val}</div>
          <div class="sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<div class="seccion-titulo">4. Interpretacion epidemiologica</div>', unsafe_allow_html=True)

ic_tag = "SI incluye el 1" if ic1 else "NO incluye el 1"

if ic1:
    cls    = "conclusion-neutro"
    titulo = "Sin asociacion estadisticamente significativa"
    texto  = (
        f"El IC al {int(nivel*100)}% {ic_tag} ({ic_inf:.4f} – {ic_sup:.4f}). "
        f"Aunque la RP observada fue {rp:.4f}, el rango de valores plausibles incluye 1, "
        f"lo que indica que la diferencia podria deberse al azar. "
        f"No hay evidencia suficiente de asociacion entre **{nom_exp}** y **{nom_enf}**."
    )
elif rp > 1:
    cls    = "conclusion-riesgo"
    titulo = "Asociacion significativa — Factor de riesgo"
    texto  = (
        f"El IC al {int(nivel*100)}% {ic_tag} y esta completamente por encima de 1 "
        f"({ic_inf:.4f} – {ic_sup:.4f}). "
        f"Los expuestos a **{nom_exp}** tienen {rp:.2f} veces la prevalencia de **{nom_enf}** "
        f"en comparacion con los no expuestos. "
        f"La exposicion actua como **factor de riesgo** en esta poblacion."
    )
else:
    cls    = "conclusion-protector"
    titulo = "Asociacion significativa — Factor protector"
    texto  = (
        f"El IC al {int(nivel*100)}% {ic_tag} y esta completamente por debajo de 1 "
        f"({ic_inf:.4f} – {ic_sup:.4f}). "
        f"Los expuestos a **{nom_exp}** tienen {rp:.2f} veces la prevalencia de **{nom_enf}** "
        f"en comparacion con los no expuestos. "
        f"La exposicion actua como **factor protector** en esta poblacion."
    )

st.markdown(f"""
<div class="{cls}">
  <strong>{titulo}</strong><br><br>
  {texto}
</div>
""", unsafe_allow_html=True)

st.info(
    "Una asociacion estadistica no implica causalidad. "
    "Para afirmar que la exposicion causa la enfermedad se necesita evidencia adicional "
    "(criterios de Bradford Hill).",
    icon=None
)

st.markdown('<div class="seccion-titulo">5. Visualizaciones</div>', unsafe_allow_html=True)

g1, g2 = st.columns([3, 2], gap="large")

with g1:
    st.pyplot(forest_plot(rp, ic_inf, ic_sup, nom_exp, nom_enf))
    st.caption(
        "El punto central es la RP estimada. Si la barra no cruza la linea punteada (RP = 1), "
        "la asociacion es estadisticamente significativa."
    )

with g2:
    st.pyplot(barras_prevalencias(pe, pn, nom_exp))
    st.caption("Comparacion directa de la prevalencia en cada grupo.")

st.markdown('<div class="seccion-titulo">6. Tabla resumen</div>', unsafe_allow_html=True)

resumen = pd.DataFrame({
    "Indicador": [
        "Total participantes", "Expuestos", "No expuestos",
        f"Prevalencia en expuestos ({nom_exp})",
        "Prevalencia en no expuestos",
        "Razon de Prevalencias (RP)",
        f"IC {int(nivel*100)}% — Limite inferior",
        f"IC {int(nivel*100)}% — Limite superior",
        "IC incluye el 1?",
        "Conclusion estadistica",
    ],
    "Resultado": [
        f"{a+b+c+d} personas",
        f"{a+b} personas",
        f"{c+d} personas",
        f"{pe:.4f}  ({pe*100:.2f}%)",
        f"{pn:.4f}  ({pn*100:.2f}%)",
        f"{rp:.4f}",
        f"{ic_inf:.4f}",
        f"{ic_sup:.4f}",
        "SI" if ic1 else "NO",
        "No significativa" if ic1 else (
            f"Significativa — Factor de riesgo (RP = {rp:.4f})" if rp > 1
            else f"Significativa — Factor protector (RP = {rp:.4f})"
        ),
    ],
})
st.dataframe(resumen, use_container_width=True, hide_index=True)

with st.expander("Limitaciones del software"):
    st.markdown("""
    Este metodo **no aplica** en los siguientes casos:

    - Si algun valor de la tabla es 0, la RP no esta definida.
    - Muestra muy pequeña: el IC sera amplio y poco informativo.
    - Estudios de cohorte o casos y controles: usar **Odds Ratio (OR)**, no RP.
    - Si hay variables de confusion: se necesita analisis estratificado o regresion de Poisson.

    **Supuestos del metodo de Katz:**
    - Todos los conteos a, b, c, d > 0
    - Muestra suficientemente grande (aprox. >30 por grupo)
    """)

st.markdown("---")
st.caption(
    "Software pedagogico para Bioestadistica — "
    "Tema 6: Razon de Prevalencias con IC — "
    "Metodo: Katz et al."
)
