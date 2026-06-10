# Calculadora de Razón de Prevalencias — Streamlit

Software pedagógico para Bioestadística.  
Tema 6: Razón de Prevalencias (RP) con Intervalo de Confianza e interpretación epidemiológica.

---

## Estructura del proyecto

```
rp_streamlit/
├── app.py              ← aplicación principal
├── requirements.txt    ← librerías necesarias
└── README.md           ← este archivo
```

---

## Cómo ejecutar localmente

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Lanzar la app
streamlit run app.py
```

Se abre automáticamente en el navegador en `http://localhost:8501`

---

## Cómo publicar en Streamlit Community Cloud (gratis)

1. Subir la carpeta a un repositorio de GitHub
2. Ir a [share.streamlit.io](https://share.streamlit.io)
3. Conectar el repositorio
4. Seleccionar `app.py` como archivo principal
5. Clic en **Deploy**

---

## Qué hace la aplicación

- Recibe los 4 valores de la tabla 2×2: `a`, `b`, `c`, `d`
- Valida que los datos sean correctos antes de calcular
- Muestra el cálculo paso a paso con las fórmulas explícitas
- Calcula la RP, el IC al 90/95/99% y genera interpretación automática
- Muestra un forest plot y gráfica comparativa de prevalencias
- Incluye 4 ejercicios de práctica cargables con un clic
- Explica cuándo NO aplica el método

---

## Librerías

| Librería     | Uso |
|-------------|-----|
| streamlit   | Interfaz web |
| numpy       | Cálculos matemáticos |
| pandas      | Tabla resumen |
| matplotlib  | Gráficas |
| scipy       | Valor z del IC |

---

*Bioestadística — Trabajo individual*
