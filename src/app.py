# your code here
import streamlit as st
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, date
import plotly.express as px
import plotly.graph_objects as go
import os

# Configuración de la página
st.set_page_config(
    page_title="Predictor de Ventas",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Función para cargar el modelo con rutas flexibles
@st.cache_resource
def load_model():
    try:
        # Detectar si estamos en Streamlit Cloud o local
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Posibles rutas para los modelos
        possible_paths = [
            os.path.join(current_dir, '..', 'models'),  # Local: src/../models
            os.path.join(os.getcwd(), 'models'),        # Cloud: ./models
            'models',                                    # Directa
            '../models'                                  # Relativa
        ]
        
        model_file = 'xgboost_optimizado_prediccion_ventas_model_42.pkl'
        scaler_file = 'scaler_xgboost_model_42.pkl'
        selector_file = 'selector_xgboost_model_42.pkl'
        
        models_path = None
        
        # Buscar la ruta correcta
        for path in possible_paths:
            if os.path.exists(os.path.join(path, model_file)):
                models_path = path
                break
        
        if models_path is None:
            st.error("No se encontró la carpeta models. Estructura actual:")
            st.error(f"Directorio actual: {os.getcwd()}")
            st.error(f"Contenido: {os.listdir('.')}")
            return None, None, None
        
        # Cargar archivos
        with open(os.path.join(models_path, model_file), 'rb') as file:
            model = pickle.load(file)
        
        with open(os.path.join(models_path, scaler_file), 'rb') as file:
            scaler = pickle.load(file)
        
        with open(os.path.join(models_path, selector_file), 'rb') as file:
            selector = pickle.load(file)
        
        return model, scaler, selector
        
    except Exception as e:
        st.error(f"Error cargando el modelo: {e}")
        st.error(f"Directorio de trabajo: {os.getcwd()}")
        st.error(f"Archivos disponibles: {os.listdir('.')}")
        return None, None, None

# Función para crear características (ajustada para coincidir con el entrenamiento)
def create_features(afluencia, fecha):
    features = {}
    
    # Características básicas (exactamente como en el entrenamiento)
    features['afluencia'] = afluencia
    features['mes'] = fecha.month
    features['dia_semana'] = fecha.weekday()
    features['trimestre'] = (fecha.month - 1) // 3 + 1
    features['es_fin_semana'] = 1 if fecha.weekday() >= 5 else 0
    features['es_inicio_mes'] = 1 if fecha.day <= 7 else 0
    features['es_fin_mes'] = 1 if fecha.day >= 24 else 0
    
    # Transformaciones de afluencia (exactamente como en el entrenamiento)
    features['afluencia_log'] = np.log1p(afluencia)
    features['afluencia_sqrt'] = np.sqrt(afluencia)
    
    # Características simuladas (exactamente como en el entrenamiento)
    features['afluencia_ma_7'] = afluencia * 0.95  # Simulación de media móvil
    features['afluencia_ma_30'] = afluencia * 0.98
    features['afluencia_std_7'] = afluencia * 0.1
    features['afluencia_lag_1'] = afluencia * 0.9
    features['afluencia_lag_7'] = afluencia * 0.85
    features['diferencia_ma_7'] = afluencia - features['afluencia_ma_7']
    
    return features

# Título principal
st.title("📈 Predictor de Ventas - Plaza Comercial")
st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Configuración")

# Cargar modelo
model, scaler, selector = load_model()

if model is None:
    st.error("No se pudo cargar el modelo. Verifica que los archivos estén en la carpeta models/")
    st.stop()

st.sidebar.success("✅ Modelo cargado exitosamente")

# Inputs del usuario
st.sidebar.subheader("📊 Datos de Entrada")

# Input de afluencia
afluencia = st.sidebar.number_input(
    "Afluencia de visitantes",
    min_value=0,
    max_value=100000,
    value=15000,
    step=500,
    help="Número de visitantes esperados"
)

# Input de fecha
fecha = st.sidebar.date_input(
    "Fecha de predicción",
    value=datetime.now().date(),
    min_value=date(2024, 1, 1),
    max_value=date(2026, 12, 31)
)

# Botón de predicción
if st.sidebar.button("🚀 Realizar Predicción", type="primary"):
    
    try:
        # Crear características
        features_dict = create_features(afluencia, fecha)
        
        # Lista de características en el orden correcto (según el entrenamiento)
        feature_columns = [
            'afluencia', 'mes', 'dia_semana', 'trimestre', 'es_fin_semana',
            'es_inicio_mes', 'es_fin_mes', 'afluencia_log', 'afluencia_sqrt',
            'afluencia_ma_7', 'afluencia_ma_30', 'afluencia_std_7',
            'afluencia_lag_1', 'afluencia_lag_7', 'diferencia_ma_7'
        ]
        
        # Convertir a DataFrame con el orden correcto
        features_df = pd.DataFrame([features_dict])[feature_columns]
        
        # Aplicar escalado
        features_scaled = scaler.transform(features_df)
        
        # Aplicar selección de características
        features_selected = selector.transform(features_scaled)
        
        # Realizar predicción
        prediccion = model.predict(features_selected)[0]
        
        # Almacenar en session_state
        st.session_state.prediccion = prediccion
        st.session_state.afluencia = afluencia
        st.session_state.fecha = fecha
        st.session_state.features = features_dict
        
    except Exception as e:
        st.error(f"Error en la predicción: {e}")
        st.sidebar.error("Error procesando los datos")

# Mostrar resultados si hay predicción
if hasattr(st.session_state, 'prediccion'):
    
    # Layout principal
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎯 Resultado de la Predicción")
        
        # Métrica principal
        st.metric(
            label="Ventas Netas Predichas",
            value=f"${st.session_state.prediccion:,.2f}",
            delta=f"Afluencia: {st.session_state.afluencia:,} visitantes"
        )
        
        # Información adicional
        st.info(f"""
        **Fecha de predicción:** {st.session_state.fecha.strftime('%d/%m/%Y')}  
        **Día de la semana:** {'Fin de semana' if st.session_state.features['es_fin_semana'] else 'Entre semana'}  
        **Trimestre:** Q{st.session_state.features['trimestre']}  
        **Mes:** {st.session_state.fecha.strftime('%B')}
        """)
    
    with col2:
        st.subheader("📊 Análisis")
        
        # Categorización de la predicción
        if st.session_state.prediccion > 50000:
            categoria = "🟢 Alta"
        elif st.session_state.prediccion > 25000:
            categoria = "🟡 Media"
        else:
            categoria = "🔴 Baja"
        
        st.markdown(f"**Categoría de ventas:** {categoria}")
    
    # Gráficos
    st.markdown("---")
    st.subheader("📈 Visualizaciones")
    
    col3, col4 = st.columns(2)
    
    with col3:
        # Gráfico de barras - Comparación con rangos
        rangos = ['Ventas Bajas\n(< $25K)', 'Ventas Medias\n($25K - $50K)', 'Ventas Altas\n(> $50K)']
        valores_rangos = [20000, 37500, 60000]
        colores = ['red' if st.session_state.prediccion < 25000 else 'lightcoral',
                  'orange' if 25000 <= st.session_state.prediccion <= 50000 else 'lightsalmon',
                  'green' if st.session_state.prediccion > 50000 else 'lightgreen']
        
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=rangos,
            y=valores_rangos,
            name='Rangos de referencia',
            marker_color=colores,
            opacity=0.7
        ))
        
        fig_bar.add_trace(go.Scatter(
            x=[f'Predicción\n${st.session_state.prediccion:,.0f}'],
            y=[st.session_state.prediccion],
            mode='markers',
            marker=dict(size=20, color='darkblue'),
            name='Predicción actual'
        ))
        
        fig_bar.update_layout(
            title="Comparación con Rangos de Ventas",
            yaxis_title="Ventas ($)",
            height=400
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col4:
        # Gráfico de gauge
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = st.session_state.prediccion,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Ventas Predichas ($)"},
            delta = {'reference': 37500},
            gauge = {
                'axis': {'range': [None, 75000]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 25000], 'color': "lightgray"},
                    {'range': [25000, 50000], 'color': "gray"},
                    {'range': [50000, 75000], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 60000
                }
            }
        ))
        
        fig_gauge.update_layout(height=400)
        st.plotly_chart(fig_gauge, use_container_width=True)

# Sección de análisis de sensibilidad
st.markdown("---")
st.subheader("🔄 Análisis de Sensibilidad")

if st.checkbox("Mostrar análisis de sensibilidad"):
    try:
        # Crear rango de afluencias
        afluencia_base = st.session_state.afluencia if hasattr(st.session_state, 'afluencia') else 15000
        fecha_base = st.session_state.fecha if hasattr(st.session_state, 'fecha') else datetime.now().date()
        
        afluencias = np.arange(5000, 30000, 1000)
        predicciones = []
        
        # Lista de características en el orden correcto
        feature_columns = [
            'afluencia', 'mes', 'dia_semana', 'trimestre', 'es_fin_semana',
            'es_inicio_mes', 'es_fin_mes', 'afluencia_log', 'afluencia_sqrt',
            'afluencia_ma_7', 'afluencia_ma_30', 'afluencia_std_7',
            'afluencia_lag_1', 'afluencia_lag_7', 'diferencia_ma_7'
        ]
        
        for afl in afluencias:
            features_dict = create_features(afl, fecha_base)
            features_df = pd.DataFrame([features_dict])[feature_columns]
            features_scaled = scaler.transform(features_df)
            features_selected = selector.transform(features_scaled)
            pred = model.predict(features_selected)[0]
            predicciones.append(pred)
        
        # Crear DataFrame para el gráfico
        df_sensibilidad = pd.DataFrame({
            'Afluencia': afluencias,
            'Ventas_Predichas': predicciones
        })
        
        # Gráfico de línea
        fig_sens = px.line(
            df_sensibilidad, 
            x='Afluencia', 
            y='Ventas_Predichas',
            title='Sensibilidad de Ventas vs Afluencia',
            labels={'Afluencia': 'Afluencia de Visitantes', 'Ventas_Predichas': 'Ventas Predichas ($)'}
        )
        
        fig_sens.add_vline(
            x=afluencia_base, 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"Afluencia actual: {afluencia_base:,}"
        )
        
        st.plotly_chart(fig_sens, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error en análisis de sensibilidad: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>🤖 Modelo XGBoost Optimizado</strong> | Predicción de Ventas en Plazas Comerciales</p>
    <p><em>Desarrollado con Streamlit y XGBoost</em></p>
</div>
""", unsafe_allow_html=True)

# Información de debug
st.sidebar.info("💡 Aplicación funcionando correctamente")

# Debug info para desarrollo
if st.sidebar.checkbox("Mostrar información técnica"):
    st.sidebar.write(f"**Directorio actual:** {os.getcwd()}")
    st.sidebar.write(f"**Archivos disponibles:** {os.listdir('.')}")
    if os.path.exists('models'):
        st.sidebar.write(f"**Modelos encontrados:** {os.listdir('models')}")