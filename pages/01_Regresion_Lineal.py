import os
import numpy
import pandas
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.title("Regresion Lineal")

st.subheader("Carga del Archivo")

archivo = st.file_uploader("Elegir archivo:", type=['csv','xls','xlsx','json'])

if archivo is not None:
    file_name = os.path.splitext(archivo.name)

    nombre = file_name[0]
    extension = file_name[1]

    if extension == ".csv":
        data = pandas.read_csv(archivo)
    elif extension == ".xls" or extension == ".xlsx":
        data = pandas.read_excel(archivo)
    elif extension == ".json":
        data = pandas.read_json(archivo)
    else:
        data = pandas.read_csv(archivo)
    
    st.markdown("#### Data del Archivo:")

    st.dataframe(data)

    st.subheader("Parametros:")

    st.markdown("#### Variable Independiente (X)")
    var_x = st.selectbox("Elegir una entrada:", data.keys(), key="variableX")

    st.markdown("#### Variable Dependiente (Y)")
    var_y = st.selectbox("Elegir una entrada:", data.keys(), key="variableY")

    st.markdown("#### Prediccion:")
    valor_prediccion = st.number_input("Ingrese el valor al cual predecir:",None,None,0,1)

    array_x = numpy.asarray(data[var_x]).reshape(-1,1)
    array_y = data[var_y]

    if st.button("Calcular"):
        modelo_regresion = LinearRegression()
        modelo_regresion.fit(array_x, array_y)

        prediccion_y = modelo_regresion.predict(array_x)

        r2 = r2_score(array_y, prediccion_y)
        error_cuadrado = mean_squared_error(array_y, prediccion_y)

        prediccion = modelo_regresion.predict([[valor_prediccion]])

        grafica = plt.figure()

        plt.style.use("fivethirtyeight")
        plt.scatter(array_x, array_y)
        plt.plot(array_x, prediccion_y)
        plt.title("Regresion Lineal")
        plt.ylabel(var_y)
        plt.xlabel(var_x)

        st.subheader("Grafica:")
        st.pyplot(grafica)

        st.markdown("#### Datos Graficados:")

        col_1, col_2 = st.columns(2)

        pendiente = float(modelo_regresion.coef_)

        indicadorPendiente = "+ Positiva" if pendiente>=0 else "- Negativa" 
        col_1.metric("Pendiente", pendiente, indicadorPendiente)
        
        intercepto = float(modelo_regresion.intercept_)
        indicadorIntercepto = "+ Positivo" if intercepto>=0 else "- Negativo"
        col_2.metric("Intercepto", intercepto, indicadorIntercepto)
        
        col_3, col_4 = st.columns(2)
        col_3.metric("Coeficiente de Determinación", r2)
        col_4.metric("Error Cuadrático",error_cuadrado)
        
        st.subheader("Función de Tendencia")
        
        operador = "+ " if intercepto>=0 else "" 
        st.latex(f"f(x)={pendiente}X {operador}{intercepto}")
        st.subheader("Predicción")
        indicadorPrediccion = "+ Positiva" if prediccion>=0 else "- Negativa"
        st.metric(f"El valor de la predicción para {valor_prediccion} es de: ",prediccion, indicadorPrediccion)
