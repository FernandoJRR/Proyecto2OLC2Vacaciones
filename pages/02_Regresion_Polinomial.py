import os
import numpy
import pandas
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

def get_operator(valor: float) -> str:
    operador = "+ " if valor>=0 else "" 
    return operador

st.title("Regresion Polinomial")

st.subheader("Carga del Archivo")

archivo = st.file_uploader("Elegir archivo:", type=['csv','xls','xlsx','json'])

if archivo is not None:
    file_name = os.path.splitext(archivo.name)

    nombre = file_name[0]
    extension = file_name[1]

    if extension == ".csv":
        data = pandas.read_csv(archivo)
    elif extension == ".xls" or extension == ".xlsx":
        data = pandas.read_excel(archivo, engine='openpyxl')
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

    column1, column2 = st.columns(2)
    with column1:
        st.markdown("#### Grado de la Función")
        grado = st.slider("Elija el Grado de la Función", 2, 5, 2, 1)

    st.markdown("#### Prediccion:")
    valor_prediccion = st.number_input("Ingrese el valor al cual predecir:",None,None,0,1)

    data = data.sort_values(by=var_x)

    #CAMBIAR DESPUES DE EXAMEN
    array_x = numpy.asarray(data[var_x]).reshape(-1,1)
    array_y = data[var_y]

    shape = array_x.shape

    #array_x = numpy.sort(array_x,0)

    if st.button("Calcular"):
        modelo_polinomial = PolynomialFeatures(degree=grado)
        features_polinomial = modelo_polinomial.fit_transform(array_x .reshape(shape))

        modelo_regresion = LinearRegression()
        modelo_regresion.fit(features_polinomial, array_y)

        prediccion_y = modelo_regresion.predict(features_polinomial)

        error_medio = numpy.sqrt(mean_squared_error(array_y, prediccion_y))
        r2 = r2_score(array_y, prediccion_y)
        error_cuadrado = mean_squared_error(array_y, prediccion_y)


        array_x_prediccion = numpy.array([[valor_prediccion]])
        predicction_transf = modelo_polinomial.fit_transform(array_x_prediccion)
        prediccion = modelo_regresion.predict(predicction_transf)

        grafica = plt.figure()

        plt.style.use("fivethirtyeight")
        plt.scatter(array_x, array_y)
        plt.plot(array_x, prediccion_y)
        plt.title("Regresion Polinomial")
        plt.ylabel(var_y)
        plt.xlabel(var_x)


        st.subheader("Grafica:")
        st.pyplot(grafica)

        st.markdown("#### Datos de la Gráfica")

        col_1, col_2= st.columns(2)
        col_1.write("Coeficientes de la Función")
        col_1.write(modelo_regresion.coef_)
        
        intercepto = float(modelo_regresion.intercept_)
        indicadorIntercepto = "+ Positivo" if intercepto>=0 else "- Negativo"
        col_2.metric("Intercepto", intercepto, indicadorIntercepto)
        
        col_3, col_4 = st.columns(2)
        col_3.metric("Coeficiente de Determinación",r2)
        col_4.metric("Error Cuadrático Medio", error_medio)

        st.subheader("Función de Tendencia:")
        
        if(grado == 2):
            b1 = float(modelo_regresion.coef_[1])
            b2 = float(modelo_regresion.coef_[2])
            
            st.latex(f"f(x)={b2}X^2 {get_operator(b1)} {b1}X {get_operator(intercepto)}{intercepto}")
        elif(grado == 3):
            b1 = float(modelo_regresion.coef_[1])
            b2 = float(modelo_regresion.coef_[2])
            b3 = float(modelo_regresion.coef_[3])
            
            st.latex(f"f(x)= {b3}X^3 {get_operator(b2)}{b2}X^2 {get_operator(b1)} {b1}X {get_operator(intercepto)}{intercepto}")

        elif(grado == 4):
            b1 = float(modelo_regresion.coef_[1])
            b2 = float(modelo_regresion.coef_[2])
            b3 = float(modelo_regresion.coef_[3])
            b4 = float(modelo_regresion.coef_[4])
            
            st.latex(f"f(x)= {b4}X^4 {get_operator(b3)}{b3}X^3 {get_operator(b2)}{b2}X^2")
            st.latex(f"{get_operator(b1)} {b1}X {get_operator(intercepto)}{intercepto}")

        elif(grado == 5):
            b1 = float(modelo_regresion.coef_[1])
            b2 = float(modelo_regresion.coef_[2])
            b3 = float(modelo_regresion.coef_[3])
            b4 = float(modelo_regresion.coef_[4])
            b5 = float(modelo_regresion.coef_[5])
            
            st.latex(f"f(x)= {b5}X^5 {get_operator(b4)}{b4}X^4 {get_operator(b3)}{b3}X^3")
            st.latex(f"{get_operator(b2)}{b2}X^2 {get_operator(b1)} {b1}X {get_operator(intercepto)}{intercepto}")

        st.subheader("Predicción")
        label_prediccion = "+ Positiva" if prediccion>=0 else "- Negativa"
        st.metric(f"La predicción para {valor_prediccion} es: ",prediccion, label_prediccion)
