import os
import numpy
import pandas
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.naive_bayes import GaussianNB

st.title("Clasificador Gaussiano")

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

    data = data.dropna()

    st.dataframe(data)

    st.subheader("Parametros:")
    st.markdown("#### Variable Independiente:")
    variable_independiente = st.selectbox("Elige una opcion", data.keys(), key="variableObjetivo")

    array_independiente = numpy.array(data[variable_independiente]).reshape(-1,1)
    
    st.markdown("#### Variable Dependiente:")
    variable_dependiente = st.selectbox("Elige una opcion", data.keys(), key="variableDependiente")
    #variable_dependiente = st.text_input(f"Ingrese las {len(array_independiente)} variables dependientes separadas por coma")
    #2,0,2,1,1,1,1,1,1,1,1,0,0,0,0,0,0

    array_dependiente = numpy.array(data[variable_dependiente]).reshape(-1,1)

    st.markdown("#### Clasificacion:")
    valor_prediccion = st.number_input("Ingrese el valor a clasificar:",None,None,0,1)

    col_1, col_2, col_3= st.columns(3)
    columna_eje = col_1.checkbox("Usar columna como eje X")
    
    if columna_eje:
        columna_eje_x = col_2.selectbox("Elija la columna que se usara como eje X", data.keys(), key="columnaEje")
        valor_eje_x = col_3.number_input("Ingrese el valor en el eje X del valor a clasificar",None,None,0,1)

    if st.button('Calcular'):
        #if str(variable_dependiente).strip() is not "":
            #array_dependiente = variable_dependiente.split(',')
            #array_dependiente = list(map(str.strip, array_dependiente))
            #array_dependiente = list(map(int, array_dependiente))
            #array_dependiente = numpy.array(array_dependiente).reshape(-1,1)

            classifier = GaussianNB()
            classifier.fit(array_independiente, array_dependiente)

            prediccion = classifier.predict(numpy.array(valor_prediccion).reshape(-1,1))

            grafica = plt.figure()

            plt.style.use("fivethirtyeight")

            clases = numpy.unique(array_dependiente)

            colores_clases = dict()

            if not columna_eje:

                for clase in clases: 
                    color = numpy.random.rand(3,)
                    colores_clases[clase] = color
                    plt.scatter(array_dependiente[array_dependiente == clase], array_independiente[array_dependiente == clase],
                                marker='o',color=color,label=f'Clase {clase}')

                plt.xlabel('Clases')

                punto_x = prediccion[0]
                punto_y = valor_prediccion

                plt.plot(punto_x, punto_y, marker='D',color=colores_clases[punto_x])
                #plt.scatter(array_independiente, array_dependiente)
                #plt.plot(array_independiente, prediccion_y)
                #plt.xticks(range(clases[0],clases[-1]+1))

            else:
                datos_columna_eje = numpy.array(data[columna_eje_x]).reshape(-1,1)

                for clase in clases: 
                    color = numpy.random.rand(3,)
                    colores_clases[clase] = color
                    plt.scatter(datos_columna_eje[array_dependiente == clase], array_independiente[array_dependiente == clase],
                                marker='o',color=color,label=f'Clase {clase}')

                plt.xlabel(columna_eje_x)

                punto_x = valor_eje_x
                punto_y = valor_prediccion

                plt.plot(punto_x, punto_y, marker='D',color=colores_clases[prediccion[0]])

                plt.locator_params(axis='x', integer=True, tight=False)

            plt.ylabel(variable_independiente)
            plt.legend(prop={'size':10})
            grafica.text(0.17, -0.1, "El diamante representa el valor clasificado", ha='center', size='x-small')
            plt.title("Clasificador Gaussiano")
            st.subheader("Grafica:")
            st.pyplot(grafica)

            st.subheader("Clasificacion:")
            st.metric(f"El valor {valor_prediccion}, pertenece a la clase:",prediccion[0])
