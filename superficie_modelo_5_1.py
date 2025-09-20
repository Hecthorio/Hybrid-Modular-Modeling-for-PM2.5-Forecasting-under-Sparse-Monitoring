# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 18:42:33 2025

@author: hecto
"""

"""
Script para evaluar la concentración superficial de PM2.5 a partir de un modelo
de balance de materia (difusión-convección) discretizado por la tecnica clasica 
de diferencias finitas

#en este script evaluamos para el día indicado los promedios horarios con respec-
to a una base de monitoreo

En esta versión v5.1 se modifica el tipo de mapa de basemap de matplotlib a 
contextily, además de que este script genera una malla (grilla) con diferentes
mapas a lo largo del día en lugar de hacer "refresh" a la figura (animación)
"""


#librerias
import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.basemap import Basemap
import contextily as ctx
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

#definimos el día del mes que se va a analizar 1-31
dia_mes = 15 #15,2

#definimos de que estacion vamos a tomar los datos
estacion = 'SMA'

#1ro vamos a definir la ruta de los multiples archivos de donde esta la velocidad
#del viento y la concentración de PM2.5 en la estaciones de monitoreo
ruta = 'C:/Users/hecto/OneDrive/Documentos/ITA/Posdoc Proyecto/articulo 3/DatosPM25/'

#aquí definimos en el diccionario la base datos de PM2.5 de la estación y la base de
#datos de RUOA que le corresponde de ese periodo de tiempo
base_datos = {'CBT':['Datos SINAICA - CBTIS - PM2.5 - 2022-06-01 - 2022-07-01.csv', '2022-06-agsc_minuto_L1.csv'],
              'CEN':['Datos SINAICA - Centro - PM2.5 - 2022-06-01 - 2022-07-01.csv', '2022-06-agsc_minuto_L1.csv'],
              'IED':['Datos SINAICA - Instituto Educativo - PM2.5 - 2022-12-01 - 2023-01-01.csv', '2022-12-agsc_minuto_L1.csv'],
              'SMA':['Datos SINAICA - Secretaría de Medio Ambiente - PM2.5 - 2023-02-01 - 2023-03-01.csv', '2023-02-agsc_minuto_L1.csv']}

#cargamos el df con la info de de la velocidad y dirección del viento
df = pd.read_csv(ruta + base_datos[estacion][1], encoding='latin1', skiprows=7)


#generamos un dataframe independiente para generar la base de datos que se 
#utiliza para evaluar la altura de la mezcla
df_alt = df*1

#ahora eliminamos las columnas que no nos sirvan para el analsis (T, humedad, etc)
df.drop(columns = ['°C', '%', 'm/s.1', 'deg.1', 'mm', 'hPa', 'W/m^2'], inplace = True)

#eliminamos las variables que no nos sirvan para la evaluación de h
df_alt.drop(columns = ['%', 'm/s.1', 'deg','deg.1', 'mm', 'hPa'], inplace = True)

#convertimos a variable tipo fecha la 1ra columna
#df['yyyy-mm-dd HH:MM:SS'] = pd.to_datetime(df['yyyy-mm-dd HH:MM:SS'], dayfirst = True)
#df_alt['yyyy-mm-dd HH:MM:SS'] = pd.to_datetime(df_alt['yyyy-mm-dd HH:MM:SS'], dayfirst = True)
df['yyyy-mm-dd HH:MM:SS'] = pd.to_datetime(df['yyyy-mm-dd HH:MM:SS'], format='mixed', dayfirst = True)
df_alt['yyyy-mm-dd HH:MM:SS'] = pd.to_datetime(df_alt['yyyy-mm-dd HH:MM:SS'], format='mixed', dayfirst = True)

#filtramso el día seleccionado
df = df[df['yyyy-mm-dd HH:MM:SS'].dt.day == dia_mes]
df_alt = df_alt[df_alt['yyyy-mm-dd HH:MM:SS'].dt.day == dia_mes]

#evaluamos que día de la semana es ese día y que día del año
dia = df['yyyy-mm-dd HH:MM:SS'].dt.day_of_week.iloc[0]
doy = df['yyyy-mm-dd HH:MM:SS'].dt.dayofyear.iloc[0]

#guardamos la fecha
fecha = df['yyyy-mm-dd HH:MM:SS'].iloc[0]
fecha = fecha.strftime('%Y-%m-%d')

#eliminamos la columna de fecha
df.drop(columns = ['yyyy-mm-dd HH:MM:SS'], inplace = True)

# #leemos el archivo que tiene la información de la velocidad y dirección del viento
# df = pd.read_csv('viento.csv')

#cargamos el modelo de red neuronal de emisiones a la memoria
red = load_model('modelo_emisiones.h5')

#convertimos la velocidad del viento en m/h
vo = df['m/s'].to_numpy()*3600
do = df['deg'].to_numpy()

#convertimos el df en un arreglo de numpy
df_alt = np.array(df_alt)

#madamos llamar la función
import modulo_alt_mezcla as mam

#evaluamos la altura de la mezcla a partir de la función
h_mezcla = mam.altura_mezcla(df_alt)

#la pasamos a arreglo de numpy
h_mezcla = np.array(h_mezcla)

#los valores menores a cero los hacemos iguales a 150
h_mezcla[h_mezcla < 150] = 150

#definimos cuantos valores (nodos usaremos).
#recordar que los nodos de las orillas representan las fronteras
nodos = [24, 24]

###############################################################################
#ESTOS BLOQUES DE CODIGO SON PARA DEFINIR LA CONDICIÓN INICIAL (CI)
#leemos el df con los datos
df_con_i = pd.read_csv(ruta + base_datos[estacion][0], encoding='latin1')
#df = pd.read_csv(ruta + base_datos[estacion][0])

#eliminamos la 1ra fila del df
df_con_i.drop(index = 0, inplace = True)

#convertimos a formato de fecha
df_con_i.Fecha = pd.to_datetime(df_con_i.Fecha)

#filtramos solmente los datos del día que nos interesa
df_con_i = df_con_i[df_con_i.Fecha.dt.day == dia_mes]
###############################################################################

#definimos los incrementos en tiempo y espacio
#OJO: RECORDAR QUE LAS COORDENADAS QUE SE USARON PARA EL ANALSIS DE ESTA
#SUPERFICIE SON DE 12X12 KM, POR LO QUE HAY QUE HACER COINCIDIR EL PRODUCTO
#DE EL NUMERO DE NODOS CON EL DE LOS INCREMENTOS (dh)
dh = 500                    #incremento espacio (m)
dt = 1/60                    #incremento tiempo (h)
Dab = 0.001/(100**2)*3600     #difusividad (m2/h)
h = 1500                        #altura de la mezcla (m)
Coi = df_con_i['Concentraciones horarias'].iloc[0]                    #concentración inicial simulación (ug/m3)
del(df_con_i)                #eliminamos este df porque solos se utilizó para sacar la CI
#voi = 2000                  #velocidad del viento inicial (m/h)
#el día que se va a simular (este día modifica las emisiones del modelo de red)

#definimos un modelo en el que pasamos la hora del día y el día de la semana 
#y lo convertimos a variables continuas a partir del seno y coseno
def tiempo(hora, dia):
    x = np.zeros((1,4))
    x[0,0] = np.sin(hora*2*np.pi/24)
    x[0,1] = np.cos(hora*2*np.pi/24)
    x[0,2] = np.sin((dia + hora/24)*2*np.pi/7)
    x[0,3] = np.cos((dia + hora/24)*2*np.pi/7)
    return x

#definimos las matrices concentración anterior, actual, velocidad del viento y 
#dirección anterior, actual y de emisiones
Co = np.full(nodos,Coi)
Ct = np.full(nodos,Coi)
#vo = np.full(nodos, voi)
#do = np.random.randint(250,310,nodos)
#do = np.full(nodos, 150)

#evaluamos las componente de la velocidad del viento
#vox = vo*np.cos(np.deg2rad(do))
#voy = vo*np.sin(np.deg2rad(do))

#aumento de grados para poder agregar a las fuentes de área en el mapa
#la relación "promedio" es 1°=111km
inc = 1/111

#definimos los limites de nuestro sistema
lat_lim = [21.79818125851826-inc, 21.978493676558536+inc]
lon_lim = [-102.37665302280578-inc, -102.1857655833361+inc]

#definimos las coordenadas de las fuentes fijas (lat, lon) y factor de emisión
#para eso leemos el archiv con la info de las fuentes fijas
df_funtes_fijas = pd.read_csv('fuentes_fijas_bueno.csv')

#Nissan, marelli, leche san marcos, ciudad industrial (sur)
# coord_lat_lon = np.array([[21.80412826718264, -102.28116638903427],
#                           [21.972041519143712, -102.27840185880999],
#                           [21.939049806696044, -102.2911846631989],
#                           [21.844704703966205, -102.2784224890615]])
#definimos el arreglo con solo la información del la latitud, longitud y emisiones
coord_lat_lon = np.array(df_funtes_fijas[['Latitud','Longitud','emision']])
del(df_funtes_fijas)

#aplicamos el factor de conversión de Ton/año a ug/h
coord_lat_lon[:,2] = coord_lat_lon[:,2]*1000*1000*1e6/(365*24)

#filtramos las coordenadas que cumplan con los limites establecidos en el sistema
coord_lat_lon = coord_lat_lon[(coord_lat_lon[:,0] >= lat_lim[0]) & (coord_lat_lon[:,0] <= lat_lim[1]) & 
                              ((coord_lat_lon[:,1] >= lon_lim[0]) & (coord_lat_lon[:,1] <= lon_lim[1]))]

#convertimos las coordenadas en distancia en función de los nodos y la distancia (dh)
coord_dist_lat = 0 + (nodos[0]*dh - 0)/(lat_lim[1] - lat_lim[0])*(coord_lat_lon[:,0] - lat_lim[0])
coord_dist_lon = 0 + (nodos[1]*dh - 0)/(lon_lim[1] - lon_lim[0])*(coord_lat_lon[:,1] - lon_lim[0])

#unimos los arreglos para dejar las coordenadas de cada una
coord_dist = np.column_stack((coord_dist_lat, coord_dist_lon))
del(coord_dist_lat, coord_dist_lon)

#definimos una matriz de flujos de emisión
q_fijas = np.full(nodos,0.0)
for i in range(len(coord_lat_lon)):
    lat = round(coord_dist[i,0]/dh)
    lon = round(coord_dist[i,1]/dh)
    if lat == nodos[0]:
        lat = nodos[0]-1
    #q_fijas[lat,lon] = 38089041095.8904/(dh*dh)/4   #flujo de emisión (ug/h.m2)
    q_fijas[lat,lon] = coord_lat_lon[i,2]/(dh*dh)   #flujo de emisión (ug/h.m2)

del(lat,lon)

#definimos unas coordenadas de lat y lon para las fuentes moviles
coord_lat_lon_movil = np.array([[21.879143498711024, -102.29194896088771],
                          [21.87807035002179, -102.27147995936701],
                          [21.89792229389972, -102.27980633286695],
                          [21.900390180244592, -102.29287411349881],
                          [21.897278490434662, -102.30524802967233],
                          [21.873884993015444, -102.31183974202645],
                          [21.867767711983678, -102.30906428419313],
                          [21.865299261195698, -102.29426184241547],
                          [21.857971284779822, -102.29293329471007],
                          [21.871533917681187, -102.31880449905438],
                          [21.915328103091166, -102.29195215044295],
                          [21.870623392569424, -102.25435886342134],
                          [21.844828857930185, -102.29081336376252]])

#definimos las coordenadas donde estan cada una de las estaciones de monitoreo
#la unica utilidad de esto es graficar los puntos en el mapa dinamico del sistema
#las coordenadas corresponden a: CEN, CBT, IED, SMA
coord_lat_lon_monitoreo = np.array([[21.8833514833953, -102.29533997273451],
                                    [21.873119629257992, -102.31966634999102],
                                    [21.901582851543484, -102.27749237648966],
                                    [21.845033733629954, -102.29156999067133]])

#convertimos las coordenadas en distancia en función de los nodos y la distancia (dh)
coord_dist_lat_movil = 0 + (nodos[0]*dh - 0)/(lat_lim[1] - lat_lim[0])*(coord_lat_lon_movil[:,0] - lat_lim[0])
coord_dist_lon_movil = 0 + (nodos[1]*dh - 0)/(lon_lim[1] - lon_lim[0])*(coord_lat_lon_movil[:,1] - lon_lim[0])

#unimos los arreglos para dejar las coordenadas de cada una
coord_dist_movil = np.column_stack((coord_dist_lat_movil, coord_dist_lon_movil))
del(coord_dist_lat_movil, coord_dist_lon_movil)


#definimos esta parte como la función de las emisiones
def emisiones(coord_lat_lon_movil, coord_dist_movil, red, hora, dia, q_fijas, dh, q_area):
    #evaluamos las emisiones con la red, covertimos ton a ug, lo dividimos entre en area superficial
    #y el numero de nodos 
    #OJO! en PROAIRE se comenta que la "Distribución del parque vehicular al 2016 por municipio"
    #en aguascalientes es del 75% por eso multiplicamos por ese factor
    emisiones = red(tiempo(hora,dia)).numpy()[0][0]*1000*1000*1e6/(dh*dh)/len(coord_lat_lon_movil)
    
    #para que no se sobreescriba el valor de q_fijas
    q_movil = (q_area + q_fijas)*1
    
    #en este ciclo posicionamos en función del tamaño de malla la posición de cada
    #punto donde se emiten las fuentes moviles
    for i in range(len(coord_lat_lon_movil)):
        lat = round(coord_dist_movil[i,0]/dh)
        lon = round(coord_dist_movil[i,1]/dh)
        if lat == 10:
            lat = 9
        if lon == 10:
            lon =9
        #si se repite el valor sobre el mismo nodo se suma
        if q_movil[lat,lon] > 0:
            q_movil[lat,lon] = emisiones + q_movil[lat,lon]  #flujo de emisión (ug/h.m2)
        else:
            q_movil[lat,lon] = emisiones
    
    del(lat,lon)
    
    return q_movil

#%%
#EN ESTA SECCIÓN GENERAMOS LA MATRIZ DE FLUJO DE EMISIÓN DE FUENTES DE ÁREA
#definimos la ruta donde se encuntra el archivo con la ubicación de las fuentes de área en Ags (ladrilleras)
ruta_fue_area = 'C:/Users/hecto/OneDrive/Documentos/ITA/Posdoc Proyecto/articulo 3/INEGI_DENUE/INEGI_DENUE_26022025.csv' 

#leemos el archivo
df_fue_area = pd.read_csv(ruta_fue_area, encoding='latin-1')

#nos quedamos con las columnas que nos interesan (ubicación)
df_fue_area = df_fue_area[['Latitud', 'Longitud']]

#ahora vamos a filtrar todos aquellos puntos que se encuentren por afuera de los limites del sistema.
df_fue_area = df_fue_area[(df_fue_area.Latitud > lat_lim[0]) & (df_fue_area.Latitud < lat_lim[1]) &
                          (df_fue_area.Longitud > lon_lim[0]) & (df_fue_area.Longitud < lon_lim[1])]

#lo convertimos en un areaglo de numpy el df y despues lo eliminamos
coord_lat_lon_area = np.array(df_fue_area)
del(df_fue_area)

#convertimos las coordenadas a distancias
#convertimos las coordenadas en distancia en función de los nodos y la distancia (dh)
coord_dist_lat_area = 0 + (nodos[0]*dh - 0)/(lat_lim[1] - lat_lim[0])*(coord_lat_lon_area[:,0] - lat_lim[0])
coord_dist_lon_area = 0 + (nodos[1]*dh - 0)/(lon_lim[1] - lon_lim[0])*(coord_lat_lon_area[:,1] - lon_lim[0])

#unimos los arreglos para dejar las coordenadas de cada una
coord_dist_area = np.column_stack((coord_dist_lat_area, coord_dist_lon_area))
del(coord_dist_lat_area, coord_dist_lon_area)

#definimos una matriz de flujos de emisión de fuentes de área (ladrilleras)
q_area = np.full(nodos,0)
num_ladris = 350
emisiones_year = 1080.85     #Ton/año
for i in range(len(coord_dist_area)):
    lat = round(coord_dist_area[i,0]/dh)
    lon = round(coord_dist_area[i,1]/dh)
    if lat == nodos[0]:
        lat = nodos[0]-1
    #si cae sobre el mismo nodo que se sumen las emisiones (aquellas ladrilleras que caen sobre el mismo cuadrante del nodo)
    if q_area[lat,lon] > 0:
        q_area[lat,lon] = q_area[lat,lon] + emisiones_year/(dh*dh)/num_ladris/(365*24)*(1000*1000*1e6)   #flujo de emisión (ug/h.m2)
    else:
        q_area[lat,lon] = emisiones_year/(dh*dh)/num_ladris/(365*24)*(1000*1000*1e6)   #flujo de emisión (ug/h.m2)

del(lat,lon,num_ladris,emisiones_year)

#%%

#definimos la función de velodidad de sedimentación
u = 0.0004/100*3600         #velocidad de sedimentación (m/h)
#u = 1.01

#definimos la función discretizada
#OJO!!! los indices para definir una posición en python correponden a [filas,columnas]
#Las filas se definen por cambios en "y" y las columnas cambios en "x",
#por lo que si se utilizan los subindices i y j, siendo i=x y j=y quedaria [j,i]
def con(i,j,Co,dt,dh,Dab,vox,voy,u,h,q):
    C = Co[j,i] + dt/(2*dh)*(-vox[j,i]*(Co[j,i+1] - Co[j,i-1]) - voy[j,i]*(Co[j+1,i] - Co[j-1,i]) \
                             -Co[j,i]*(vox[j,i+1] - vox[j,i-1] + voy[j+1,i] - voy[j-1,i]) + \
                             2*dh/h*(q[j,i] - Co[j,i]*u))
                             #(q[j,i] - Co[j,i]*u)/h)
    
    #filtro
    C = 0.9*C + 0.1/8*(Co[j+1,i-1] + Co[j+1,i] + Co[j+1,i+1] + Co[j,i-1] + Co[j,i+1] + Co[j-1,i-1] + Co[j-1,i] + Co[j-1,i+1]) 
    
    #agregamos una restricci si tenemos concentraciones negativas
    if C < 0:
        #C = (Co[j+1,i] + Co[j-1,i] + Co[j,i+1] + Co[j,i-1])/4
        C = (Co[j+1,i] + Co[j-1,i] + Co[j,i+1] + Co[j,i-1] + Co[j+1,i+1] + Co[j-1,i-1] + Co[j+1,i-1] + Co[j-1,i+1])/8
        #print('NEG')
        #C = Coi
        #C = 0.0
    else:
        print('OK')
    #if C > 90:
    #    C = 90
    return C

#para la fontera superior
# def con_sup(i,j,Co,dt,dh,Dab,vox,voy,u,h,q):
#     a = Co[j,i]*(1 - dt*(1/(2*dh)*(voy[j,i+1] - voy[j,i-1]) + 4*Dab/(dh**2) + u/h))
#     b = dt*(Co[j,i+1]*(Dab/(dh**2) - vox[j,i]/(2*dh)) + Co[j,i-1]*(Dab/(dh**2) + vox[j,i]/(2*dh)))
#     C = a + b + dt*q[j,i]/h
#     #agregamos una restricci si tenemos concentraciones negativas
#     if C < 0:
#         C = (Co[j,i+1] + Co[j,i-1])/2
#     return C

#Para evaluar los nodos de las fronteras en esos puntos para evitar que halla acumulación
#en ese punto el transporte de materia es constante, por ejemplo para x seria dNAx/dx = 0
#lo que significa que la transferencia de masa en el último nodo deber ser exactamente igual al
#nodo anterior, por lo que si estamos en la ezquina izquierda, i = 0,  Co[j,i] debe de elegirse
#una anterior a ese punto, en otras palabras todos los indices de i se sumara 1 Co[j,i+1],
#de esta manera el transporte NA de esa dirección sera constante e igual al pasado lo que evita 
#la acumulación o desacumulación en ese punto frontera.
def con_sup(i,j,Co,dt,dh,Dab,vox,voy,u,h,q):
    # C = Co[j,i] + dt/(2*dh)*(-vox[j,i]*(Co[j,i+1] - Co[j,i-1]) - voy[j,i]*(0 - Co[j-1,i]) \
    #                          -Co[j,i]*(vox[j,i+1] - vox[j,i-1] + voy[j,i] - voy[j-2,i]) + \
    #                          2*dh/h*(q[j,i] - Co[j,i]*u))
    if voy[j,i] > 0:
        #salidas
        C = Co[j,i] + dt/(2*dh)*(-vox[j,i]*(Co[j,i+1] - Co[j,i-1]) - voy[j-1,i]*(Co[j,i] - Co[j-2,i]) \
                                -Co[j,i]*(vox[j,i+1] - vox[j,i-1]) - Co[j-1,i]*(voy[j,i] - voy[j-2,i]) + \
                                2*dh/h*(q[j,i] - Co[j-1,i]*u))   
                                #(q[j,i] - Co[j-1,i]*u)/h)
    else:
        #entradas
        C = Co[j,i] + dt/(2*dh)*(-vox[j,i]*(Co[j,i+1] - Co[j,i-1]) - voy[j,i]*(0 - Co[j-1,i]) \
                                 -Co[j,i]*(vox[j,i+1] - vox[j,i-1] + voy[j,i] - voy[j-2,i]) + \
                                 2*dh/h*(q[j,i] - Co[j,i]*u))
                                 #(q[j,i] - Co[j,i]*u)/h)
        C = 0.0
    
    #agregamos una restricci si tenemos concentraciones negativas
    if C < 0:
        C = (Co[j-1,i] + Co[j,i+1] + Co[j,i-1])/10000
        if C > Coi:
            C = Coi
    return C

def con_inf(i,j,Co,dt,dh,Dab,vox,voy,u,h,q):
    # C = Co[j,i] + dt/(2*dh)*(-vox[j,i]*(Co[j,i+1] - Co[j,i-1]) - voy[j,i]*(Co[j+1,i] - 0) \
    #                          -Co[j,i]*(vox[j,i+1] - vox[j,i-1] + voy[j+2,i] - voy[j,i]) + \
    #                          2*dh/h*(q[j,i] - Co[j,i]*u))
    if voy[j,i] < 0:
        #salida    
        C = Co[j,i] + dt/(2*dh)*(-vox[j,i]*(Co[j,i+1] - Co[j,i-1]) - voy[j+1,i]*(Co[j+2,i] - Co[j,i]) \
                                -Co[j,i]*(vox[j,i+1] - vox[j,i-1]) - Co[j+1,i]*(voy[j+2,i] - voy[j,i]) + \
                                2*dh/h*(q[j,i] - Co[j+1,i]*u))
                                #(q[j,i] - Co[j+1,i]*u)/h)
    else:
        #entrada
        C = Co[j,i] + dt/(2*dh)*(-vox[j,i]*(Co[j,i+1] - Co[j,i-1]) - voy[j,i]*(Co[j+1,i] - 0) \
                                 -Co[j,i]*(vox[j,i+1] - vox[j,i-1] + voy[j+2,i] - voy[j,i]) + \
                                 2*dh/h*(q[j,i] - Co[j,i]*u))
                                 #(q[j,i] - Co[j,i]*u)/h)    
        C = 0.0                             
    #agregamos una restricci si tenemos concentraciones negativas
    if C < 0:
        C = (Co[j+1,i] + Co[j,i+1] + Co[j,i-1])/10000
        if C > Coi:
            C = Coi
    return C

def con_izq(i,j,Co,dt,dh,Dab,vox,voy,u,h,q):
    # C = Co[j,i] + dt/(2*dh)*(-vox[j,i]*(Co[j,i+1] - 0) - voy[j,i]*(Co[j+1,i] - Co[j-1,i]) \
    #                          -Co[j,i]*(vox[j,i+2] - vox[j,i] + voy[j+1,i] - voy[j-1,i]) + \
    #                          2*dh/h*(q[j,i] - Co[j,i]*u))
    if vox[j,i] < 0:
        #salida
        C = Co[j,i] + dt/(2*dh)*(-vox[j,i+1]*(Co[j,i+2] - Co[j,i]) - voy[j,i]*(Co[j+1,i] - Co[j-1,i]) \
                                -Co[j,i+1]*(vox[j,i+2] - vox[j,i]) - Co[j,i]*(voy[j+1,i] - voy[j-1,i]) + \
                                2*dh/h*(q[j,i] - Co[j,i+1]*u))
                                #(q[j,i] - Co[j,i+1]*u)/h)   
    else:
        #entrada
        C = Co[j,i] + dt/(2*dh)*(-vox[j,i]*(Co[j,i+1] - 0) - voy[j,i]*(Co[j+1,i] - Co[j-1,i]) \
                                 -Co[j,i]*(vox[j,i+2] - vox[j,i] + voy[j+1,i] - voy[j-1,i]) + \
                                 2*dh/h*(q[j,i] - Co[j,i]*u))    
                                 #(q[j,i] - Co[j,i]*u)/h)
        C = 0.0
        
    #agregamos una restricci si tenemos concentraciones negativas
    if C < 0:
        C = (Co[j+1,i] + Co[j-1,i] + Co[j,i+2])/10000
        if C > Coi:
            C = Coi
    return C

def con_der(i,j,Co,dt,dh,Dab,vox,voy,u,h,q):
    # C = Co[j,i] + dt/(2*dh)*(-vox[j,i]*(0 - Co[j,i-1]) - voy[j,i]*(Co[j+1,i] - Co[j-1,i]) \
    #                          -Co[j,i]*(vox[j,i] - vox[j,i-2] + voy[j+1,i] - voy[j-1,i]) + \
    #                          2*dh/h*(q[j,i] - Co[j,i]*u))    
    if vox[j,i] > 0:
        #salida    
        C = Co[j,i] + dt/(2*dh)*(-vox[j,i-1]*(Co[j,i] - Co[j,i-2]) - voy[j,i]*(Co[j+1,i] - Co[j-1,i]) \
                                -Co[j,i-1]*(vox[j,i] - vox[j,i-2]) - Co[j,i]*(voy[j+1,i] - voy[j-1,i]) + \
                                2*dh/h*(q[j,i] - Co[j,i-1]*u))    
                                #(q[j,i] - Co[j,i-1]*u)/h)
    else:
        #entrada
        C = Co[j,i] + dt/(2*dh)*(-vox[j,i]*(0 - Co[j,i-1]) - voy[j,i]*(Co[j+1,i] - Co[j-1,i]) \
                                 -Co[j,i]*(vox[j,i] - vox[j,i-2] + voy[j+1,i] - voy[j-1,i]) + \
                                 2*dh/h*(q[j,i] - Co[j,i]*u))
                                 #(q[j,i] - Co[j,i]*u)/h)
        C = 0.0
            
    #agregamos una restricci si tenemos concentraciones negativas
    if C < 0:
        C = (Co[j+1,i] + Co[j-1,i] + Co[j,i-2])/10000
        if C > Coi:
            C = Coi
    return C


#definimos las fronteras de las esquinas del sistema
def con_sup_izq(i,j,Co,dt,dh,Dab,vox,voy,u,h,q):
    if vox[j,i] > 0 and voy[j,i] > 0:
        C = Co[j,i] + dt/(2*dh)*(-vox[j,i]*(Co[j,i+1] - 0) - voy[j-1,i]*(Co[j,i] - Co[j-2,i]) \
                                 -Co[j,i]*(vox[j,i+2] - vox[j,i] + voy[j,i] - voy[j-2,i]) + \
                                 2*dh/h*(q[j,i] - Co[j,i]*u))
    elif vox[j,i] < 0 and voy[j,i] > 0:
        C = Co[j,i] + dt/(2*dh)*(-vox[j,i+1]*(Co[j,i+2] - Co[j,i]) - voy[j-1,i]*(Co[j,i] - Co[j-2,i]) \
                                -Co[j,i+1]*(vox[j,i+2] - vox[j,i]) - Co[j-1,i]*(voy[j,i] - voy[j-2,i]) + \
                                2*dh/h*(q[j,i] - Co[j,i+1]*u))
    elif vox[j,i] > 0 and voy[j,i] < 0:
        C = Co[j,i] + dt/(2*dh)*(-vox[j,i]*(Co[j,i+1] - 0) - voy[j,i]*(0 - Co[j-1,i]) \
                                 -Co[j,i]*(vox[j,i+1] - vox[j,i-1] + voy[j,i] - voy[j-2,i]) + \
                                 2*dh/h*(q[j,i] - Co[j,i]*u))
    elif vox[j,i] < 0 and voy[j,i] < 0:
        C = Co[j,i] + dt/(2*dh)*(-vox[j,i+1]*(Co[j,i+2] - Co[j,i]) - voy[j,i]*(0 - Co[j-1,i]) \
                                -Co[j,i+1]*(vox[j,i+2] - vox[j,i]) - Co[j-1,i]*(voy[j,i] - voy[j-2,i]) + \
                                2*dh/h*(q[j,i] - Co[j,i+1]*u))
    #C = Co[j,i] + dt/(2*dh)*(-vox[j,i+1]*(Co[j,i+2] - Co[j,i]) - voy[j-1,i]*(Co[j,i] - Co[j-2,i]) \
    #                         -Co[j,i+1]*(vox[j,i+2] - vox[j,i]) - Co[j-1,i]*(voy[j,i] - voy[j-2,i]) + \
    #                         2*dh/h*(q[j,i] - Co[j,i]*u))    
    #agregamos una restricci si tenemos concentraciones negativas
    C = 0
    if C < 0:
        C = (Co[j,i+1] + Co[j-1,i])/2
        if C > Coi:
            C = Coi
    return C

def con_sup_der(i,j,Co,dt,dh,Dab,vox,voy,u,h,q):
    if vox[j,i] > 0 and voy[j,i] > 0:
        C = Co[j,i] + dt/(2*dh)*(-vox[j,i-1]*(Co[j,i] - Co[j,i-2]) - voy[j-1,i]*(Co[j,i] - Co[j-2,i]) \
                                -Co[j,i-1]*(vox[j,i] - vox[j,i-2]) - Co[j-1,i]*(voy[j,i] - voy[j-2,i]) + \
                                2*dh/h*(q[j,i] - Co[j,i-1]*u)) 
    elif vox[j,i] < 0 and voy[j,i] > 0:
        C = Co[j,i] + dt/(2*dh)*(-vox[j,i]*(0 - Co[j,i-1]) - voy[j-1,i]*(Co[j,i] - Co[j-2,i]) \
                                 -Co[j,i]*(vox[j,i] - vox[j,i-2] + voy[j,i] - voy[j-2,i]) + \
                                 2*dh/h*(q[j,i] - Co[j,i]*u))
    elif vox[j,i] > 0 and voy[j,i] < 0:
        C = Co[j,i] + dt/(2*dh)*(-vox[j,i-1]*(Co[j,i] - Co[j,i-2]) - voy[j,i]*(0 - Co[j-1,i]) \
                                 -Co[j,i-1]*(vox[j,i] - vox[j,i-2]) - Co[j-1,i]*(voy[j,i] - voy[j-2,i]) + \
                                 2*dh/h*(q[j,i] - Co[j,i]*u))
    elif vox[j,i] < 0 and voy[j,i] < 0:
        C = Co[j,i] + dt/(2*dh)*(-vox[j,i]*(0 - Co[j,i-1]) - voy[j,i]*(0 - Co[j-1,i]) \
                                 -Co[j,i]*(vox[j,i] - vox[j,i-2] + voy[j,i] - voy[j-2,i]) + \
                                 2*dh/h*(q[j,i] - Co[j,i]*u))
    #C = Co[j,i] + dt/(2*dh)*(-vox[j,i-1]*(Co[j,i] - Co[j,i-2]) - voy[j-1,i]*(Co[j,i] - Co[j-2,i]) \
    #                         -Co[j,i-1]*(vox[j,i] - vox[j,i-2]) - Co[j-1,i]*(voy[j,i] - voy[j-2,i]) + \
    #                         2*dh/h*(q[j,i] - Co[j,i]*u))    
    #agregamos una restricci si tenemos concentraciones negativas
    C = 0
    if C < 0:
        C = (Co[j,i-1] + Co[j-1,i])/2
        if C > Coi:
            C = Coi
    return C

def con_inf_izq(i,j,Co,dt,dh,Dab,vox,voy,u,h,q):
    if vox[j,i] > 0 and voy[j,i] > 0:
        C = Co[j,i] + dt/(2*dh)*(-vox[j,i]*(Co[j,i+1] - 0) - voy[j,i]*(Co[j+1,i] - 0) \
                                 -Co[j,i]*(vox[j,i+2] - vox[j,i] + voy[j+2,i] - voy[j,i]) + \
                                 2*dh/h*(q[j,i] - Co[j,i]*u))
    elif vox[j,i] < 0 and voy[j,i] > 0:
        C = Co[j,i] + dt/(2*dh)*(-vox[j,i]*(Co[j,i+2] - Co[j,i]) - voy[j,i]*(Co[j+1,i] - 0) \
                                 -Co[j,i]*(vox[j,i+2] - vox[j,i] + voy[j+2,i] - voy[j,i]) + \
                                 2*dh/h*(q[j,i] - Co[j,i]*u))      
    elif vox[j,i] > 0 and voy[j,i] < 0:
        C = Co[j,i] + dt/(2*dh)*(-vox[j,i]*(Co[j,i+1] - 0) - voy[j,i]*(Co[j+2,i] - Co[j,i]) \
                                 -Co[j,i]*(vox[j,i+2] - vox[j,i] + voy[j+2,i] - voy[j,i]) + \
                                 2*dh/h*(q[j,i] - Co[j,i]*u))    
    elif vox[j,i] < 0 and voy[j,i] < 0:
        C = Co[j,i] + dt/(2*dh)*(-vox[j,i+1]*(Co[j,i+2] - Co[j,i]) - voy[j+1,i]*(Co[j+2,i] - Co[j,i]) \
                                -Co[j,i+1]*(vox[j,i+2] - vox[j,i]) - Co[j+1,i]*(voy[j+2,i] - voy[j,i]) + \
                                2*dh/h*(q[j,i] - Co[j,i+1]*u))   
    #C = Co[j,i] + dt/(2*dh)*(-vox[j,i+1]*(Co[j,i+2] - Co[j,i]) - voy[j+1,i]*(Co[j+2,i] - Co[j,i]) \
    #                         -Co[j,i+1]*(vox[j,i+2] - vox[j,i]) - Co[j+1,i]*(voy[j+2,i] - voy[j,i]) + \
    #                         2*dh/h*(q[j,i] - Co[j,i]*u))    
    #agregamos una restricci si tenemos concentraciones negativas
    C = 0
    if C < 0:
        C = (Co[j,i+1] + Co[j+1,i])/2
        if C > Coi:
            C = Coi
    return C

def con_inf_der(i,j,Co,dt,dh,Dab,vox,voy,u,h,q):
    if vox[j,i] > 0 and voy[j,i] > 0:
        C = Co[j,i] + dt/(2*dh)*(-vox[j,i]*(Co[j,i] - Co[j,i-2]) - voy[j,i]*(Co[j+1,i] - 0) \
                                 -Co[j,i]*(vox[j,i] - vox[j,i-2] + voy[j+2,i] - voy[j,i]) + \
                                 2*dh/h*(q[j,i] - Co[j,i]*u)) 
    elif vox[j,i] < 0 and voy[j,i] > 0:
        C = Co[j,i] + dt/(2*dh)*(-vox[j,i]*(0 - Co[j,i-1]) - voy[j,i]*(Co[j+1,i] - 0) \
                                 -Co[j,i]*(vox[j,i] - vox[j,i-2] + voy[j+2,i] - voy[j,i]) + \
                                 2*dh/h*(q[j,i] - Co[j,i]*u))
    elif vox[j,i] > 0 and voy[j,i] < 0:
        C = Co[j,i] + dt/(2*dh)*(-vox[j,i-1]*(Co[j,i] - Co[j,i-2]) - voy[j+1,i]*(Co[j+2,i] - Co[j,i]) \
                                -Co[j,i-1]*(vox[j,i] - vox[j,i-2]) - Co[j+1,i]*(voy[j+2,i] - voy[j,i]) + \
                                2*dh/h*(q[j,i] - Co[j,i-1]*u)) 
    elif vox[j,i] < 0 and voy[j,i] < 0:
        C = Co[j,i] + dt/(2*dh)*(-vox[j,i]*(0 - Co[j,i-1]) - voy[j+1,i]*(Co[j+2,i] - Co[j,i]) \
                                 -Co[j,i]*(vox[j,i] - vox[j,i-2] + voy[j+2,i] - voy[j,i]) + \
                                 2*dh/h*(q[j,i] - Co[j,i]*u))   
    #C = Co[j,i] + dt/(2*dh)*(-vox[j,i-1]*(Co[j,i] - Co[j,i-2]) - voy[j+1,i]*(Co[j+2,i] - Co[j,i]) \
    #                         -Co[j,i-1]*(vox[j,i] - vox[j,i-2]) - Co[j+1,i]*(voy[j+2,i] - voy[j,i]) + \
    #                         2*dh/h*(q[j,i] - Co[j,i]*u))    
    #agregamos una restricci si tenemos concentraciones negativas
    C = 0
    if C < 0:
        C = (Co[j,i-1] + Co[j+1,i])/2
        if C > Coi:
            C = Coi
    return C

#evamos la superficie, definimos un tiempo final
tf=23.95

#generamos los vectores para "x" y "y"
x = np.linspace(0, dh*nodos[0], num = nodos[0])
y = np.linspace(0, dh*nodos[1], num = nodos[1])

#generamos la malla de los valores
X,Y = np.meshgrid(x,y)

#generamos un vector que define los niveles del diagrama de countorno
#levels = np.arange(0,30000,1000)
levels = np.arange(0.1,100,5)

#contador
n = 1
#m = dt

#convertimos las matrices X y Y a las respectivas coordenadas en latitud y longitud
Xi = (X-np.min(X))/(np.max(X) - np.min(X))*(np.max(lon_lim) - np.min(lon_lim)) + np.min(lon_lim)
Yi = (Y-np.min(Y))/(np.max(Y) - np.min(Y))*(np.max(lat_lim) - np.min(lat_lim)) + np.min(lat_lim)

#generamos una variable donde guadaremos los promedios de la concentración
#y determinamos cuales son los indices de la posición de la estación de monitoreo
#que vamos a emplear para comparar los resultados del modelo, estos indices
#nos ayudan para ir guardando la infor de la concentración del contaminantes por
#una hora y depues sacar su promedio y guardarlo en la variable
CO_con = np.array([Coi])

#convertimos las coordenadas en distancia en función de los nodos y la distancia (dh)
coord_dist_lat_monitoreo = 0 + (nodos[0]*dh - 0)/(lat_lim[1] - lat_lim[0])*(coord_lat_lon_monitoreo[:,0] - lat_lim[0])
coord_dist_lon_monitoreo = 0 + (nodos[1]*dh - 0)/(lon_lim[1] - lon_lim[0])*(coord_lat_lon_monitoreo[:,1] - lon_lim[0])

#unimos los arreglos para dejar las coordenadas de cada una
coord_dist_monitoreo = np.array([coord_dist_lat_monitoreo,coord_dist_lon_monitoreo]).T
del(coord_dist_lat_monitoreo, coord_dist_lon_monitoreo)

#evaluamos los indices de la matriz donde se encuetra el nodo más cercano a la estación de monitoreo
if estacion == 'SMA':
    SMA_lat_lon = np.array([round(coord_dist_monitoreo[-1,0]/dh),round(coord_dist_monitoreo[-1,1]/dh)])
if estacion == 'CEN':
    SMA_lat_lon = np.array([round(coord_dist_monitoreo[0,0]/dh),round(coord_dist_monitoreo[0,1]/dh)])
if estacion == 'CBT':
    SMA_lat_lon = np.array([round(coord_dist_monitoreo[1,0]/dh),round(coord_dist_monitoreo[1,1]/dh)])
if estacion == 'IED':
    SMA_lat_lon = np.array([round(coord_dist_monitoreo[2,0]/dh),round(coord_dist_monitoreo[2,1]/dh)])

CO_horario = np.array([Coi])

#definimos una ruta donde se van a guardar las imagenes para hacer un gif
ruta_img = 'C:/Users/hecto/OneDrive/Documentos/ITA/Posdoc Proyecto/articulo 3/articulo elsevier/'

#generamos una variable para determinar en que momento se alcanza el maximo y el valor de este
C_max = np.zeros((1,2))

#cargamos en memoria es escalador y el modelo para velocidad de viento
ruta_mod_vel = 'C:/Users/hecto/OneDrive/Documentos/ITA/Posdoc Proyecto/articulo 3/'
red_viento = load_model(ruta_mod_vel + 'modelo_viento.h5')
scaler = joblib.load(ruta_mod_vel + 'scaler_viento.pkl')

#importamos la función con el modelo de velocidad del viento y la función para hacer grids de matplotlib
import modulo_velocidad_viento as mvv
import matplotlib.gridspec as gridspec

#cerramos todas las ventanas
plt.close('all')

# Crear figura con espacio para colorbar (4 columnas, la última para la barra)
fig = plt.figure(figsize=(10,20)) #ancho y alto
#gs = gridspec.GridSpec(4, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.15, hspace=0.25)
gs = gridspec.GridSpec(5, 4, width_ratios=[1,1,1,0.05], height_ratios=[1,1,1,1,0.001], wspace=0.2, hspace=0.35)

axs = []
for i in range(12):
    row = i // 3
    col = i % 3
    axs.append(fig.add_subplot(gs[row, col]))

#variable para tener una referencia de tiempo para generar las figuras y guardar
t_ref = 0
ejes = 0

#evaluamos la superficie por diferencias finitas
for t in np.arange(0,tf,dt):
    
    #if t > 7 and t < 21:
    #    q = q_movil
    #else:
    #    q = qo
    #definimos los valores de h
    h = h_mezcla[n]
    #h = 500
    
    #evaluamos el modelo de emisiones
    q = emisiones(coord_lat_lon_movil, coord_dist_movil, red, t, dia, q_fijas, dh, q_area)
    
    #evaluamos por cada incremento en el tiempo las componentes de la velocidad
    #vox = np.full((nodos[0], nodos[1]), vo[n]*np.cos(np.deg2rad(do[n])))
    #voy = np.full((nodos[0], nodos[1]), vo[n]*np.sin(np.deg2rad(do[n])))
    #evaluamos el modelo de red de velocidad del viento
    ws = mvv.mod_velo_vient(lon_lim, lat_lim, nodos[0], vo[n]*np.cos(np.deg2rad(do[n]))/3600, vo[n]*np.sin(np.deg2rad(do[n]))/3600, doy, t, np, red_viento, scaler)
    vox, voy = ws[:,0]*3600, ws[:,1]*3600
    vox, voy = vox.reshape(nodos[0], nodos[1]), voy.reshape(nodos[0], nodos[1])
    
    #nodos centrales    
    for i in range(1,nodos[1]-1):
        for j in range(1,nodos[0]-1):
            Ct[j,i] = con(i,j,Co,dt,dh,Dab,vox,voy,u,h,q)
    
    #nodos superio
    for i in range(1,nodos[1]-1):
        j = nodos[1]-1
        Ct[j,i] = con_sup(i,j,Co,dt,dh,Dab,vox,voy,u,h,q)
        
    #nodos inferiores
    for i in range(1,nodos[1]-1):
          j = 0
          Ct[j,i] = con_inf(i,j,Co,dt,dh,Dab,vox,voy,u,h,q)
    
    #nodos izquierda
    for j in range(1,nodos[0]-1):
          i = 0
          Ct[j,i] = con_izq(i,j,Co,dt,dh,Dab,vox,voy,u,h,q)
    
    #nodos derecha
    for j in range(1,nodos[0]-1):
          i = nodos[0]-1
          Ct[j,i] = con_der(i,j,Co,dt,dh,Dab,vox,voy,u,h,q)
    
    #nodo esquina superior izquierda
    i = 0
    j = nodos[1] - 1
    Ct[j,i] = con_sup_izq(i,j,Co,dt,dh,Dab,vox,voy,u,h,q)
    
    #nodo esquina superior derecha
    i = nodos[0] - 1
    j = nodos[1] - 1
    Ct[j,i] = con_sup_der(i,j,Co,dt,dh,Dab,vox,voy,u,h,q)
    
    #nodo esquina inferior izquierda
    i = 0
    j = 0
    Ct[j,i] = con_inf_izq(i,j,Co,dt,dh,Dab,vox,voy,u,h,q)
    
    #nodo esquina inferior derecha
    i = nodos[0] - 1
    j = 0
    Ct[j,i] = con_inf_der(i,j,Co,dt,dh,Dab,vox,voy,u,h,q)
    
    # #modificamos la velocidad y direccion del vienyo
    # if t > 1.5:
    #     do = np.full(nodos, 300)
    
    #     #evaluamos las componente de la velocidad del viento
    #     vox = vo*np.cos(np.deg2rad(do))
    #     voy = vo*np.sin(np.deg2rad(do))
    
    # if t > 2.5:
    #     do = np.full(nodos, 125)
    
    #     #evaluamos las componente de la velocidad del viento
    #     vox = vo*np.cos(np.deg2rad(do))
    #     voy = vo*np.sin(np.deg2rad(do))
    
    
    #restricciones
    #Ct = np.where(Ct < 0, 0.1, Ct)
    #Ct = np.where(Ct > 30000, 30000, Ct)
    Co = Ct*1
    
    
    #si se alcanza un nuevo maximo lo guardamos
    if Ct.max() > C_max[0,1]:
        C_max[0,0], C_max[0,1] = t, Ct.max()
    
    # if (t == 0 and m == dt) or m > 1:
    #     Xi = (X-np.min(X))/(np.max(X) - np.min(X))*(np.max(lon_lim) - np.min(lon_lim)) + np.min(lon_lim)
    #     Yi = (Y-np.min(Y))/(np.max(Y) - np.min(Y))*(np.max(lat_lim) - np.min(lat_lim)) + np.min(lat_lim)
    
    #     #crear el mapa base centrado en Aguascalientes
    #     mapa = Basemap(llcrnrlon = min(lon_lim), llcrnrlat = min(lat_lim),
    #                     urcrnrlon = max(lon_lim), urcrnrlat = max(lat_lim), resolution='h')
    #     #mapa = Basemap(llcrnrlon=-102.371346,llcrnrlat=21.782667,urcrnrlon=-102.223352,urcrnrlat=21.992027, resolution='h')
    
    #     #añadir las calles y avenidas de la ciudad
    #     #algunos servicios interesantes: 'World_Topo_Map', 'World_Street_Map', 'NatGeo_World_Map', 'ESRI_Imagery_World_2D'
    #     #mapa.arcgisimage(service='World_Street_Map', xpixels=700, verbose=False)
    #     mapa.arcgisimage(service='World_Topo_Map', dpi = 300, verbose=False)
    #     plt.plot(coord_lat_lon[:,1], coord_lat_lon[:,0], marker = 's', markersize = 14, linestyle = 'None', markerfacecolor = 'None')
    #     plt.plot(coord_lat_lon_movil[:,1], coord_lat_lon_movil[:,0], marker = 'o', markersize = 14, linestyle = 'None', markerfacecolor = 'None')
    #     plt.contourf(Xi,Yi,Ct, cmap = 'jet', vmax = 10000, vmin = 0, extend = 'max', levels = levels, alpha = 0.2)
    #     plt.colorbar(label='Concentración ($\mu$g/m$^3$)')
    #     plt.title(f'Tiempo: {t:.0f} horas   $\\theta$: {do[n]}   Vel: {vo[n]/3600:.2f} m/s')
    #     plt.savefig(f'superficie_tiempo_{t:.0f}.png', dpi=300)
    #     plt.clf()
    #     m = dt
    
    #graficamos
    #solo en la 1ra vuelta se usa la generación del mapa, después solo eliminaremos 
    #la figura del contorno y actualizaremos
    if t == 0 or t-t_ref > 2:
        #generamos la figura y los ejes
        ax = axs[ejes]
        #crear el mapa base centrado en Aguascalientes
        # Limites y mapa base
        ax.set_xlim(lon_lim)
        ax.set_ylim(lat_lim)
        ctx.add_basemap(ax, zoom=13, crs="EPSG:4326",
                        attribution=False,
                        source=ctx.providers.OpenStreetMap.HOT)
        
        
        #añadir las calles y avenidas de la ciudad
        #algunos servicios interesantes: 'World_Topo_Map', 'World_Street_Map', 'NatGeo_World_Map', 'ESRI_Imagery_World_2D', 'World_Imagery'
        point_s = ax.scatter(coord_lat_lon[:,1], coord_lat_lon[:,0], marker = 's', s=100, color = 'blue', label = 'Point sources', edgecolor='k', alpha = 0.4)
        mobile_s = ax.scatter(coord_lat_lon_movil[:,1], coord_lat_lon_movil[:,0], marker = 'o', s = 100, color = 'green', label = 'Mobile sources', edgecolor='k', alpha = 0.4)
        area_s = ax.scatter(coord_lat_lon_area[:,1], coord_lat_lon_area[:,0], marker = 'p', s = 100, color = 'yellow', label = 'Area sources', edgecolor='k', alpha = 0.4)
        monitoring_s = ax.scatter(coord_lat_lon_monitoreo[:,1], coord_lat_lon_monitoreo[:,0], marker = '^', s = 100, color = 'purple', label = 'Monitoring station', edgecolor='k', alpha = 0.4)
        ruoa_s = ax.scatter(-102.3190, 21.9157, marker = '*', s = 100, color = 'red', label = 'RUOA', edgecolor='k', alpha = 0.4)
        # Sin ejes
        ax.set_xticks([])
        ax.set_yticks([])
        #plt.legend(ncol=3, bbox_to_anchor=(1.04, 1.178))
        #plt.xticks(np.linspace(min(Xi[0,:]), max(Xi[0,:]),5))
        #plt.yticks(np.linspace(min(Yi[:,0]), max(Yi[:,0]),5))
        contorno = ax.contourf(Xi,Yi,Ct, cmap = 'turbo', extend = 'max', vmax = 100, vmin = 0.1, levels = levels, alpha = 0.5)
        #plt.colorbar(label='PM$_{2.5}$ Concentration ($\mu$g/m$^3$)')
        ax.set_title(f'Hour: {t:.1f}\n$v_{{RUOA}}$: {vo[n]/3600:.1f} m/s | $\\theta_{{RUOA}}$: {do[n]}°')
        plt.tight_layout()
        
        #actualizamos t_ref
        t_ref = t*1
        ejes += 1
        #plt.savefig(f'{ruta_img}_{t}.png', transparent=False, facecolor='white')
        #plt.pause(0.001)
    # else:
    #     contorno.remove()
    #     contorno = plt.contourf(Xi,Yi,Ct, cmap = 'jet', vmax = 100, vmin = 0.1, extend = 'max', levels = levels, alpha = 0.2)
    #     plt.title(f'Hour: {t:.1f} | $v_{{RUOA}}$: {vo[n]/3600:.1f} m/s | $\\theta_{{RUOA}}$: {do[n]:.1f}°')
    #     # Sin ejes
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     plt.pause(0.001)
        
    # plt.plot(coord_dist[:,1], coord_dist[:,0], marker = 's', markersize = 14, linestyle = 'None', markerfacecolor = 'None')
    # plt.plot(coord_dist_movil[:,1], coord_dist_movil[:,0], marker = 'o', markersize = 14, linestyle = 'None', markerfacecolor = 'None')
    # plt.contourf(X,Y,Ct, cmap = 'jet', vmax = 10000, vmin = 0, extend = 'max', levels = levels)
    # #plt.contourf(X,Y,Ct)
    # #if t == 0:
    #     #plt.colorbar(label='Concentración ($\mu$g/m$^3$)')
    # plt.colorbar(label='Concentración ($\mu$g/m$^3$)')
    # plt.title(f'Tiempo: {t:.2f} horas   $\\theta$: {do[n]}   Vel: {vo[n]/3600:.2f} m/s')
    # plt.draw()
    # plt.pause(0.01)
    # if t < tf-dt:
    #     plt.clf()
    #aumentamos en 1 el contador
    n = n + 1
    #m = m + dt
    
    #generamos una variable que nos apoyara para ir guardando la info
    if t == 0:
        horas = t*1
    
    if t - horas > 1:
        #reseteamos el valor de horas y eliminamos los otros valores de CO_horarios
        #de la lista
        horas = t*1
        CO_con = np.concatenate((CO_con, [CO_horario.mean()]))
        CO_horario = np.array([CO_horario[-1]])
    else:
        #vamos ir guardando la información de la concentración
        CO_horario = np.concatenate((CO_horario, [Ct[SMA_lat_lon[0], SMA_lat_lon[1]]]))
    
    #plt.colorbar(label = 'Concentración ($\mu$g/m$^3$)')

#este sección de codigo es para agregar el legend en la parte inferior
legend_ax = fig.add_subplot(gs[4, 0:3])
legend_ax.axis('off')
legend_ax.legend(handles = [point_s, mobile_s, area_s, monitoring_s, ruoa_s], loc='center', fontsize=12, frameon=True, ncol=5)
#labels=['Point sources','Mobile sources','Area sources','Monitoring station','RUOA']

#esta sección de codigo es para agregar la barra de colores en la lateral
cbar_ax = fig.add_subplot(gs[0:4, 3])
cbar = fig.colorbar(contorno, cax=cbar_ax)
cbar.set_label('PM$_{2.5}$ concentration ($\mu$g/m$^3$)', fontsize=14)
cbar.ax.tick_params(labelsize=12) 
fig.subplots_adjust(left=0.03, right=0.925, top=0.95, bottom=0.05)

#guardamos la figura
plt.savefig(ruta_img + 'PM25_con_sup.pdf', dpi = 300)

###############################################################################
#                         GENERAMOS EL GIF ANIMADO                            #
###############################################################################
# #generamos el gif
# import imageio

# #generamos una lista vacia donde  y donde iremos guardando en memoria todos los
# #frames (imagenes) que se fueron guardando de la simulación
# frames = []
# for t in np.arange(0,tf,dt):
#     image = imageio.v2.imread(f'{ruta_img}_{t}.png')
#     frames.append(image)

# #generamos el gif animado a partir de todos los frames
# imageio.mimsave('simulacion.gif', # output gif
#                 frames,          # array of input frames
#                 fps=15)         # optional: frames per second

###############################################################################
#   COMPRAMOS LOS RESULTADOS DEL MODELO EN LAS ESTACIONES DE MONITOREO        #
###############################################################################

#madamos llamar a la base de datos con la info de la estación de SMA para compararla
#con la obtenida por el modelo
#definimos la ruta donde esta el archivo
#ruta = 'Datos SINAICA - Secretaría de Medio Ambiente - CO -  - 2022-06-01 - 2022-06-01.csv'

#leemos el df con los datos
df = pd.read_csv(ruta + base_datos[estacion][0], encoding='latin1')
#df = pd.read_csv(ruta + base_datos[estacion][0])

#eliminamos la 1ra fila del df
df.drop(index = 0, inplace = True)

#convertimos a formato de fecha
df.Fecha = pd.to_datetime(df.Fecha)

#filtramos solmente los datos del día que nos interesa
df = df[df.Fecha.dt.day == dia_mes]

#df = df.iloc[0:24]

#convertimos las ppm en ug/m3 aplicando el siguiente factor de conversión de la NOM
#df.valor = df.valor*10000/9

# CO_con_mod1 = np.array([0.1,16.353,28.4999,19.1739,17.397,12.2127,11.0459,12.4978,990.717,2912.16,2216.64,
#                         723.253,740.914,910.608,894.099,807.346,656.755,931.095,1051.79,100.53,305.552,515.125,
#                         357.062,264.085])

# CO_con_mod1 = np.array([1.00000000e-01, 1.63530431e+01, 2.84998564e+01, 1.91738909e+01,
#                        1.73969592e+01, 1.38911308e+01, 2.14326658e+02, 7.18147429e+02,
#                        2.05377266e+03, 3.14644954e+03, 2.37479515e+03, 4.43185046e+02,
#                        6.99519412e+02, 1.01729402e+03, 1.00446853e+03, 6.53014348e+02,
#                        5.51037184e+02, 1.19988911e+03, 1.21035300e+03, 9.35355259e+01,
#                        2.73850036e+01, 2.61370755e+01, 1.01586132e+02, 8.22118403e+01])

#graficamos los datos y los modelos
plt.figure()
plt.plot(np.linspace(0,23,24), df['Concentraciones horarias'], 'o', markerfacecolor = 'None', color = 'black', ms = 9, label = f'Estación: {estacion}')
#plt.plot(np.linspace(0,23,24), CO_con_mod1, label = 'Modelo 1')
plt.plot(np.linspace(0,23,24), CO_con, label = 'Modelo')
plt.hlines(41, 0, 24, ls='--', color = 'r', label='NOM Lím: Año 1')
plt.hlines(33, 0, 24, ls='--', color = 'y', label='NOM Lím: Año 3')
plt.hlines(25, 0, 24, ls='--', color = 'm', label='NOM Lím: Año 5')
plt.xlabel('Hora del día')
plt.ylabel('Concentración PM$_{2.5}$ ($\mu$g/m$^3$)')
plt.title(f'Concentración promedio horaria, {estacion} ({fecha})')
plt.xlim(0,23)
plt.ylim(0,np.max([max(df['Concentraciones horarias']),max(CO_con), 41])*1.2)
plt.xticks(np.linspace(0, 23, 24))
plt.legend()

# plt.figure()

# #convertimos las matrices X y Y a las respectivas coordenadas en latitud y longitud
# Xi = (X-np.min(X))/(np.max(X) - np.min(X))*(np.max(lon_lim) - np.min(lon_lim)) + np.min(lon_lim)
# Yi = (Y-np.min(Y))/(np.max(Y) - np.min(Y))*(np.max(lat_lim) - np.min(lat_lim)) + np.min(lat_lim)

# #crear el mapa base centrado en Aguascalientes
# mapa = Basemap(llcrnrlon = min(lon_lim), llcrnrlat = min(lat_lim),
#                 urcrnrlon = max(lon_lim), urcrnrlat = max(lat_lim), resolution='h')
# #mapa = Basemap(llcrnrlon=-102.371346,llcrnrlat=21.782667,urcrnrlon=-102.223352,urcrnrlat=21.992027, resolution='h')

# #añadir las calles y avenidas de la ciudad
# #algunos servicios interesantes: 'World_Topo_Map', 'World_Street_Map', 'NatGeo_World_Map', 'ESRI_Imagery_World_2D'
# #mapa.arcgisimage(service='World_Street_Map', xpixels=700, verbose=False)
# mapa.arcgisimage(service='World_Topo_Map', dpi = 300, verbose=False)
# line1, = plt.plot(coord_lat_lon[:,1], coord_lat_lon[:,0], marker = 's', markersize = 14, linestyle = 'None', markerfacecolor = 'None')
# line2, = plt.plot(coord_lat_lon_movil[:,1], coord_lat_lon_movil[:,0], marker = 'o', markersize = 14, linestyle = 'None', markerfacecolor = 'None')
# line3 = plt.contourf(Xi,Yi,Ct, cmap = 'jet', vmax = 10000, vmin = 0, extend = 'max', levels = levels, alpha = 0.2)
# plt.colorbar(label='Concentración ($\mu$g/m$^3$)')
# plt.title(f'Tiempo: {t:.2f} horas   $\\theta$: {do[n]}   Vel: {vo[n]/3600:.2f} m/s')

# #INTENTAR REMOVER CADA UNA DE LAS LINEAS, lo hago mañana
# #line3.remove()
