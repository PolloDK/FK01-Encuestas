#   * ____________________________________________ *
#   *|   ______   _    _                          |*
#   *|  |  ____| | |  / /    *      *             |*
#   *|  | |      | | / /     ****   ****          |*
#   *|  | |____  | |/ /      ***************      |*
#   *|  |  ____| |   /       ******************   |*
#   *|  | |      |   \       ***************      |*
#   *|  | |      | |\ \      ****   ****          |*
#   *|  | |      | | \ \     *      *             |*
#   *|  |_|      |_|  \_\                         |*
#   *|                                            |*
#   *|  E C O N O M I C S                         |*
#   *|____________________________________________|*
#   *                                              *
# ******************************************************************************
# AUTOR: FK Economics (@vsanmartin)
# FECHA: 10 de abril 2025
# ACTUAL: 10 de abril 2025
# DESCR: Series de tiempo sentiment analysis
# ******************************************************************************

# Load the required libraries
library(readxl)
library(forecast)
library(ggplot2)
library(dplyr)
library(writexl)
library(xts)
library(zoo)

# Load the data
data_score <- read_excel("../_data/scores-sentiment.xlsx")
data_encuesta <- read_excel("../_data/encuesta.xlsx")

############################## Data score ######################################
serie_positive <- xts(data_score$score_positive, order.by = data_score$date)
autoplot(serie_positive)
serie_negative <- xts(data_score$score_negative, order.by = data_score$date)
autoplot(serie_negative)
serie_neutral <- xts(data_score$score_neutral, order.by = data_score$date)
autoplot(serie_neutral)

#  Fit our ARIMA model:
modelo_arima_positive <- auto.arima(serie_positive)
modelo_arima_positive
checkresiduals(modelo_arima_positive) # Al límte

modelo_arima_negative <- auto.arima(serie_negative)
modelo_arima_negative
checkresiduals(modelo_arima_negative) # Al límite

modelo_arima_neutral <- auto.arima(serie_neutral)
modelo_arima_neutral
checkresiduals(modelo_arima_neutral) # Mal modelo

############################## Data encuesta ###################################
##Fechas no estaban en formato fecha
fecha_inicio <- as.Date("2022-03-18")  
fecha_fin <- as.Date("2025-03-07")    
fechas_semanales <- seq(from = fecha_inicio, to = fecha_fin, by = "week")
data_encuesta$fechas <- fechas_semanales

serie_encuesta <- xts(data_encuesta$aprobacion_boric, order.by = data_encuesta$fechas)
autoplot(serie_encuesta)

modelo_encuesta <- auto.arima(serie_encuesta)
modelo_encuesta
checkresiduals(modelo_encuesta) ## Muy buen modelo
