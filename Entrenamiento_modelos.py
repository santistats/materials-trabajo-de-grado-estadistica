###Importamos las librerías necsarias
import numpy as np 
import pandas as pd
from datetime import datetime
import ta
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import het_breuschpagan, het_arch, acorr_ljungbox
import statsmodels.api as sm
import itertools
import warnings
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error, silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.cluster import KMeans
from scipy import stats
from statsmodels.tsa.api import VAR
import math
from scipy.special import inv_boxcox
from statsmodels.stats.stattools import durbin_watson
from scipy import stats 
from typing import Union
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from statsmodels.tsa.seasonal import seasonal_decompose
from torch.utils import DataLoader, TensorDataset
from sklearn.model_selection import ParameterGrid
import copy
import time
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

plt.style.use('seaborn-v0_8-darkgrid')

price_data = pd.read_excel(r"Datos proyecto.xlsx", index_col=0)

#Añadimos la variable RSI
price_data.index = pd.to_datetime(price_data.index, format="%Y-%m-%d")
price_data['RSI'] = ta.momentum.rsi(price_data['close'], window=14)
#Interpolamos y rellenamos NA 
price_data.replace([0, np.inf, -np.inf], np.nan, inplace=True)
price_data.interpolate(method= "time", inplace=True)
price_data['RSI'] = price_data['RSI'].bfill()
##Tomamos hasta la fecha de corte que es el 6 de marzo
end_index = "2026-03-06"
end_index = pd.to_datetime(end_index)
price_data = price_data.loc[price_data.index <= end_index]

def stationarity_and_homoscedasticity_tests(
    data: Union[pd.DataFrame, pd.Series], 
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Aplica pruebas de estacionariedad (ADF) y homocedasticidad (Breusch-Pagan)
    a variables numéricas. Soporta tanto DataFrames como Series individuales.
    """
    
    #Si es una Serie, la convertimos a DataFrame para mantener la lógica
    if isinstance(data, pd.Series):
        df = data.to_frame()
    else:
        df = data.copy()

    results = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        series = df[col].dropna()
        
        if len(series) == 0:
            continue

        ##ADF
        adf_res = adfuller(series, autolag="AIC")
        adf_stat = adf_res[0]
        adf_pvalue = adf_res[1]
        estacionaria = adf_pvalue < alpha

        ###Breusch-Pagan
        X = np.arange(len(series))
        X = sm.add_constant(X)
        model = sm.OLS(series.values, X).fit()
        
        bp_test = het_breuschpagan(model.resid, X)
        bp_stat = bp_test[0]
        bp_pvalue = bp_test[1]
        homocedastica = bp_pvalue > alpha
        results.append({
            "Variable": col,
            "ADF estadístico": round(adf_stat, 4),
            "ADF p-valor": round(adf_pvalue, 4),
            "¿Estacionaria?": "Sí" if estacionaria else "No",
            "BP estadístico": round(bp_stat, 4),
            "BP p-valor": round(bp_pvalue, 4),
            "¿Homocedástica?": "Sí" if homocedastica else "No"
        })

    return pd.DataFrame(results)

##ARIMA
close = pd.DataFrame(price_data["close"])
##Dividimos en train y test
train_size_close = int(len(close)*0.80)
train_close, test_close = close[0:train_size_close], close[train_size_close:len(close)]
print(f"Observaciones totales: {len(close)}")
print(f"Entrenamiento: {len(train_close)}")
print(f"Prueba: {len(test_close)}")

##Hacemos el test
stationarity_and_homoscedasticity_tests(train_close)

##Transformamos con boxcox 
train_close["train_boxcox"], lambda_optim_close = stats.boxcox(train_close["close"])
print(f"Lamba optimo para boxcox: {lambda_optim_close}")

train_close["close_boxcox_diff"] = pd.Series(train_close["train_boxcox"]).diff().dropna()
##Comprobamos
stationarity_and_homoscedasticity_tests(train_close["close_boxcox_diff"])

##Función para buscar mejor ARIMA
def buscar_mejor_arima(train_series, max_p=3, max_q=3):
    best_aic = float("inf")
    best_order = None
    best_results = None 
    
    ps = range(max_p + 1)
    qs = range(max_q + 1)
    params = list(itertools.product(ps, [0], qs))
    
    print(f"Iniciando búsqueda...")
    
    for order in params:
        try:
            model = ARIMA(train_series, order=order, trend='c')
            results = model.fit()
            
            if results.aic < best_aic:
                best_aic = results.aic
                best_order = order
                best_results = results ##Para guardar el ganador
                
            print(f'Modelo ARIMA{order} | AIC: {results.aic:.4f}')
            
        except Exception as e:
            continue
            
    print("-" * 40)
    print(f"GANADOR: ARIMA{best_order}")
    
    print("\nParámetros del modelo ganador:")
    print(best_results.params) 
    try:
        #Intentamos como si fuera una Serie de Pandas con etiquetas
        intercepto = best_results.params['const']
    except:
        #Si es un arreglo de Numpy o no encuentra 'const', tomamos el primer valor
        intercepto = best_results.params[0]
    
    print(f"\nIntercepto (Drift): {intercepto:.6f}")
    
    if intercepto < 0:
        print("Interpretación: El modelo detectó una tendencia promedio BAJISTA (Drift negativo).")
    else:
        print("Interpretación: El modelo detectó una tendencia promedio ALCISTA (Drift positivo).")
        
    return best_order, best_results

mejor_p_q, resultados_win = buscar_mejor_arima(train_close["close_boxcox_diff"], max_p=4, max_q=4)

mejor_order = (0, 0, 0) ##En el trbajo es d= 1 porque el orden se halló con los datos diferenciados
model_final = ARIMA(train_close["close_boxcox_diff"], order=mejor_order).fit()
print(model_final.summary())

##Validamos supuestos

def validate_residuals(data: Union[pd.DataFrame, pd.Series], lags_lb=10):
    """
    Realiza pruebas de diagnóstico sobre los residuos:
    ARCH, Ljung-Box, Anderson-Darling (con p-valor) y t-student.
    """
    #Manejo de entrada univariada/multivariada
    if isinstance(data, pd.Series):
        resid_df = data.to_frame()
    else:
        resid_df = data.copy()
        
    results = []
    
    for col in resid_df.columns:
        series = resid_df[col].dropna()
        ##Anderson Darling
        ad_res = stats.anderson(series, dist='norm', method='interpolate')
        ad_stat = ad_res.statistic
        ad_p = ad_res.pvalue  
        is_normal = "Sí" if ad_p > 0.05 else "No"
        
        #Ljung box
        lb_test = acorr_ljungbox(series, lags=[lags_lb], return_df=True)
        lb_p = lb_test['lb_pvalue'].values[0]
        is_independent = "Sí" if lb_p > 0.05 else "No"
        
        #ARCH
        arch_res = het_arch(series)
        arch_p = arch_res[1] 
        is_homog = "Sí" if arch_p > 0.05 else "No"
        
        #t-student
        t_stat, t_p = stats.ttest_1samp(series, 0)
        is_mean_zero = "Sí" if t_p > 0.05 else "No"
        
        results.append({
            'Variable': col,
            'Normal?': is_normal,
            'Indep?': is_independent,
            'Homog?': is_homog,
            'Media 0?': is_mean_zero,
            'AD p-val': round(ad_p, 4),
            'LB p-val': round(lb_p, 4),
            'ARCH p-val': round(arch_p, 4),
            't p-val': round(t_p, 4)
        })
        
    return pd.DataFrame(results)

validate_residuals(pd.DataFrame(model_final.resid))

##Métricas y evaluación
forecast_diff = model_final.forecast(steps=len(test_close))

##Revertimos diferenciación

last_boxcox_train = train_close["train_boxcox"].iloc[train_size_close - 1]

#Reconstruimos la serie en escala Box-Cox usando suma acumulada
forecast_boxcox = last_boxcox_train + np.cumsum(forecast_diff)

##Reveertimos Boxcox
#Usamos el lambda_opt que guardamos al principio
forecast_final = inv_boxcox(forecast_boxcox, lambda_optim_close)

test_real = close['close'].iloc[train_size_close:train_size_close + len(test_close)].values

##Métricas en escala original
rmse = np.sqrt(mean_squared_error(test_real, forecast_final))
mae = mean_absolute_error(test_real, forecast_final)
mape = mean_absolute_percentage_error(test_real, forecast_final)
r2 = r2_score(test_real, forecast_final) 

print("-" * 30)
print(f"Métricas en Escala Real (EURUSD):")
print(f"RMSE: {rmse:.6f}")
print(f"MAE:  {mae:.6f}")
print(f"MAPE: {mape:.4%} (Porcentaje de error)")
print(f"R²:   {r2:.6f}") 
print("-" * 30)

full_index = close.index 
train_index = full_index[:train_size_close]
test_index = full_index[train_size_close:train_size_close + len(test_close)]

##Para graficar
##Obtenemos ajustes dentro de train
fitted_diff = model_final.fittedvalues

fitted_boxcox = train_close["train_boxcox"].iloc[:train_size_close].shift(1) + fitted_diff
fitted_final = inv_boxcox(fitted_boxcox, lambda_optim_close)


plt.figure(figsize=(12, 6))
plt.plot(full_index, close['close'], label='Observaciones reales', color='lightgray', alpha=0.8, linewidth=1.5)
plt.plot(train_index, fitted_final, label='Ajuste en entrenamiento', color='forestgreen', linestyle=':', alpha=0.7)
plt.plot(test_index, forecast_final, label='Predicción para prueba', color='darkorange', linewidth=2.5)
plt.axvline(x=full_index[train_size_close], color='red', linestyle='--', alpha=0.5, label='Inicio de prueba')
plt.title(f'Ajuste y predicciones del modelo ARIMA{mejor_order}', fontsize=15)
plt.xlabel('Tiempo', fontsize=12)
plt.ylabel('Precio de cierre (EURUSD)', fontsize=12)
plt.legend(loc='upper left', frameon=True)
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()

###Modelo VAR

##Función para aplicar boxcox a varias variables

def aplicar_boxcox(df, columns_to_transform):
    df_transformed = df.copy()
    lambdas = {}
    
    for col in columns_to_transform:
        if col in df_transformed.columns:
            df_transformed[col], lmbda = stats.boxcox(df_transformed[col])
            lambdas[col] = lmbda
            print(f"Variable '{col}' transformada (Box-Cox). Lambda: {lmbda:.4f}")
        else:
            print(f"Advertencia: La columna '{col}' no se encuentra en el DataFrame.")
            
    return df_transformed, lambdas

##Para diferenciación
def aplicar_diff(df, columns_to_diff, order=1):
    df_diff = df.copy()
    
    for col in columns_to_diff:
        if col in df_diff.columns:
            df_diff[col] = df_diff[col].diff(periods=order)
            print(f"Variable '{col}' diferenciada (orden {order}).")
        else:
            print(f"Advertencia: La columna '{col}' no se encuentra en el DataFrame.")
    return df_diff.dropna()


##Dividimos primero para evitar fuga de datos
train_size = int(len(price_data) * 0.80)
train_raw = price_data.iloc[:train_size].copy()
test_raw = price_data.iloc[train_size:].copy()

print(f"Observaciones totales: {len(price_data)}")
print(f"Entrenamiento: {len(train_raw)}")
print(f"Prueba: {len(test_raw)}")

##Verificamos estacionariedad y homoscedasticidad
stationarity_and_homoscedasticity_tests(train_raw)

##Hacemos las trasnformaciones necesarias basados en los test de significancia que hice
vars_to_boxcox = ["close", "VIX", "RSI"]
vars_to_diff = ["close", "DXY"]

train_box, dict_lanbda = aplicar_boxcox(train_raw, vars_to_boxcox)
print(f"Estos son los lambdas calculados en Train: \n {dict_lanbda}")

train_data = aplicar_diff(train_box, vars_to_diff)
print(f"Estos son los datos con diferenciados: \n {train_data}")

##Verificamos estacionariedad y homoscedasticidad 
stationarity_and_homoscedasticity_tests(train_data)

##Función para buscar el valor p óptimo
def grid_search_var(train, max_lags=15):
    """
    Realiza una búsqueda exhaustiva del orden de rezago p óptimo 
    basándose en el mejor AIC en el conjunto de entrenamiento.
    """
    best_aic = float('inf')
    best_p = 0
    results = []

    print(f"{'p':<5} | {'AIC':<12}")
    print("-" * 20)

    for p in range(1, max_lags + 1):
        try:
            model = VAR(train)
            model_fitted = model.fit(p)
            aic = model_fitted.aic
            
            results.append({'p': p, 'aic': aic})
            print(f"{p:<5} | {aic:<12.4f}")

            if aic < best_aic:
                best_aic = aic
                best_p = p
        except Exception as e:
            continue

    print("-" * 20)
    print(f"Orden óptimo por AIC: p = {best_p}")
    return best_p, results

p_opt, history = grid_search_var(
train= train_data, 
max_lags= 12
)

##Entrenamos el modelo con el p óptimo
var_model = VAR(train_data)
var_model = var_model.fit(p_opt)

##Sacamos el vector de medias 
delta = var_model.intercept
Phi_sum = np.sum(var_model.coefs, axis=0)
mu_hat = np.linalg.inv(np.eye(len(delta)) - Phi_sum) @ delta

print("Vector de medias incondicionales estimado:")
print(mu_hat)

#Las matrices de coeficientes (Phi_1, Phi_2, ..., Phi_10)
#var_model.coefs devuelve un array de forma (lags, k, k)
coefs_matrices = var_model.coefs
Phi_1 = coefs_matrices[0]
Phi_2 = coefs_matrices[1]

print("Matriz Phi_1:")
print(Phi_1)

print("Matriz Phi_2:")
print(Phi_2)

##Validamos residuos
validate_residuals(var_model.resid)

#Generar el pronóstico para el conjunto de prueba
#Necesitamos los últimos p valores del train para proyectar el futuro
forecast_input = train_data.values[-p_opt:]
forecast = var_model.forecast(y=forecast_input, steps=len(test_raw))

#Convertir a DataFrame
df_forecast = pd.DataFrame(forecast, index=test_raw.index, columns=train_data.columns)

def calculate_mape(true, pred):
    return np.mean(np.abs((true - pred) / true)) * 100

##Métricas de evaluación
targets = ["close", "VIX", "RSI"]
results_real = {}

for col in targets:
    #Inversión de entrenamiento
    fitted_transformed = var_model.fittedvalues[col]
    
    if col in vars_to_diff:
        anchor_train = train_box[col].shift(1).loc[fitted_transformed.index]
        fitted_boxcox = anchor_train + fitted_transformed
    else:
        fitted_boxcox = fitted_transformed
    
    lmbda = dict_lanbda.get(col)
    fitted_real = inv_boxcox(fitted_boxcox, lmbda) if lmbda is not None else fitted_boxcox
    
    #Inversión de prueba
    forecast_transformed = df_forecast[col]
    
    if col in vars_to_diff:
        last_val_train = train_box[col].iloc[-1]
        forecast_boxcox = last_val_train + forecast_transformed.cumsum()
    else:
        forecast_boxcox = forecast_transformed
        
    forecast_real = inv_boxcox(forecast_boxcox, lmbda) if lmbda is not None else forecast_boxcox
    
    #Cálculo de métricas
    y_true = price_data[col].loc[test_raw.index]
    
    results_real[col] = {
        'train_fit': fitted_real,
        'test_pred': forecast_real,
        'rmse': np.sqrt(mean_squared_error(y_true, forecast_real)),
        'mae': mean_absolute_error(y_true, forecast_real),
        'mape': calculate_mape(y_true, forecast_real),
        'r2': r2_score(y_true, forecast_real)
    }

#Tabla de métricas
print("\n" + "="*65)
print(f"{'Variable':<15} | {'RMSE':<10} | {'MAE':<10} | {'MAPE':<10} | {'R2':<10}")
print("-" * 65)
for col, m in results_real.items():
    print(f"{col:<15} | {m['rmse']:<10.4f} | {m['mae']:<10.4f} | {m['mape']:<10.2f}% | {m['r2']:<10.4f}")
print("="*65)

##Gráficos
titles = {"close": "EUR/USD Price", "VIX": "VIX Volatility Index", "RSI": "RSI Indicator"}

for col in targets:
    plt.figure(figsize=(12, 6))
    plt.plot(price_data[col].index, price_data[col], color='lightgray', alpha=0.6, label='Observaciones reales')
    plt.plot(results_real[col]['train_fit'].index, results_real[col]['train_fit'], 
            color='forestgreen', linestyle=':', label='Ajuste en entrenamiento')
    plt.plot(results_real[col]['test_pred'].index, results_real[col]['test_pred'], 
            color='darkorange', linewidth=2.5, label='Predicción para prueba')
    plt.axvline(x=test_raw.index[0], color='red', linestyle='--', alpha=0.5, label='Inicio de prueba')
    plt.title(f'Resultados del Modelo VAR: {titles[col]}', fontsize=15)
    plt.xlabel('Tiempo', fontsize=12)
    plt.ylabel('Valor', fontsize=12)
    plt.legend(loc='upper left', frameon=True)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show() 

###Ajustamos el modelo LSTM univariado

##Función para dividir en entrenamiento/val/prueba
def train_test_split2(dataframe, tr_size=0.8, ts_size=0.2, use_validation=False, vl_size=0.1):
    """
    Divide un DataFrame en conjuntos de entrenamiento, validación (opcional) y prueba, retornando además los índices de inicio de cada partición.
    """
    N = dataframe.shape[0]
    Ntrain = int(tr_size * N)

    if use_validation:
        Nval = int(vl_size * N)
        Ntst = N - Ntrain - Nval

        train = dataframe[:Ntrain]
        val = dataframe[Ntrain:Ntrain + Nval]
        test = dataframe[Ntrain + Nval:]

        return train, val, test, 0, Ntrain, Ntrain + Nval  #índices reales
    else:
        Ntst = int(ts_size * N)
        Ntrain = N - Ntst

        train = dataframe[:Ntrain]
        test = dataframe[Ntrain:]

        return train, test, 0, Ntrain  #índices reales

train_lstm, val_lstm, test_lstm, start_idx_train_lstm, start_idx_val_lstm, start_idx_test_lstm = train_test_split2(
    price_data["close"], tr_size = 0.7, ts_size=0.2, vl_size=0.1, use_validation=True
)
print(f"Tamaño del set de train: {train_lstm.shape}")
print(f"tamaño del set prueba: {test_lstm.shape}")
print(f"Tamaño del set de val: {val_lstm.shape}")

##Escalamos
scaler_lstm = StandardScaler()
train_scaled_lstm = scaler_lstm.fit_transform(train_lstm.values.reshape(-1, 1))

##Aplicamos escalado con los parametros de train
val_scaled_lstm = scaler_lstm.transform(val_lstm.values.reshape(-1, 1))
test_scaled_lstm = scaler_lstm.transform(test_lstm.values.reshape(-1, 1))

print(f"Estos son los datos escalados de train: \n{train_scaled_lstm}")
print(f"Estos son los datos escalados de val: \n {val_scaled_lstm}")
print(f"Estos son los datos escalados de test: \n {test_scaled_lstm}")

print(f"Esta es la media del scaler: {scaler_lstm.mean_}")
print(f"Esta es la varianza de scaler: {scaler_lstm.var_}")

##Función para crear secuencias

def create_multivariate_sequences2(data, time_steps, target_indices=None, start_index=0):
    """
    Crea secuencias para modelos LSTM, devolviendo también los índices absolutos respecto al dataset original.

    Args:
        data: Array numpy 2D de forma (muestras, features).
        time_steps: Número de pasos temporales en cada secuencia de entrada.
        target_indices: Índices de las columnas a predecir. Puede ser:
            - None: Todas las columnas (multivariado).
            - int: Una sola columna.
            - list: Múltiples columnas.
        start_index: Desplazamiento del índice inicial respecto al dataset original.
    Returns:
        X: Array de entrada de forma (n_samples, time_steps, n_features).
        y: Array objetivo de forma (n_samples, n_targets).
        y_indices: Índices absolutos en el dataset original.
    """
    #Validaciones
    if len(data.shape) != 2:
        raise ValueError("El array 'data' debe ser 2D (muestras, features).")
    if time_steps <= 0:
        raise ValueError("time_steps debe ser >= 1.")
    
    #Convertir target_indices a lista si es un entero
    if isinstance(target_indices, int):
        target_indices = [target_indices]

    X, y, y_indices = [], [], []
    for i in range(len(data) - time_steps):
        X.append(data[i : i + time_steps])
        
        #Para seleccionar targets
        if target_indices is None:
            target = data[i + time_steps]  #Todas las columnas
        else:
            target = data[i + time_steps, target_indices]
        
        y.append(target)
        y_indices.append(start_index + i + time_steps)  #Índice absoluto

    X = np.array(X)
    y = np.array(y)
    y_indices = np.array(y_indices)

    #Para asegurar la salida 2D
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    return X, y, y_indices

##Obtenemos los target
var_pred_lstm = list(price_data.columns)
target_cols_lstm = ["close"]
target_indices_lstm = [var_pred_lstm.index(col) for col in target_cols_lstm]
print(f"Este es el target: {target_indices_lstm}")

##Obtenemos las secuencias
X_train_lstm, y_train_lstm, y_indices_train_lstm  = create_multivariate_sequences2(
    train_scaled_lstm, 
    time_steps=60, 
    target_indices = target_indices_lstm, 
    start_index=start_idx_train_lstm
)

X_val_lstm, y_val_lstm, y_indices_val_lstm = create_multivariate_sequences2(
    val_scaled_lstm, 
    time_steps=60, 
    target_indices=target_indices_lstm,
    start_index=start_idx_val_lstm
)

X_test_lstm, y_test_lstm, y_indices_test_lstm = create_multivariate_sequences2(
    test_scaled_lstm, 
    time_steps=60,
    target_indices=target_indices_lstm,
    start_index=start_idx_test_lstm
)

print(X_train_lstm.shape, y_train_lstm.shape)
print(X_test_lstm.shape, y_test_lstm.shape)

##Creamos los tensores

X_train_lstm = torch.tensor(X_train_lstm, dtype=torch.float32)
y_train_lstm = torch.tensor(y_train_lstm, dtype=torch.float32)

X_val_lstm = torch.tensor(X_val_lstm, dtype=torch.float32)
y_val_lstm = torch.tensor(y_val_lstm, dtype=torch.float32)

X_test_lstm = torch.tensor(X_test_lstm, dtype=torch.float32)
y_test_lstm = torch.tensor(y_test_lstm, dtype=torch.float32)

##Arquitectura LSTM univariado
class LSTMSsimple1(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(LSTMSsimple1, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                          batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:,-1,:])
        return out

##Definimos funciones para hacer el grid search
##Para reproducibilidad
def set_seeds(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

##Función de entrenamiento

def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs=20,
    early_stopping=True,
    patience=5,
    min_delta=1e-6,
    scheduler=None,
    device=None,
    seed=1,
    verbose=True
):
    """
    Entrena el modelo con early stopping robusto y checkpoint del mejor estado.

    Args:
        model         : Modelo PyTorch a entrenar.
        train_loader  : DataLoader de entrenamiento.
        val_loader    : DataLoader de validación (puede ser None).
        criterion     : Función de pérdida (ej. nn.MSELoss()).
        optimizer     : Optimizador (ej. optim.Adam).
        num_epochs    : Máximo de épocas.
        early_stopping: Si True, detiene el entrenamiento cuando la pérdida
                        de validación no mejora tras `patience` épocas.
        patience      : Épocas a esperar sin mejora antes de detener.
        min_delta     : Mejora mínima en val_loss para considerarse una
                        mejora real. Evita que ruido numérico reinicie el
                        contador (ej. 1e-6).
        scheduler     : LR scheduler (como ReduceLROnPlateau).
        device        : 'cuda', 'cpu' o None (autodetecta).
        seed          : Semilla para reproducibilidad.
        verbose       : Imprime información durante el entrenamiento.

    Returns:
        model         : Modelo con los pesos del mejor epoch restaurados.
        train_losses  : Lista de pérdidas promedio por epoch (entrenamiento).
        val_losses    : Lista de pérdidas promedio por epoch (validación),
                        o None si no se proporcionó val_loader.
        best_epoch    : Epoch donde se encontró el mejor val_loss.
    """
    set_seeds(seed)

    ##Definfimos el dispositivo
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_losses  = []
    val_losses    = []
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None
    best_epoch    = 1

    for epoch in range(num_epochs):
        epoch_start = time.time()

        #Entrenamiento
        model.train()
        train_loss  = 0.0
        batch_count = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            if y_batch.dim() == 1:
                y_batch = y_batch.unsqueeze(1)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss  += loss.item()
            batch_count += 1

            if verbose and (batch_count % 10 == 0 or batch_count == len(train_loader)):
                print(f"  Época {epoch+1}/{num_epochs} | "
                      f"Lote {batch_count}/{len(train_loader)} | "
                      f"Pérdida: {loss.item():.6f}")

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        #Validación 
        if val_loader is not None:
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for X_val, y_val in val_loader:
                    X_val = X_val.to(device)
                    y_val = y_val.to(device)
                    if y_val.dim() == 1:
                        y_val = y_val.unsqueeze(1)
                    outputs  = model(X_val)
                    val_loss += criterion(outputs, y_val).item()

            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            improved = avg_val_loss < (best_val_loss - min_delta)
            epoch_time = time.time() - epoch_start

            if verbose:
                tag = "--> nuevo mejor" if improved else f" (paciencia {patience_counter + (0 if improved else 1)}/{patience})"
                print(f"Época {epoch+1}/{num_epochs} | "
                      f"Train: {avg_train_loss:.6f} | "
                      f"Val: {avg_val_loss:.6f}{tag} | "
                      f"{epoch_time:.1f}s")

            #Scheduler
            if scheduler is not None:
                scheduler.step(avg_val_loss)

            #Early stopping
            if early_stopping:
                if improved:
                    best_val_loss  = avg_val_loss
                    best_epoch  = epoch + 1
                    patience_counter = 0
                    #deepcopy: congela los pesos en este instante
                    best_model_state = copy.deepcopy(model.state_dict())
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    if verbose:
                        print(f"\n⛔ Early stopping activado en época {epoch+1}. "
                              f"Mejor época: {best_epoch} "
                              f"(val_loss={best_val_loss:.6f})")
                    break
            else:
                #Sin early stopping: igual guardamos el mejor checkpoint
                if improved:
                    best_val_loss    = avg_val_loss
                    best_epoch       = epoch + 1
                    best_model_state = copy.deepcopy(model.state_dict())

        else:
            if verbose:
                print(f"Época {epoch+1}/{num_epochs} | "
                      f"Train: {avg_train_loss:.6f}")

    #Para restarurar el mejor estado
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        if verbose and val_loader is not None:
            print(f"Pesos restaurados al mejor epoch: {best_epoch} "
                  f"(val_loss={best_val_loss:.6f})\n")

    return model, train_losses, (val_losses if val_loader is not None else None), best_epoch

##Para validar resultados

def plot_validation_results(val_loss_history):
    """Visualiza los resultados de validación para distintas configuraciones."""
    plt.figure(figsize=(15, 10))

    #Batch size
    plt.subplot(2, 2, 1)
    for bs in sorted(set(r["params"]["batch_size"] for r in val_loss_history)):
        losses = [r["best_val_loss"] for r in val_loss_history
                  if r["params"]["batch_size"] == bs]
        plt.hist(losses, alpha=0.6, label=f"batch={bs}")
    plt.xlabel("Mejor val_loss"); plt.ylabel("Frecuencia")
    plt.title("Distribución por batch_size"); plt.legend()

    #Learning rate
    plt.subplot(2, 2, 2)
    for lr in sorted(set(r["params"]["learning_rate"] for r in val_loss_history)):
        losses = [r["best_val_loss"] for r in val_loss_history
                  if r["params"]["learning_rate"] == lr]
        plt.hist(losses, alpha=0.6, label=f"lr={lr}")
    plt.xlabel("Mejor val_loss"); plt.ylabel("Frecuencia")
    plt.title("Distribución por learning_rate"); plt.legend()

    #hidden_size 
    plt.subplot(2, 2, 3)
    hs_key = next((k for k in ("hidden_size", "lstm_hidden_size")
                   if k in val_loss_history[0]["params"]), None)
    if hs_key:
        hs     = [r["params"][hs_key] for r in val_loss_history]
        losses = [r["best_val_loss"]  for r in val_loss_history]
        plt.scatter(hs, losses, alpha=0.6)
        plt.xlabel(hs_key); plt.ylabel("Mejor val_loss")
        plt.title(f"Pérdida vs. {hs_key}")

    #num_layers
    plt.subplot(2, 2, 4)
    if "num_layers" in val_loss_history[0]["params"]:
        nl     = [r["params"]["num_layers"] for r in val_loss_history]
        losses = [r["best_val_loss"]        for r in val_loss_history]
        plt.scatter(nl, losses, alpha=0.6)
        plt.xlabel("num_layers"); plt.ylabel("Mejor val_loss")
        plt.title("Pérdida vs. num_layers")

    plt.tight_layout()
    plt.show()


##Función para búsqueda de hierparámetros

def find_best_lstm_model(
    model_class,
    param_grid,
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    batch_sizes=[32, 64],
    learning_rates=[0.001, 0.005],
    train_model_fn=None,
    criterion=None,
    num_epochs=20,
    early_stopping=True,
    patience=5,
    min_delta=1e-6,
    scheduler_patience=None,
    scheduler_factor=0.5,
    use_scheduler=True,
    device=None,
    verbose=True,
    seed=1
):
    """
    Busca la mejor combinación de hiperparámetros para un modelo LSTM.

    Args:
        model_class       : Clase del modelo 
        param_grid        : Diccionario de hiperparámetros del modelo.
                            Ejemplo: {'hidden_size': [32, 64], 'num_layers': [1, 2]}.
        X_train           : Tensor de entrenamiento, shape (N, T, F).
        y_train           : Tensor de etiquetas de entrenamiento.
        X_val             : Tensor de validación (opcional).
        y_val             : Tensor de etiquetas de validación (opcional).
        batch_sizes       : Lista de batch sizes a probar.
        learning_rates    : Lista de learning rates a probar.
        train_model_fn    : Función de entrenamiento personalizada. Si None,
                            se usa `train_model` de este módulo.
        criterion         : Función de pérdida. Por defecto: nn.MSELoss().
        num_epochs        : Máximo de épocas por configuración.
        early_stopping    : Activa el early stopping.
        patience          : Épocas a esperar sin mejora (early stopping).
        min_delta         : Umbral mínimo de mejora para el early stopping.
        scheduler_patience: Patience del scheduler. Si None, se calcula como
                            max(2, patience // 2) para ser más agresivo que
                            el early stopping.
        scheduler_factor  : Factor de reducción del LR (default 0.5 = mitad).
        use_scheduler     : Activa el scheduler ReduceLROnPlateau.
        device            : 'cuda', 'cpu' o None (autodetecta).
        verbose           : Imprime info detallada.
        seed              : Semilla para reproducibilidad.

    Returns:
        best_model       : Mejor modelo encontrado (pesos del mejor epoch).
        best_params      : Diccionario con los mejores hiperparámetros.
        val_loss_history : Lista de dicts con resultados por configuración
                           (None si no se usó validación).
        best_train_losses: Lista de train losses del mejor modelo.
        best_val_losses  : Lista de val losses del mejor modelo (o None).
    """
    if criterion is None:
        criterion = nn.MSELoss()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if train_model_fn is None:
        train_model_fn = train_model

    #Patience del scheduler menor que el del early stopping
    if scheduler_patience is None:
        scheduler_patience = max(2, patience // 2)

    set_seeds(seed)

    #Ajustar forma de y
    if isinstance(y_train, torch.Tensor) and y_train.dim() == 1:
        y_train = y_train.unsqueeze(1)
    if y_val is not None and isinstance(y_val, torch.Tensor) and y_val.dim() == 1:
        y_val = y_val.unsqueeze(1)

    #Construir grid completo
    full_param_grid = []
    for params in ParameterGrid(param_grid):
        for batch_size in batch_sizes:
            for lr in learning_rates:
                full_params = params.copy()
                full_params["batch_size"]     = batch_size
                full_params["learning_rate"]  = lr
                full_param_grid.append(full_params)

    total = len(full_param_grid)
    if verbose:
        print(f"Probando {total} combinaciones de hiperparámetros "
              f"(device: {device})\n{'─'*60}")

    best_loss         = float("inf")
    best_model        = None
    best_params       = None
    val_loss_history  = []
    best_train_losses = None
    best_val_losses   = None

    #Iterar configuraciones
    for params in tqdm(full_param_grid, desc="Evaluando hiperparámetros"):
        torch.manual_seed(seed)
        np.random.seed(seed)

        #Extraer batch_size y lr sin mutar el dict 
        batch_size = params["batch_size"]
        lr         = params["learning_rate"]
        model_params = {k: v for k, v in params.items()
                        if k not in ("batch_size", "learning_rate", "input_size")}

        if verbose:
            print(f"\n▶ Config: {params}")

        #Dataloaders
        train_dataset = TensorDataset(X_train, y_train)
        train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_loader = None
        if X_val is not None and y_val is not None:
            val_dataset = TensorDataset(X_val, y_val)
            val_loader  = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        #Modelo
        input_size = X_train.shape[2]
        model = model_class(input_size=input_size, **model_params)

        #Optimizador y scheduler
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = None
        if use_scheduler and val_loader is not None:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=scheduler_factor,
                patience=scheduler_patience   
            )

        #Entrenar 
        trained_model, train_losses, val_losses, best_epoch = train_model_fn(
            model, train_loader, val_loader, criterion, optimizer,
            num_epochs=num_epochs,
            early_stopping=early_stopping,
            patience=patience,
            min_delta=min_delta,
            scheduler=scheduler,
            device=device,
            seed=seed,
            verbose=verbose
        )

        #Registrar resultados
        if val_loader is not None and val_losses:
            current_best_val = min(val_losses)

            config_result = {
                "params":   params.copy(), #dict completo con bs y lr
                "final_val_loss": val_losses[-1],
                "best_val_loss":  current_best_val,
                "best_epoch":     best_epoch,
                "epochs_trained": len(val_losses),
                "train_losses":   train_losses,
                "val_losses":     val_losses,
            }
            val_loss_history.append(config_result)

            if current_best_val < best_loss:
                best_loss    = current_best_val
                best_model  = copy.deepcopy(trained_model)
                best_params  = params.copy()
                best_train_losses = train_losses
                best_val_losses   = val_losses

        else:
            if train_losses and train_losses[-1] < best_loss:
                best_loss = train_losses[-1]
                best_model = copy.deepcopy(trained_model)
                best_params  = params.copy()
                best_train_losses = train_losses
                best_val_losses = None

    #Resultados finales
    if verbose:
        print(f"\n{'═'*60}")
        if val_loss_history:
            sorted_results = sorted(val_loss_history, key=lambda x: x["best_val_loss"])
            print("Top 5 configuraciones:")
            for i, r in enumerate(sorted_results[:5]):
                print(f"  [{i+1}] val_loss={r['best_val_loss']:.6f} "
                      f"(época {r['best_epoch']}/{r['epochs_trained']}) | {r['params']}")
            print(f"\nMejor configuración: {best_params}")
            print(f"   Mejor val_loss:      {best_loss:.6f}")

            plot_validation_results(val_loss_history)
        else:
            print(f"Mejor configuración: {best_params}")
            print(f"   Mejor train_loss:{best_loss:.6f}")

    return (
        best_model,
        best_params,
        val_loss_history if val_loss_history else None,
        best_train_losses,
        best_val_losses
    )

##Para buscar los mejores hiperparametros
param_grid_simple = {
    'hidden_size': [32, 64],
    'num_layers': [1, 2],
    'dropout': [0.0, 0.2]}

best_model_LSTM, best_params_LSTM, history_LSTM, train_losses_LSTM, val_losses_LSTM = find_best_lstm_model(
    model_class=LSTMSsimple1, 
    param_grid=param_grid_simple, 
    X_train=X_train_lstm, y_train = y_train_lstm, 
    X_val = X_val_lstm, y_val = y_val_lstm,
    batch_sizes=[32, 64],
    learning_rates=[0.001, 0.005],
    num_epochs= 12, 
    early_stopping=True,
    patience=5, 
    min_delta=1e-5, 
    scheduler_patience=2, 
    criterion=nn.MSELoss()
)

##Concatenamos entrenamiento + validación

X_train_full_LSTM = torch.cat([X_train_lstm, X_val_lstm], dim = 0)
y_train_full_LSTM = torch.cat([y_train_lstm, y_val_lstm], dim = 0)
y_indices_train_full_lstm = np.concatenate([y_indices_train_lstm, y_indices_val_lstm])

print(len(X_train_full_LSTM))
print(len(X_train_lstm) + len(X_val_lstm)) 

##Incluimos el input size en el param grid
best_params_lstm = {
    'input_size': best_model_LSTM.lstm.input_size,
    'hidden_size': best_model_LSTM.lstm.hidden_size,
    'num_layers': best_model_LSTM.lstm.num_layers,
    'dropout': best_model_LSTM.lstm.dropout if hasattr(best_model_LSTM.lstm, 'dropout') else 0.0,
    'learning_rate': 0.005, ##Estos hacen parte de la mejor configuración, solo que los agregué manualmente
    'batch_size': 32
}
print(best_params_lstm)

##Función para reentrenar

def retrain_best_model(
    model_class,
    best_params,
    X_trainval,
    y_trainval,
    X_test=None,
    y_test=None,
    criterion=None,
    num_epochs=20,
    use_scheduler=True,
    device=None,
    seed=1,
    verbose=True
):
    """
    Reentrena el mejor modelo con train+val combinados y evalúa en test.

    Sin conjunto de validación no es posible hacer early stopping, por eso
    se usa CosineAnnealingLR para ir reduciendo el LR de forma suave a lo
    largo de todas las épocas (simula el efecto de un scheduler sin val).

    Args:
        model_class  : Clase del modelo.
        best_params  : Dict con los mejores hiperparámetros. NO se muta.
                       Debe contener 'batch_size' y 'learning_rate'.
        X_trainval   : Tensor combinado (train + val), shape (N, T, F).
        y_trainval   : Tensor de etiquetas combinado (train + val).
        X_test       : Tensor de test (opcional). Si None, no se evalúa.
        y_test       : Tensor de etiquetas de test (opcional).
        criterion    : Función de pérdida. Por defecto: nn.MSELoss().
        num_epochs   : Épocas de reentrenamiento.
        use_scheduler: Si True, usa CosineAnnealingLR para reducir el LR
                       gradualmente a lo largo de las épocas.
        device       : 'cuda', 'cpu' o None (autodetecta).
        seed         : Semilla para reproducibilidad.
        verbose      : Imprime progreso.

    Returns:
        model        : Modelo reentrenado.
        train_losses : Lista de pérdidas promedio por época.
        predictions  : Tensor con predicciones sobre X_test,
                       o None si no se proporcionó X_test.
        test_loss    : Pérdida promedio en test, o None si no hay X_test.
    """
    if criterion is None:
        criterion = nn.MSELoss()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seeds(seed)

    #Extraer params SIN mutar el dict original
    batch_size    = best_params["batch_size"]
    learning_rate = best_params["learning_rate"]
    model_params  = {k: v for k, v in best_params.items()
                     if k not in ("batch_size", "learning_rate", "input_size")}

    #Ajustar forma de y
    if isinstance(y_trainval, torch.Tensor) and y_trainval.dim() == 1:
        y_trainval = y_trainval.unsqueeze(1)
    if y_test is not None and isinstance(y_test, torch.Tensor) and y_test.dim() == 1:
        y_test = y_test.unsqueeze(1)

    #DataLoaders
    trainval_dataset = TensorDataset(X_trainval, y_trainval)
    trainval_loader  = DataLoader(trainval_dataset, batch_size=batch_size, shuffle=True)

    test_loader = None
    if X_test is not None and y_test is not None:
        test_dataset = TensorDataset(X_test, y_test)
        test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    #Modelo
    input_size = X_trainval.shape[2]
    model = model_class(input_size=input_size, **model_params).to(device)

    #Optimizador y scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = None
    if use_scheduler:
        #CosineAnnealingLR: reduce LR suavemente hasta casi 0 en la última época
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=learning_rate * 0.01
        )

    if verbose:
        print(f"Reentrenando con train+val combinados | "
              f"device: {device} | épocas: {num_epochs}")
        print(f"   Params: {best_params}")
        print(f"{'─'*60}")

    #Entrenamiento
    train_losses = []

    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        epoch_loss  = 0.0

        for X_batch, y_batch in trainval_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            if y_batch.dim() == 1:
                y_batch = y_batch.unsqueeze(1)

            output = model(X_batch)
            loss   = criterion(output, y_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        if scheduler is not None:
            scheduler.step()

        avg_loss = epoch_loss / len(trainval_loader)
        train_losses.append(avg_loss)

        if verbose:
            current_lr = optimizer.param_groups[0]["lr"]
            elapsed    = time.time() - epoch_start
            print(f"  Época {epoch+1:>3}/{num_epochs} | "
                  f"Pérdida: {avg_loss:.6f} | "
                  f"LR: {current_lr:.2e} | "
                  f"{elapsed:.1f}s")

    #Evaluación en test
    predictions = None
    test_loss   = None

    if test_loader is not None:
        model.eval()
        preds_list = []
        total_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                if y_batch.dim() == 1:
                    y_batch = y_batch.unsqueeze(1)

                output = model(X_batch)
                total_loss += criterion(output, y_batch).item()
                preds_list.append(output.cpu())

        predictions = torch.cat(preds_list, dim=0)
        test_loss   = total_loss / len(test_loader)

        if verbose:
            print(f"\nPérdida en test: {test_loss:.6f}")

    if verbose:
        print(f"\nReentrenamiento completado.")

    return model, train_losses, predictions, test_loss

##Reentrenamos el mejor modelo
lstm_model, train_losses_lstm, predictions_lstm, test_loss_lstm = retrain_best_model(
    model_class = LSTMSsimple1, 
    best_params = best_params_lstm,
    X_trainval = X_train_full_LSTM,
    y_trainval = y_train_full_LSTM,
    X_test = X_test_lstm,
    y_test = y_test_lstm,
    num_epochs = 12, 
    criterion = nn.MSELoss(),
    use_scheduler = True, 
    seed = 1
)

##Para visualizar métricas y graficos

def predict_and_evaluate_deep(
    model,
    X_train_full, y_train_full, y_indices_train_full,
    X_test, y_test, y_indices_test,
    scaler,
    target_columns,
    target_indices,
    date_index,
    original_data,         
    figsize=(14, 5)
):
    """
    Genera predicciones, invierte el escalado, grafica y calcula métricas
    para modelos profundos multivariados (sin Box-Cox).

    Parámetros:
    -----------
    model                : Modelo PyTorch entrenado
    X_train_full         : Tensor de entrada train+val
    y_train_full         : Tensor objetivo train+val
    y_indices_train_full : Índices absolutos de y_train_full
    X_test               : Tensor de entrada test
    y_test               : Tensor objetivo test
    y_indices_test       : Índices absolutos de y_test
    scaler               : StandardScaler ajustado sobre train original
    target_columns       : Lista de nombres de variables objetivo
    target_indices       : Lista de índices de columna en el dataframe original
    date_index           : Índice de fechas del dataframe original 
    figsize              : Tamaño de figura por variable
    
    Retorna:
    --------
    results : dict con predicciones y métricas por variable
    """

    model.eval()
    n_features = scaler.n_features_in_

    #Predicciones en escala estandarizada
    with torch.no_grad():
        train_pred_scaled = model(X_train_full).cpu().numpy()
        test_pred_scaled  = model(X_test).cpu().numpy()

    y_train_np = y_train_full.cpu().numpy()
    y_test_np  = y_test.cpu().numpy()

    #Funcion para invertir el escalado 
    def inverse_scale(values_1d, col_idx):
        """Reconstruye matriz completa de ceros e invierte solo la columna col_idx."""
        dummy = np.zeros((len(values_1d), n_features))
        dummy[:, col_idx] = values_1d
        return scaler.inverse_transform(dummy)[:, col_idx]

    #Funcion para calcular las metricas
    def calc_metrics(y_true, y_pred):
        mask  = np.isfinite(y_true) & np.isfinite(y_pred)
        y_t   = y_true[mask]
        y_p   = y_pred[mask]
        eps   = 1e-10
        y_safe = np.where(np.abs(y_t) < eps, eps, y_t)

        mse   = np.mean((y_t - y_p) ** 2)
        rmse  = np.sqrt(mse)
        mae   = np.mean(np.abs(y_t - y_p))
        mape  = np.mean(np.abs((y_t - y_p) / y_safe)) * 100
        r2    = r2_score(y_t, y_p)

        return {"MSE": mse, "RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}

    #Estilos de grafico
    plt.style.use('seaborn-v0_8-darkgrid')
    colors = {
        'actual': '#999999',   #gris
        'train': '#1a6e1a',   #verde punteado (ajuste)
        'test': '#E67D22',   #naranja (predicción test)
        'vline': '#E53935'    #rojo vertical
    }

    results = {}

    #Procesar cada variable objetivo
    for i, (col_name, col_idx) in enumerate(zip(target_columns, target_indices)):

        # Invertir escalado
        train_pred_orig = inverse_scale(train_pred_scaled[:, i], col_idx)
        test_pred_orig  = inverse_scale(test_pred_scaled[:, i],  col_idx)
        y_train_orig    = inverse_scale(y_train_np[:, i],         col_idx)
        y_test_orig     = inverse_scale(y_test_np[:, i],          col_idx)

        #Métricas
        train_metrics = calc_metrics(y_train_orig, train_pred_orig)
        test_metrics  = calc_metrics(y_test_orig,  test_pred_orig)

        results[col_name] = {
            'train_pred'    : train_pred_orig,
            'test_pred'     : test_pred_orig,
            'y_train_orig'  : y_train_orig,
            'y_test_orig'   : y_test_orig,
            'train_metrics' : train_metrics,
            'test_metrics'  : test_metrics
        }

        #Imprimir métricas
        print(f"\n{'='*55}")
        print(f"  VARIABLE: {col_name}")
        print(f"{'='*55}")
        for split, m in [('TRAIN', train_metrics), ('TEST', test_metrics)]:
            print(f"  {split}:")
            for k, v in m.items():
                print(f"    {k:>6}: {v:.6f}")

        #Graficar
        fig, ax = plt.subplots(figsize=figsize, dpi=100)

        #Serie real completa (fondo gris)
        ax.plot(date_index,
                original_data.iloc[:, col_idx].values, 
                color=colors['actual'], linewidth=1.2,
                alpha=0.6, label=f'Precio real', zorder=1)

        #Ajuste en entrenamiento
        train_dates = date_index[y_indices_train_full]
        ax.plot(train_dates, train_pred_orig,
                color=colors['train'], linewidth=1.0,
                linestyle='dotted', label='Ajuste en entrenamiento',
                alpha=0.85, zorder=2)

        #Línea vertical inicio test
        test_start_date = date_index[y_indices_test[0]]
        ax.axvline(x=test_start_date, color=colors['vline'],
                   linestyle='--', linewidth=1.5,
                   label='Inicio de prueba', zorder=3)

        #Predicción test
        test_dates = date_index[y_indices_test]
        ax.plot(test_dates, test_pred_orig,
                color=colors['test'], linewidth=2.0,
                linestyle='-', label='Predicción para prueba',
                alpha=1.0, zorder=4)

        #Formato
        ax.set_title(f'Ajuste y predicciones del modelo — {col_name}',
                     fontsize=14, fontweight='bold', pad=15, color='#2C3E50')
        ax.set_xlabel('Tiempo', fontsize=12, labelpad=8)
        ax.set_ylabel(f'Precio de cierre ({col_name})', fontsize=12, labelpad=8)

        ax.legend(loc='upper right', frameon=True, fancybox=True,
                  shadow=True, fontsize=10, framealpha=0.95)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', labelsize=10)
        ax.grid(True, alpha=0.25, linewidth=0.7)

        plt.tight_layout()
        plt.show()

    return results

##Resultados del modelo LSTM univariado
results_lstm = predict_and_evaluate_deep(
    model               = lstm_model,
    X_train_full        = X_train_full_LSTM,
    y_train_full        = y_train_full_LSTM,
    y_indices_train_full= y_indices_train_full_lstm,
    X_test              = X_test_lstm,
    y_test              = y_test_lstm,
    y_indices_test      = y_indices_test_lstm,
    scaler              = scaler_lstm,
    target_columns      = ['close'],
    target_indices      = target_indices_lstm,     
    date_index          = price_data.index,
    original_data=price_data,
    figsize=(12, 6)
)

###Modelos profundos multivariados sin régimen
train_deep, val_deep, test_deep, start_idx_train, start_idx_val, start_idx_test = train_test_split2(
    price_data, tr_size=0.7, ts_size=0.2, vl_size=0.1, use_validation=True
)
print(f"Tamaño del set de train: {train_deep.shape}")
print(f"Tamaño del set de prueba: {test_deep.shape}")
print(f"Tamaño del set de val: {val_deep.shape}")

scaler1 = StandardScaler()
train_scaled_deep = scaler1.fit_transform(train_deep)
##Aplicamos escalado con los parametros de train
val_scaled_deep = scaler1.transform(val_deep)
test_scaled_deep  = scaler1.transform(test_deep)

print(f"Esta son los datos escalados de train: \n {train_scaled_deep} ")
print(f"Estos son los datos escalados de val: \n {val_scaled_deep}")
print(f"Estos son los datos escalados de test: \n {test_scaled_deep}")

print(f"Esta es la media dl scaler: {scaler1.mean_}")
print(f"Esta es la varianza del scaler: {scaler1.var_}")

##Seleccionamos las variables que queremos proedecir 
var_pred_deep = list(price_data.columns)
target_cols = ["close", "VIX", "RSI"]
target_indices = [var_pred_deep.index(col) for col in target_cols]
print(f"Estos son los target: {target_indices}")

X_train, y_train, y_indices_train  = create_multivariate_sequences2(
    train_scaled_deep, 
    time_steps=60, 
    target_indices = target_indices, 
    start_index=start_idx_train
)

X_val, y_val, y_indices_val = create_multivariate_sequences2(
    val_scaled_deep, 
    time_steps=60, 
    target_indices=target_indices,
    start_index=start_idx_val
)

X_test, y_test, y_indices_test = create_multivariate_sequences2(
    test_scaled_deep, 
    time_steps=60,
    target_indices=target_indices,
    start_index=start_idx_test
)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

X_train = torch.tensor(X_train, dtype = torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

X_val = torch.tensor(X_val, dtype = torch.float32)
y_val = torch.tensor(y_val, dtype = torch.float32)

X_test = torch.tensor(X_test, dtype = torch.float32)
y_test = torch.tensor(y_test, dtype = torch.float32)

##Creamos las arquitecturas 

###LSTM SIMPLE MULTIVARIADO
class LSTMSsimple(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(LSTMSsimple, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                          batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 3)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:,-1,:])
        return out

#Modelo CNNLSTM simple
class CNNLSTM(nn.Module): 
    def __init__(self, input_size, cnn_out_channels, kernel_size, lstm_hidden_size,
                 num_layers, dropout):
        super(CNNLSTM, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=input_size, out_channels= cnn_out_channels, 
                                kernel_size=kernel_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size = cnn_out_channels, hidden_size=lstm_hidden_size, 
                            num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(lstm_hidden_size, 3)
    def forward(self, x): 
        x = x.permute(0,2,1) #reshape a [batch_size, input_size, seq_len] pra conv1d
        x = self.relu(self.conv1d(x))
        x = x.permute(0,2,1) #Volver a [batch_size, input_size, seq_len] para LSTM
        output, _ = self.lstm(x)
        return self.fc(output[:, -1, :])

##Modelo CNNSLSTM con mecanismo de atención
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1)
    def forward(self, lstm_output): #lstm_output: [batch, seq_len, hidden]
        weights = torch.softmax(self.attn(lstm_output), dim = 1)
        context = torch.sum(weights*lstm_output, dim = 1)
        return context
class CNN_LSTMBi_Attention(nn.Module): 
    def __init__(self, input_size, cnn_out_channels, lstm_hidden_size, num_layers, dropout):
        super(CNN_LSTMBi_Attention, self).__init__()
        self.conv3 = nn.Conv1d(in_channels = input_size, out_channels = cnn_out_channels, kernel_size = 3, padding = 1)
        self.conv5 = nn.Conv1d(in_channels = input_size, out_channels = cnn_out_channels, kernel_size = 5, padding = 2)
        self.conv7 = nn.Conv1d(in_channels = input_size, out_channels = cnn_out_channels, kernel_size = 7, padding = 3)
        #Normalización 
        self.bn = nn.BatchNorm1d(num_features = cnn_out_channels*3)
        #LSTM bidireccional
        self.lstm = nn.LSTM(input_size = cnn_out_channels*3, hidden_size = lstm_hidden_size,
                            num_layers = num_layers, batch_first = True, dropout = dropout, bidirectional = True)
        #Atención 
        self.attention = Attention(hidden_dim = lstm_hidden_size*2)
        #Capa final 
        self.fc = nn.Linear(lstm_hidden_size*2,3) #Salidas: Close, RSI, VIX
    def forward(self, x):
        #x: [batch, seq_len, features] --> Permutar para Conv1d
        x = x.permute(0,2,1)
        #Multi-kernel: CNN + activación
        x3 = torch.relu(self.conv3(x))
        x5 = torch.relu(self.conv5(x))
        x7 = torch.relu(self.conv7(x))
        #Concatenamos 
        x = torch.cat([x3, x5, x7], dim = 1) #[batch, cnn_out_channels*3, seq_len]
        #Normalización 
        x = self.bn(x)
        #Volvemos a formato LSTM
        x = x.permute(0,2,1)
        #LSTM
        lstm_out, _ = self.lstm(x)
        #Atención
        context = self.attention(lstm_out)
        #Predicción final
        return self.fc(context)

##Busqueda para LSTM 
param_grid_simple = {
    'hidden_size': [32, 64],
    'num_layers': [1, 2],
    'dropout': [0.0, 0.2]}

best_model_simple, best_params_simple, history_simple, train_losses_simple, val_losses_simple = find_best_lstm_model(
    model_class=LSTMSsimple, 
    param_grid=param_grid_simple, 
    X_train=X_train, y_train = y_train, 
    X_val = X_val, y_val = y_val,
    batch_sizes=[32, 64],
    learning_rates=[0.001, 0.005],
    num_epochs= 12, 
    early_stopping=True,
    patience=5, 
    min_delta=1e-5, 
    scheduler_patience=2, 
    criterion=nn.MSELoss()
)

##Concatenamos entrenamiento + validación
#Combinamos los datos de entrenamiento y validación
X_train_full = torch.cat([X_train, X_val], dim=0)  # Combina X_train y X_val
y_train_full = torch.cat([y_train, y_val], dim=0)  # Combina y_train y y_val
y_indices_train_full = np.concatenate([y_indices_train, y_indices_val])

print(len(X_train_full))
print(len(X_train) + len(X_val))

best_params_lstm_multivariate = {
    'input_size': best_model_simple.lstm.input_size,
    'hidden_size': best_model_simple.lstm.hidden_size,
    'num_layers': best_model_simple.lstm.num_layers,
    'dropout': best_model_simple.lstm.dropout if hasattr(best_model_simple.lstm, 'dropout') else 0.0,
    'learning_rate': 0.005, ##Igual, lo ponemos manualmente, pero son los mismos de la mejor config
    'batch_size': 64
}
print(best_params_lstm_multivariate)

##Reentrenamos
lstm_model_multivariate, train_losses_lstm_multivariate, predictions_lstm_multivariate, test_loss_lstm_multivariate = retrain_best_model(
    model_class = LSTMSsimple, 
    best_params = best_params_lstm_multivariate,
    X_trainval = X_train_full,
    y_trainval = y_train_full,
    X_test = X_test,
    y_test = y_test,
    num_epochs = 8, 
    criterion = nn.MSELoss(),
    use_scheduler = True, 
    seed = 1
)

##Métricas y graficas
results_lstm_multivariate = predict_and_evaluate_deep(
    model               = lstm_model_multivariate,
    X_train_full        = X_train_full,
    y_train_full        = y_train_full,
    y_indices_train_full= y_indices_train_full,
    X_test              = X_test,
    y_test              = y_test,
    y_indices_test      = y_indices_test,
    scaler              = scaler1,
    target_columns      = ['close', 'VIX', 'RSI'],
    target_indices      = target_indices,     
    date_index          = price_data.index,
    original_data=price_data,
    figsize=(12, 6)
)

##CNNLSTM sin regimen

param_grid = {'cnn_out_channels': [16, 32],
              'kernel_size': [1,2,3],
              'lstm_hidden_size': [32, 64],
              'num_layers': [1,2],
              'dropout': [0.1, 0.2]}

best_model_CNNLSTM, best_params_CNNLSTM, history_CNNNLSTM, train_losses_CNNLSTM, val_losses_CNNLSTM = find_best_lstm_model(
    model_class=CNNLSTM,
    param_grid = param_grid,
    X_train=X_train, y_train=y_train,
    X_val=X_val,     y_val=y_val,
    batch_sizes=[32, 64],
    learning_rates=[0.001, 0.005],
    num_epochs=12,
    early_stopping=True,
    patience=5,          
    min_delta=1e-5,     
    scheduler_patience=2
)

##Modelo CNNLSTM sin pesos
best_params_CNNLSTM_multivariate = {'input_size': best_model_CNNLSTM.conv1d.in_channels,
              'cnn_out_channels': best_model_CNNLSTM.conv1d.out_channels,
              'kernel_size': best_model_CNNLSTM.conv1d.kernel_size[0],
              'lstm_hidden_size': best_model_CNNLSTM.lstm.hidden_size,
              'num_layers': best_model_CNNLSTM.lstm.num_layers,
              'dropout': best_model_CNNLSTM.lstm.dropout if hasattr(best_model_CNNLSTM.lstm, 'dropout') else 0.0,
              'learning_rate': 0.005,
              'batch_size':64

               }
print(best_params_CNNLSTM_multivariate)

##Reentrenamos
cnnlstm_model_multivariate, train_losses_cnnlstm_multivariate, predictions_cnnlstm_multivariate, test_loss_cnnlstm_multivariate = retrain_best_model(
    model_class=CNNLSTM,
    best_params=best_params_CNNLSTM_multivariate,   
    X_trainval=X_train_full,
    y_trainval=y_train_full,
    X_test=X_test,
    y_test=y_test,
    num_epochs=8,
    criterion=nn.MSELoss(),         
    use_scheduler=True,
    seed=1
)
##Graficas y metricas
results_cnnlstm_multivariate = predict_and_evaluate_deep(
    model               = cnnlstm_model_multivariate,
    X_train_full        = X_train_full,
    y_train_full        = y_train_full,
    y_indices_train_full= y_indices_train_full,
    X_test              = X_test,
    y_test              = y_test,
    y_indices_test      = y_indices_test,
    scaler              = scaler1,
    target_columns      = ['close', 'VIX', 'RSI'],
    target_indices      = target_indices,     
    date_index          = price_data.index,
    original_data=price_data,
    figsize=(12, 6)
)

##CNNBiLSTM sin regimen

param_grid = {'cnn_out_channels': [16, 32],
              'lstm_hidden_size': [32, 64],
              'num_layers': [1,2],
              'dropout': [0.1, 0.2]}

best_model_CNNLSTMBi, best_params_CNNLSTMBi, history_CNNNLSTMBi, train_losses_CNNLSTMBi, val_losses_CNNLSTMBi = find_best_lstm_model(
    model_class=CNN_LSTMBi_Attention,
    param_grid = param_grid,
    X_train=X_train, y_train=y_train,
    X_val=X_val,     y_val=y_val,
    batch_sizes=[32, 64],
    learning_rates=[0.001, 0.005],
    num_epochs=12,
    early_stopping=True,
    patience=5,         
    min_delta=1e-5,     
    scheduler_patience=2 
)

#Agregamos input size
best_params_CNNLSTMBi_multivariate = {
    'input_size': 4,
    'cnn_out_channels': best_params_CNNLSTMBi['cnn_out_channels'], 
    'lstm_hidden_size': best_params_CNNLSTMBi['lstm_hidden_size'],
    'num_layers': best_params_CNNLSTMBi['num_layers'],
    'dropout': best_params_CNNLSTMBi['dropout'],
    'learning_rate': best_params_CNNLSTMBi['learning_rate'],
    'batch_size': best_params_CNNLSTMBi['batch_size']
}

print(best_params_CNNLSTMBi_multivariate)

##Reentrenamos
cnnlstmbi_model_multivariate, train_losses_cnnlstmbi_multivariate, predictions_cnnlstmbi_multivariate, test_loss_cnnlstmbi_multivariate = retrain_best_model(
    model_class=CNN_LSTMBi_Attention,
    best_params=best_params_CNNLSTMBi_multivariate,   
    X_trainval=X_train_full,
    y_trainval=y_train_full,
    X_test=X_test,
    y_test=y_test,
    num_epochs=12,
    criterion=nn.MSELoss(),           
    use_scheduler=True,
    seed=1
)

results_cnnlstm_multivariate = predict_and_evaluate_deep(
    model               = cnnlstmbi_model_multivariate,
    X_train_full        = X_train_full,
    y_train_full        = y_train_full,
    y_indices_train_full= y_indices_train_full,
    X_test              = X_test,
    y_test              = y_test,
    y_indices_test      = y_indices_test,
    scaler              = scaler1,
    target_columns      = ['close', 'VIX', 'RSI'],
    target_indices      = target_indices,     
    date_index          = price_data.index,
    original_data=price_data,
    figsize=(12, 6)
)

##Hallamos la variable regimen
train_reg, val_reg, test_reg, start_idx_train_reg, start_idx_val_reg, start_idx_test = train_test_split2(
    price_data, tr_size = 0.7, ts_size = 0.2, vl_size = 0.1, use_validation = True
)

print(f"Tamaño del set de train: {train_reg.shape}")
print(f"Tamaño del set de prueba: {test_reg.shape}")
print(f"Tamaño del set de val: {val_reg.shape}")

scaler_reg = StandardScaler()
train_scaled_reg = scaler_reg.fit_transform(train_reg)

##Aplicamos escalado con los parametros de train
val_scaled_reg = scaler_reg.transform(val_reg)
test_scaled_reg = scaler_reg.transform(test_reg)

print(f"Esta son los datos escalados de train: \n {train_scaled_reg} ")
print(f"Estos son los datos escalados de val: \n {val_scaled_reg}")
print(f"Estos son los datos escalados de test: \n {test_scaled_reg}")

print(f"Esta es la media dl scaler: {scaler_reg.mean_}")
print(f"Esta es la varianza del scaler: {scaler_reg.var_}")

##Método del codo solo con train
inertias = []
K_range = range(1, 10)

for k in K_range:
    #Entrenamos el K-means solamente con el set de entrenamiento
    km = KMeans(n_clusters=k, random_state=1, n_init=10)
    km.fit(train_scaled_reg) 
    inertias.append(km.inertia_)

#Gráfico del Codo
plt.figure(figsize=(12, 6), dpi=100)
plt.plot(K_range, inertias, marker='o', color="#3A7CA5", lw=2)
plt.title("Selección clusteres (solo train)", weight="bold")
plt.xlabel("Número de clusters")
plt.ylabel("Inercia")
plt.grid(alpha=0.3)
plt.show()

for k in range(2,6):
    km = KMeans(n_clusters=k, random_state=1, n_init=10)
    labels = km.fit_predict(train_scaled_reg)
    score = silhouette_score(train_scaled_reg, labels)
    print(f"K={k} → Silhouette={score:.4f}")

##Ajuste kmeans
kmeans_final = KMeans(n_clusters=3, random_state=1, n_init=10)

kmeans_final.fit(train_scaled_reg)

#El predict se hace en los tres conjuntos
regimen_train = kmeans_final.predict(train_scaled_reg)
regimen_val   = kmeans_final.predict(val_scaled_reg)
regimen_test  = kmeans_final.predict(test_scaled_reg)

##Regimen en el tiempo 
price_reg = price_data.copy()

price_reg['Regimen'] = np.concatenate([
    regimen_train,
    regimen_val,
    regimen_test
])

plt.figure(figsize=(14,4))

plt.step(
    price_reg.index,
    price_reg['Regimen'],
    where='post',
    linewidth=1.8
)


plt.xlabel("Fecha")
plt.ylabel("Regímen identificado")

plt.yticks([0,1,2],
           ["Régimen 0",
            "Régimen 1",
            "Régimen 2"])

plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

##Pca

pca = PCA(n_components=2)
pca.fit(train_scaled_reg)

#Transformamos los tres conjuntos
train_pca = pca.transform(train_scaled_reg)
val_pca   = pca.transform(val_scaled_reg)
test_pca  = pca.transform(test_scaled_reg)

df_train_pca = pd.DataFrame(train_pca, columns=['PC1', 'PC2'])
df_train_pca['Cluster'] = regimen_train
df_train_pca['Set'] = 'Entrenamiento (70%)'

df_val_pca = pd.DataFrame(val_pca, columns=['PC1', 'PC2'])
df_val_pca['Cluster'] = regimen_val
df_val_pca['Set'] = 'Validación (10%)'

df_test_pca = pd.DataFrame(test_pca, columns=['PC1', 'PC2'])
df_test_pca['Cluster'] = regimen_test
df_test_pca['Set'] = 'Prueba (20%)'


df_plot = pd.concat([
    df_train_pca,
    df_val_pca,
    df_test_pca
])

plt.figure(figsize=(12, 6), dpi=110)

sns.scatterplot(
    data=df_plot,
    x="PC1",
    y="PC2",
    hue="Cluster",         
    style="Set",           
    palette="Set2",      
    alpha=0.65,
    edgecolor="white",
    s=60
)


plt.xlabel("Componente principal 1")
plt.ylabel("Componente principal 2")

plt.grid(alpha=0.2)

plt.legend(
    bbox_to_anchor=(1.05, 1),
    loc='upper left'
)

plt.tight_layout()
plt.show()

print("Distribución Train:")
print(np.unique(regimen_train, return_counts=True))

print("\nDistribución Validation:")
print(np.unique(regimen_val, return_counts=True))

print("\nDistribución Test:")
print(np.unique(regimen_test, return_counts=True))


##Concatenamos con regimen

##Los convertimos a vectores de columna
r_train_col = regimen_train.reshape(-1, 1)
r_val_col   = regimen_val.reshape(-1, 1)
r_test_col  = regimen_test.reshape(-1, 1)

#Concatenamos: Datos escalados + columna de régimen
train_final = np.hstack([train_scaled_reg, r_train_col])
val_final   = np.hstack([val_scaled_reg, r_val_col])
test_final  = np.hstack([test_scaled_reg, r_test_col])

print(f"Nueva forma del set de entrenamiento: {train_final.shape}")
##Los convertimos a DF
nombres_columnas = list(price_data.columns) + ["Regimen_Rt"]
train_final = pd.DataFrame(train_final, columns=nombres_columnas)
val_final = pd.DataFrame(val_final, columns=nombres_columnas)
test_final = pd.DataFrame(test_final, columns=nombres_columnas)

print(f"Estos son los datos de train: \n {train_final} y dimensión \n {train_final.shape}")
print(f"Estos son los de val: \n {val_final}\ y dimension {val_final.shape}")
print(f"Estos son los de test: \n {test_final} y dimension {test_final.shape}")

###Modelo VARX

##Concatenamos train+val
train_varx = pd.concat([train_final, val_final], ignore_index=True)
test_varx = test_final
print(f"Estos son los datos para el modelo VARX: \n {train_varx}")
print(f"Esta es la dimensión de los datos: {train_varx.shape}")

print(f"Este es el conjunto de prueba para VARX: \n {test_varx}")
print(f"Esta es la dimensión de prueba: {test_varx.shape}")

n_train_varx = len(train_varx) 
idx_train_varx = price_data.index[:n_train_varx]

##supuestos
stationarity_and_homoscedasticity_tests(train_varx)

##Funcion para yeo johnson


def aplicar_yeojohnson(df, columns_to_transform):
    df_transformed = df.copy()
    lambdas = {}
    
    for col in columns_to_transform:
        if col in df_transformed.columns:
            df_transformed[col], lmbda = stats.yeojohnson(df_transformed[col])
            lambdas[col] = lmbda
            print(f"Variable '{col}' transformada (Yeo-Johnson). Lambda: {lmbda:.4f}")
        else:
            print(f"Advertencia: La columna '{col}' no se encuentra en el DataFrame.")
            
    return df_transformed, lambdas


vars_to_yeo_reg = ["close", "VIX", "RSI"]
train_varx_yeo, dict_lambdas_varx = aplicar_yeojohnson(train_varx, vars_to_yeo_reg)
print(f"Estos son los datos con Box cox: \n  {train_varx_yeo} y estos son sus lambda: {dict_lambdas_varx}")


vars_to_diff_yeo = ["close", "DXY"]
train_varx_diff = aplicar_diff(train_varx_yeo, vars_to_diff_yeo)
print(f"Estos son los datos diferenciados: \n {train_varx_diff}")

train_varx_diff["Regimen_Rt"] = train_varx_diff["Regimen_Rt"].astype(int)

stationarity_and_homoscedasticity_tests(train_varx_diff)

##Funcion para mejor orden

def select_varx_order_ps(
    train_df: pd.DataFrame,
    endog_cols: list[str],
    regime_col: str = "Regimen_Rt",
    max_p: int = 12,
    max_s: int = 5,
    verbose: bool = True
):
    """
    Selecciona conjuntamente los órdenes (p, s) de un modelo VARX(p, s)
    mediante AIC, utilizando exclusivamente el conjunto de entrenamiento.

    Definición utilizada:
        Y_t = delta + sum_{j=1}^{p} Phi_j Y_{t-j} + sum_{i=1}^{s} Theta_i R_{t-i} + a_t
    Por tanto:
        s = 1  -> usa R_{t-1}
        s = 2  -> usa R_{t-1}, R_{t-2}
        s = 3  -> usa R_{t-1}, R_{t-2}, R_{t-3}
        ...
    Todos los modelos candidatos se evalúan sobre una muestra común
    para que sus AIC sean comparables.
    """
    required_cols = endog_cols + [regime_col]
    missing_cols = [col for col in required_cols if col not in train_df.columns]
    if missing_cols:
        raise ValueError(f"Faltan columnas requeridas: {missing_cols}")

    work = train_df[required_cols].copy()
    work = work.replace([np.inf, -np.inf], np.nan)
    work = work.apply(pd.to_numeric, errors="coerce")

    #Para construir R_{t-1}, R_{t-2}, ..., R_{t-max_s}
    for lag in range(1, max_s + 1):
        work[f"{regime_col}_L{lag}"] = work[regime_col].shift(lag)

    all_exog_cols = [f"{regime_col}_L{lag}" for lag in range(1, max_s + 1)]

    #Muestra común para comparar AIC entre distintos valores de s
    common_data = work.dropna(subset=endog_cols + all_exog_cols).copy()

    if common_data.empty:
        raise ValueError(
            "La muestra común quedó vacía. Revise los datos o reduzca max_s."
        )

    if verbose:
        print("Diagnóstico de muestra común:")
        print(f"Observaciones disponibles: {len(common_data)}")
        print("NaN en endógenas:", common_data[endog_cols].isna().sum().sum())
        print("NaN en exógenas:", common_data[all_exog_cols].isna().sum().sum())
        print()
        print(f"{'p':<5} | {'s':<5} | {'AIC':<15} | {'Mejor':<6}")
        print("-" * 42)

    results = []
    best_aic = np.inf
    best_p = None
    best_s = None
    best_exog_cols = None

    endog = common_data[endog_cols]

    for s in range(1, max_s + 1):
        exog_cols = [f"{regime_col}_L{lag}" for lag in range(1, s + 1)]
        exog = common_data[exog_cols]

        for p in range(1, max_p + 1):
            try:
                fitted = VAR(endog, exog=exog).fit(p)
                aic = fitted.aic

                is_best = aic < best_aic
                if is_best:
                    best_aic = aic
                    best_p = p
                    best_s = s
                    best_exog_cols = exog_cols.copy()

                results.append({
                    "p": p,
                    "s": s,
                    "aic": aic,
                    "nobs": fitted.nobs,
                    "exog_cols": ", ".join(exog_cols)
                })

                if verbose:
                    marker = "  ◄" if is_best else ""
                    print(f"{p:<5} | {s:<5} | {aic:<15.6f} | {marker}")

            except Exception as e:
                if verbose:
                    print(f"{p:<5} | {s:<5} | Error: {e}")

    results_df = pd.DataFrame(results).sort_values("aic").reset_index(drop=True)

    if best_p is None:
        raise RuntimeError(
            "No fue posible estimar ningún modelo candidato VARX(p,s)."
        )

    if verbose:
        print("-" * 42)
        print(
            f"Orden óptimo seleccionado: VARX({best_p},{best_s}) "
            f"con AIC = {best_aic:.6f}"
        )
        print(f"Variables exógenas utilizadas: {best_exog_cols}")

    return best_p, best_s, best_exog_cols, results_df

##Hacemos la busqueda
var_endo = ["close", "VIX", "RSI", "DXY"]

best_p, best_s, best_exog_cols, aic_table = select_varx_order_ps(
    train_df=train_varx_diff,
    endog_cols=var_endo,
    regime_col="Regimen_Rt",
    max_p=12,
    max_s=5,
    verbose=True
)

print("\nCinco mejores especificaciones:")
print(aic_table.head())

def preparar_varx_final(df, endog_cols, regime_col="Regimen_Rt", s=1):
    data = df[endog_cols + [regime_col]].copy()
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.apply(pd.to_numeric, errors="coerce")

    if s < 1:
        raise ValueError("El orden s debe ser >= 1 para usar solo rezagos disponibles.")

    for lag in range(1, s + 1):
        data[f"{regime_col}_L{lag}"] = data[regime_col].shift(lag)

    exog_cols = [f"{regime_col}_L{lag}" for lag in range(1, s + 1)]

    data = data.dropna(subset=endog_cols + exog_cols)

    return data[endog_cols], data[exog_cols], exog_cols


##Reconstruimos índices

n_train_varx = len(train_varx)
n_test_varx  = len(test_final)

idx_train_varx = price_data.index[:n_train_varx]
idx_test_varx  = price_data.index[n_train_varx:n_train_varx + n_test_varx]

train_varx_diff_dated = train_varx_diff.copy()
train_varx_diff_dated.index = idx_train_varx[
    len(idx_train_varx) - len(train_varx_diff_dated):
]

test_final_dated = test_final.copy()
test_final_dated.index = idx_test_varx

print("Train fechado:", train_varx_diff_dated.index[:3])
print("Test fechado :", test_final_dated.index[:3])

##Ajustamos el modelo
train_endog_model, train_exog_model, exog_cols = preparar_varx_final(
    train_varx_diff_dated,
    endog_cols=var_endo,
    regime_col="Regimen_Rt",
    s=best_s
)

print("Exógenas finales:", exog_cols)
print("Train endógeno :", train_endog_model.shape)
print("Train exógeno  :", train_exog_model.shape)
print("Índice train modelo:", train_endog_model.index[:3])


model_varx = VAR(train_endog_model, exog=train_exog_model)
varx_fitted = model_varx.fit(best_p)

print(varx_fitted.summary())

print("Intercepto:")
print(varx_fitted.intercept)

print("Coeficientes exógenos:")
print(varx_fitted.coefs_exog)

print("Coeficientes autorregresivos:")
for j, Phi_j in enumerate(varx_fitted.coefs, start=1):
    print(f"Phi_{j}:")
    print(Phi_j)

##Construimos exogenas

regimen_total = pd.concat([
    train_varx_diff_dated[["Regimen_Rt"]],
    test_final_dated[["Regimen_Rt"]]
]).sort_index()

for lag in range(0, best_s + 1):
    regimen_total[f"Regimen_Rt_L{lag}"] = regimen_total["Regimen_Rt"].shift(lag)

test_exog_lags = regimen_total.loc[test_final_dated.index, exog_cols].copy()

print(test_exog_lags.head())
print("NaN en exógenas test:")
print(test_exog_lags.isna().sum())

#Validamos residuos
validate_residuals(varx_fitted.resid)

#para graficar métrica

def evaluate_varx(
    varx_fitted,
    train_varx_diff,
    test_final_dated,
    test_exog_lags,
    price_data,
    scaler_reg,
    dict_lambdas_varx,
    var_endo,
    vars_to_diff,
    vars_to_yeo,
    best_p,
    figsize=(12, 6)
):
    """
    Evalúa VARX invirtiendo el pipeline completo:
    Diferenciación → Yeo-Johnson → Z-score
    """
    from scipy import stats

    n_features = scaler_reg.n_features_in_
    col_names = list(price_data.columns)  #orden original del scaler

    def inv_zscore(series, col):
        """Invierte Z-score para una sola columna."""
        col_idx = col_names.index(col)
        dummy   = np.zeros((len(series), n_features))
        dummy[:, col_idx] = series.values
        return pd.Series(
            scaler_reg.inverse_transform(dummy)[:, col_idx],
            index=series.index
        )

    def inv_yeojohnson(series, lmbda):
        y = series.values
        x  = np.zeros_like(y, dtype=float)
        eps = 1e-8
        pos = y >= 0
        neg = ~pos

        if abs(lmbda)>eps:
            x[pos] = np.power(np.maximum(y[pos]*lmbda + 1, 0), 1/lmbda) - 1
        else:
            x[pos] = np.exp(y[pos]) - 1

        if abs(lmbda- 2) > eps:
            x[neg]= 1 -np.power(np.maximum(-(2 - lmbda)*y[neg] + 1, 0), 1/(2 - lmbda))
        else:
            x[neg] = 1- np.exp(-y[neg])

        return pd.Series(x, index=series.index)

    def inv_diff(series, anchor_series, is_forecast=False):
        """
        Invierte diferenciación.
        - is_forecast=False (Train): Suma la diferencia predicha al valor real del t-1.
        - is_forecast=True (Test): Suma acumulada dinámica.
        """
        if is_forecast:
            anchor_date  = series.index[0]
            anchor_loc   = anchor_series.index.get_loc(anchor_date)
            anchor_value = anchor_series.iloc[anchor_loc - 1]
            return series.cumsum() + anchor_value
        else:
            shifted_anchor = anchor_series.shift(1)
            return series + shifted_anchor.loc[series.index]

    def full_invert(series, col, is_diff, is_yeo, is_forecast=False):
        """Aplica inversión completa en orden correcto con anclas correctas."""
        result = series.copy()
        col_idx = col_names.index(col)

        #Invertir diferenciación (si sí)
        if is_diff:
            #Calcular serie Ancla (Z-Score -> Yeo)
            raw_col = price_data[col]
            z_col = (raw_col - scaler_reg.mean_[col_idx]) / scaler_reg.scale_[col_idx]
            
            if is_yeo and col in dict_lambdas_varx:
                anchor_series = apply_yeo_fixed(z_col, dict_lambdas_varx[col])
            else:
                anchor_series = z_col
                
            result = inv_diff(result, anchor_series, is_forecast)

        #Invertir Yeo-Johnson (si sí)
        if is_yeo and col in dict_lambdas_varx:
            result = inv_yeojohnson(result, dict_lambdas_varx[col])

        #Invertir Z-score (siempre)
        result = result * scaler_reg.scale_[col_idx] + scaler_reg.mean_[col_idx]

        return result

    #Para preparar test transformado
    def apply_yeo_fixed(series, lmbda):
        """Aplica Yeo-Johnson con lambda fijo."""
        y   = series.values
        x   = np.zeros_like(y, dtype=float)
        eps = 1e-8
        pos = y >= 0
        neg = ~pos
        if abs(lmbda) > eps:
            x[pos] = (np.power(y[pos] + 1, lmbda) - 1) / lmbda
        else:
            x[pos] = np.log(y[pos] + 1)
        if abs(lmbda - 2) > eps:
            x[neg] = -(np.power(-y[neg] + 1, 2 - lmbda) - 1) / (2 - lmbda)
        else:
            x[neg] = -np.log(-y[neg] + 1)
        return pd.Series(x, index=series.index)

    #Para aplicar Yeo-Johnson al test con lambdas de train
    test_yeo = test_final_dated.copy()
    for col in vars_to_yeo:
        test_yeo[col] = apply_yeo_fixed(test_yeo[col], dict_lambdas_varx[col])

    #Para diferenciar columnas que corresponde
    test_diff = test_yeo.copy()

    for col in vars_to_diff:
        test_diff[col] = test_diff[col].diff()

    test_diff = test_diff.dropna().copy()
    test_diff["Regimen_Rt"] = test_diff["Regimen_Rt"].astype(int)

    #Pronóstico fuera de muestra

    #Últimas observaciones endógenas realmente utilizadas por el modelo
    forecast_input = train_varx_diff[var_endo].iloc[-best_p:].to_numpy()

    #test_diff pierde la primera fila por diferenciación;
    #las exógenas deben quedar exactamente alineadas con ese índice.
    exog_test = test_exog_lags.reindex(test_diff.index).copy()

    if exog_test.isna().any().any():
        raise ValueError(
            "Las variables exógenas de prueba contienen NaN después de alinearlas "
            "con test_diff. Revisa los índices de train/test."
        )

    if len(exog_test) != len(test_diff):
        raise ValueError(
            f"Longitudes incompatibles: test_diff={len(test_diff)}, "
            f"exog_test={len(exog_test)}."
        )

    forecast_arr = varx_fitted.forecast(
        y=forecast_input,
        steps=len(test_diff),
        exog_future=exog_test.to_numpy()
    )

    df_forecast = pd.DataFrame(
        forecast_arr,
        index=test_diff.index,
        columns=var_endo
    )

    #Ajuste en muestra

    fitted_vals = varx_fitted.fittedvalues.copy()

    #El modelo ya fue ajustado con índice de fechas;
    #por seguridad, se fuerza la misma alineación temporal de la muestra usada.
    fitted_vals.index = train_varx_diff.index[-len(fitted_vals):]
    #Métricas y gráficos por variable
    metrics_all = {}

    for col in var_endo:
        is_diff = col in vars_to_diff
        is_yeo  = col in vars_to_yeo

        #Invertir ajuste train (is_forecast = False)
        fitted_real   = full_invert(fitted_vals[col], col, is_diff, is_yeo, is_forecast=False)

        #Invertir pronóstico test (is_forecast = True)
        forecast_real = full_invert(df_forecast[col],col, is_diff, is_yeo, is_forecast=True)

        #Valores reales en escala original
        y_true = price_data[col].loc[test_diff.index]
        y_pred = forecast_real

        common = y_true.index.intersection(y_pred.index)
        y_true = y_true.loc[common]
        y_pred = y_pred.loc[common]

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae= mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        r2 = r2_score(y_true, y_pred)

        metrics_all[col] = {
            'MSE': mse, 'RMSE': rmse,
            'MAE': mae, 'MAPE': mape, 'R2': r2
        }

        print(f"\n{'='*45}")
        print(f"  VARIABLE: {col}")
        print(f"{'='*45}")
        print(f"  RMSE : {rmse:.6f}")
        print(f"  MAE  : {mae:.6f}")
        print(f"  MAPE : {mape:.2f}%")
        print(f"  R2   : {r2:.4f}")

        ##Gráfico

        fig, ax = plt.subplots(figsize=figsize, dpi=100)

        ax.plot(price_data.index, price_data[col],
                color='lightgray', linewidth=1.5, alpha=0.8,
                label=f'Precio real ({col})', zorder=1)

        ax.plot(fitted_real.index, fitted_real,
                color='forestgreen', linestyle=':', linewidth=1.2,
                alpha=0.8, label='Ajuste VARX (Train)', zorder=2)

        ax.axvline(x=test_diff.index[0],
                   color='red', linestyle='--',
                   linewidth=1.5, alpha=0.6,
                   label='Inicio del Test', zorder=3)

        ax.plot(forecast_real.index, forecast_real,
                color='darkorange', linewidth=2.2,
                label='Predicción VARX (Test)', zorder=4)

        ax.set_title(
            f'Ajuste y predicciones del modelo VARX — {col}',
            fontsize=14, fontweight='bold', pad=15, color='#2C3E50'
        )
        ax.set_xlabel('Tiempo', fontsize=12, labelpad=8)
        ax.set_ylabel(col, fontsize=12, labelpad=8)
        ax.legend(loc='upper left', frameon=True,
                  fancybox=True, shadow=True,
                  fontsize=10, framealpha=0.95)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(alpha=0.25, linewidth=0.7)
        plt.tight_layout()
        plt.show()

    return metrics_all, df_forecast


metrics_varx, forecast_varx = evaluate_varx(
    varx_fitted    = varx_fitted,
    train_varx_diff  = train_endog_model,
    test_final_dated  = test_final_dated,
    test_exog_lags  = test_exog_lags,
    price_data    = price_data,
    scaler_reg = scaler_reg,
    dict_lambdas_varx = dict_lambdas_varx,
    var_endo       = var_endo,
    vars_to_diff     = vars_to_diff_yeo,
    vars_to_yeo       = vars_to_yeo_reg,
    best_p            = best_p,
    figsize = (12, 6)
)


##Modelos profundos con regimen

var_pred_deep_reg = list(price_data.columns)
target_cols_reg = ["close", "VIX", "RSI"]
target_indices_reg = [var_pred_deep_reg.index(col) for col in target_cols_reg]
print(f"Estos son los target: {target_indices_reg}")

start_idx_test_reg = start_idx_test

##Creamos las secuencias 
X_train_reg, y_train_reg, y_indices_train_reg = create_multivariate_sequences2(
    train_final.values, 
    time_steps = 60, 
    target_indices = target_indices_reg,
    start_index = start_idx_train_reg
)

X_val_reg, y_val_reg, y_indices_val_reg = create_multivariate_sequences2(
    val_final.values, 
    time_steps=60,
    target_indices=target_indices_reg,
    start_index= start_idx_val_reg
)

X_test_reg, y_test_reg, y_indices_test_reg = create_multivariate_sequences2(
    test_final.values, 
    time_steps=60, 
    target_indices=target_indices_reg,
    start_index = start_idx_test_reg
)

print(X_train_reg.shape, y_train_reg.shape)
print(X_test_reg.shape, y_test_reg.shape)

X_train_reg = torch.tensor(X_train_reg, dtype = torch.float32)
y_train_reg = torch.tensor(y_train_reg, dtype = torch.float32)

X_val_reg = torch.tensor(X_val_reg, dtype = torch.float32)
y_val_reg = torch.tensor(y_val_reg, dtype = torch.float32)

X_test_reg = torch.tensor(X_test_reg, dtype = torch.float32)
y_test_reg = torch.tensor(y_test_reg, dtype = torch.float32)

##Búsqueda de mejores hiperparámetros para LSTM con regimen
param_grid_lstm_reg = {
    'hidden_size': [32, 64],
    'num_layers': [1, 2],
    'dropout': [0.0, 0.2]}

best_model_lstm_reg, best_params_lstm_reg, history_lstm_reg, train_losses_lstm_reg, val_losses_lstm_reg = find_best_lstm_model(
    model_class=LSTMSsimple, 
    param_grid=param_grid_lstm_reg, 
    X_train=X_train_reg, y_train = y_train_reg, 
    X_val = X_val_reg, y_val = y_val_reg,
    batch_sizes=[32, 64],
    learning_rates=[0.001, 0.005],
    num_epochs= 12, 
    early_stopping=True,
    patience=5, 
    min_delta=1e-5, 
    scheduler_patience=2, 
    criterion=nn.MSELoss()
)

##Concatenamos entren + val
X_train_full_reg = torch.cat([X_train_reg, X_val_reg], dim = 0)
y_train_full_reg = torch.cat([y_train_reg, y_val_reg], dim = 0)
y_indices_full_train_reg = np.concatenate([y_indices_train_reg, y_indices_val_reg])

print(len(X_train_full_reg))
print(len(X_train_reg) + len(X_val_reg))

best_params_lstm_multivariate_reg = {
    'input_size': best_model_lstm_reg.lstm.input_size,
    'hidden_size': best_model_lstm_reg.lstm.hidden_size,
    'num_layers': best_model_lstm_reg.lstm.num_layers,
    'dropout': best_model_lstm_reg.lstm.dropout if hasattr(best_model_lstm_reg.lstm, 'dropout') else 0.0,
    'learning_rate': 0.005, ##Igual que anteriormente
    'batch_size': 64
}
print(best_params_lstm_multivariate_reg)

##Reentrenamos
lstm_model_multivariate_reg, train_losses_lstm_multivariate_reg, predictions_lstm_multivariate_reg, test_loss_lstm_multivariate_reg = retrain_best_model(
    model_class = LSTMSsimple, 
    best_params = best_params_lstm_multivariate_reg,
    X_trainval = X_train_full_reg,
    y_trainval = y_train_full_reg,
    X_test = X_test_reg,
    y_test = y_test_reg,
    num_epochs = 8, 
    criterion = nn.MSELoss(),
    use_scheduler = True, 
    seed = 1
)

results_lstm_multivariate = predict_and_evaluate_deep(
    model               = lstm_model_multivariate_reg,
    X_train_full        = X_train_full_reg,
    y_train_full        = y_train_full_reg,
    y_indices_train_full= y_indices_full_train_reg,
    X_test              = X_test_reg,
    y_test              = y_test_reg,
    y_indices_test      = y_indices_test_reg,
    scaler              = scaler_reg,
    target_columns      = ['close', 'VIX', 'RSI'],
    target_indices      = target_indices_reg,     
    date_index          = price_data.index,
    original_data=price_data,
    figsize=(12, 6)
)

##CNNLSTM CON REGIMEN

param_grid_reg = {'cnn_out_channels': [16, 32],
              'kernel_size': [1,2,3],
              'lstm_hidden_size': [32, 64],
              'num_layers': [1,2],
              'dropout': [0.1, 0.2]}

best_model_CNNLSTM_reg, best_params_CNNLSTM_reg, history_CNNNLSTM_reg, train_losses_CNNLSTM_reg, val_losses_CNNLSTM_reg = find_best_lstm_model(
    model_class=CNNLSTM,
    param_grid = param_grid_reg,
    X_train=X_train_reg, y_train=y_train_reg,
    X_val=X_val_reg,     y_val=y_val_reg,
    batch_sizes=[32, 64],
    learning_rates=[0.001, 0.005],
    num_epochs=12,
    early_stopping=True,
    patience=5,          
    min_delta=1e-5,     
    scheduler_patience=2 
)


##Modelo CNNLSTM sin pesos
best_params_CNNLSTM_multivariate_reg = {'input_size': best_model_CNNLSTM_reg.conv1d.in_channels,
              'cnn_out_channels': best_model_CNNLSTM_reg.conv1d.out_channels,
              'kernel_size': best_model_CNNLSTM_reg.conv1d.kernel_size[0],
              'lstm_hidden_size': best_model_CNNLSTM_reg.lstm.hidden_size,
              'num_layers': best_model_CNNLSTM_reg.lstm.num_layers,
              'dropout': best_model_CNNLSTM_reg.lstm.dropout if hasattr(best_model_CNNLSTM_reg.lstm, 'dropout') else 0.0,
              'learning_rate': 0.005,
              'batch_size':64

               }
print(best_params_CNNLSTM_multivariate_reg)

##Reentrenamos 
##CNNLSTM sin pesos
cnnlstm_model_multivariate_reg, train_losses_cnnlstm_multivariate_reg, predictions_cnnlstm_multivariate_reg, test_loss_cnnlstm_multivariate_reg = retrain_best_model(
    model_class=CNNLSTM,
    best_params=best_params_CNNLSTM_multivariate_reg,  
    X_trainval=X_train_full_reg,
    y_trainval=y_train_full_reg,
    X_test=X_test_reg,
    y_test=y_test_reg,
    num_epochs=12,
    criterion=nn.MSELoss(),          
    use_scheduler=True,
    seed=1
)

##Métricas y graficas
results_cnnlstm_multivariate_reg = predict_and_evaluate_deep(
    model               = cnnlstm_model_multivariate_reg,
    X_train_full        = X_train_full_reg,
    y_train_full        = y_train_full_reg,
    y_indices_train_full= y_indices_full_train_reg,
    X_test              = X_test_reg,
    y_test              = y_test_reg,
    y_indices_test      = y_indices_test_reg,
    scaler              = scaler_reg,
    target_columns      = ['close', 'VIX', 'RSI'],
    target_indices      = target_indices_reg,     
    date_index          = price_data.index,
    original_data=price_data,
    figsize=(12, 6)
)

##CNNLSTMBi con regimen

param_grid_cnnlstmbi_reg = {'cnn_out_channels': [16, 32],
              'lstm_hidden_size': [32, 64],
              'num_layers': [1,2],
              'dropout': [0.1, 0.2]}

best_model_CNNLSTMBi_reg, best_params_CNNLSTMBi_reg, history_CNNNLSTMBi_reg, train_losses_CNNLSTMBi_reg, val_losses_CNNLSTMBi_reg = find_best_lstm_model(
    model_class=CNN_LSTMBi_Attention,
    param_grid = param_grid_cnnlstmbi_reg,
    X_train=X_train_reg, y_train=y_train_reg,
    X_val=X_val_reg,     y_val=y_val_reg,
    batch_sizes=[32, 64],
    learning_rates=[0.001, 0.005],
    num_epochs=12,
    early_stopping=True,
    patience=5,         
    min_delta=1e-5,      
    scheduler_patience=2 
)

best_params_CNNLSTMBi_multivariate_reg = {
    'input_size': 5,
    'cnn_out_channels': best_params_CNNLSTMBi_reg['cnn_out_channels'],
    'lstm_hidden_size': best_params_CNNLSTMBi_reg['lstm_hidden_size'],
    'num_layers': best_params_CNNLSTMBi_reg['num_layers'],
    'dropout': best_params_CNNLSTMBi_reg['dropout'],
    'learning_rate': best_params_CNNLSTMBi_reg['learning_rate'],
    'batch_size': best_params_CNNLSTMBi_reg['batch_size']
}

print(best_params_CNNLSTMBi_multivariate_reg)

##Reentrenamos
cnnlstmbi_model_multivariate_reg, train_losses_cnnlstmbi_multivariate_reg, predictions_cnnlstmbi_multivariate_reg, test_loss_cnnlstmbi_multivariate_reg = retrain_best_model(
    model_class=CNN_LSTMBi_Attention,
    best_params=best_params_CNNLSTMBi_multivariate_reg,   
    X_trainval=X_train_full_reg,
    y_trainval=y_train_full_reg,
    X_test=X_test_reg,
    y_test=y_test_reg,
    num_epochs = 8,
    criterion=nn.MSELoss(),
    use_scheduler=True,
    seed=1
)

##metricas y graficos
results__multivariate = predict_and_evaluate_deep(
    model               = cnnlstmbi_model_multivariate_reg,
    X_train_full        = X_train_full_reg,
    y_train_full        = y_train_full_reg,
    y_indices_train_full= y_indices_full_train_reg,
    X_test              = X_test_reg,
    y_test              = y_test_reg,
    y_indices_test      = y_indices_test_reg,
    scaler              = scaler_reg,
    target_columns      = ['close', 'VIX', 'RSI'],
    target_indices      = target_indices_reg,     
    date_index          = price_data.index,
    original_data=price_data,
    figsize=(12, 6)
)
