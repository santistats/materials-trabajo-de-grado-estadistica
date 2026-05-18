import pandas as pd 
import numpy as np
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.special import inv_boxcox
import ta 

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

##Definimos la función de Diebold-Mariano
def diebold_mariano_test(actual, pred1, pred2, loss='mse', h=1, alpha=0.05):
    actual = np.array(actual)
    pred1  = np.array(pred1)
    pred2  = np.array(pred2)
    
    e1 = actual - pred1
    e2 = actual - pred2
    
    if loss == 'mse':
        d = e1**2 - e2**2
    elif loss == 'mae':
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError("El parámetro loss debe ser 'mse' o 'mae'")
    
    T = len(d)
    d_bar = np.mean(d)
    
    def autocovariance(x, lag):
        if lag == 0:
            return np.mean((x - np.mean(x))**2)
        x_mean = np.mean(x)
        return np.sum((x[lag:] - x_mean) * (x[:-lag] - x_mean)) / T
    
    gamma = np.array([autocovariance(d, lag) for lag in range(h)])
    V_d = gamma[0] + 2 * np.sum(gamma[1:])
    
    if V_d <= 0:
        return {'DM_statistic': float('nan'), 'p_value': 0.0, 'mean_loss_d': d_bar}
    
    DM_stat = d_bar / np.sqrt(V_d / T)
    k = np.sqrt((T + 1 - 2*h + (h * (h - 1)) / T) / T)
    DM_stat_hln = DM_stat * k
    p_value = 2 * stats.t.sf(np.abs(DM_stat_hln), df=T-1)
    
    return {'DM_statistic': DM_stat_hln, 'p_value': p_value, 'mean_loss_d': d_bar}

##Nueva función para múltiples modelos
def matriz_diebold_mariano(actual, dict_predicciones, loss='mse', h=1, alpha=0.05):
    modelos = list(dict_predicciones.keys())
    n = len(modelos)
    
    #Inicializar matrices vacías
    matriz_p_values = pd.DataFrame(np.nan, index=modelos, columns=modelos)
    matriz_stats    = pd.DataFrame(np.nan, index=modelos, columns=modelos)
    
    #Rellenar matrices
    for i in range(n):
        for j in range(n):
            if i == j:
                #Un modelo comparado consigo mismo no tiene diferencia
                matriz_p_values.iloc[i, j] = 1.0 
                matriz_stats.iloc[i, j] = 0.0
            else:
                nombre_m1 = modelos[i]
                nombre_m2 = modelos[j]
                
                resultado = diebold_mariano_test(
                    actual, 
                    dict_predicciones[nombre_m1], 
                    dict_predicciones[nombre_m2], 
                    loss=loss, h=h, alpha=alpha
                )
                
                matriz_p_values.iloc[i, j] = resultado['p_value']
                matriz_stats.iloc[i, j]    = resultado['DM_statistic']
                
    return matriz_p_values, matriz_stats

##Cargamos ARIMA

close = pd.DataFrame(price_data["close"])
close

##Dividimos en train y test
train_size_close = int(len(close)*0.80)
train_close, test_close = close[0:train_size_close], close[train_size_close:len(close)]
print(f"Observaciones totales: {len(close)}")
print(f"Entrenamiento: {len(train_close)}")
print(f"Prueba: {len(test_close)}")

##Transformamos con boxcox 
train_close["train_boxcox"], lambda_optim_close = stats.boxcox(train_close["close"])
print(f"Lamba optimo para boxcox: {lambda_optim_close}")

train_close["close_boxcox_diff"] = pd.Series(train_close["train_boxcox"]).diff().dropna()
train_close

##Obtenemos el modelo
mejor_order = (0, 0, 0) 
model_final = ARIMA(train_close["close_boxcox_diff"], order=mejor_order).fit()

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

##Hacemos las trasnformaciones necesarias basados en los test de significancia que hice
vars_to_boxcox = ["close", "VIX", "RSI"]
vars_to_diff = ["close", "DXY"]

train_box, dict_lanbda = aplicar_boxcox(train_raw, vars_to_boxcox)
print(f"Estos son los lambdas calculados en Train: \n {dict_lanbda}")

train_data = aplicar_diff(train_box, vars_to_diff)
print(f"Estos son los datos con diferenciados: \n {train_data}")


##Entrenamos el modelo con el p óptimo
p_opt = 2
var_model = VAR(train_data)
var_model = var_model.fit(p_opt)
var_model

##Modelos profundos

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

var_pred_lstm = list(price_data.columns)
target_cols_lstm = ["close"]
target_indices_lstm = [var_pred_lstm.index(col) for col in target_cols_lstm]
print(f"Este es el target: {target_indices_lstm}")

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

X_train_lstm = torch.tensor(X_train_lstm, dtype=torch.float32)
y_train_lstm = torch.tensor(y_train_lstm, dtype=torch.float32)

X_val_lstm = torch.tensor(X_val_lstm, dtype=torch.float32)
y_val_lstm = torch.tensor(y_val_lstm, dtype=torch.float32)

X_test_lstm = torch.tensor(X_test_lstm, dtype=torch.float32)
y_test_lstm = torch.tensor(y_test_lstm, dtype=torch.float32)


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

X_train_full_LSTM = torch.cat([X_train_lstm, X_val_lstm], dim = 0)
y_train_full_LSTM = torch.cat([y_train_lstm, y_val_lstm], dim = 0)
y_indices_train_full_lstm = np.concatenate([y_indices_train_lstm, y_indices_val_lstm])

print(len(X_train_full_LSTM))
print(len(X_train_lstm) + len(X_val_lstm)) 

##Cargamos el modelo
import json
ruta_params_lstm_univariate = r"hiperparametros_lstm_univariado.json"
with open(ruta_params_lstm_univariate, "r") as f_lstm_uni:
    best_params_lstm_uni = json.load(f_lstm_uni)

best_params_lstm_uni = {k: v for k, v in best_params_lstm_uni.items()
                        if k not in ("learning_rate", "batch_size")}

model_lstm_univariate = LSTMSsimple1(**best_params_lstm_uni)
ruta_model_lstm_univariate = r"modelo_lstm_univariado.pth"
model_lstm_univariate.load_state_dict(torch.load(ruta_model_lstm_univariate))
print(best_params_lstm_uni)
print(model_lstm_univariate)

##Modelos profundos sin regimen
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

#Arquitecturas
###LSTM SIMPLE
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

#Combinamos los datos de entrenamiento y validación
X_train_full = torch.cat([X_train, X_val], dim=0)  # Combina X_train y X_val
y_train_full = torch.cat([y_train, y_val], dim=0)  # Combina y_train y y_val
y_indices_train_full = np.concatenate([y_indices_train, y_indices_val])

print(len(X_train_full))
print(len(X_train) + len(X_val))

##LSTM multi
ruta_params_lstm_multivariate = r"hiperparametros_lstm_multivariado.json"
with open(ruta_params_lstm_multivariate, "r") as f_lstm_multi:
    best_params_lstm_multivariate = json.load(f_lstm_multi)

best_params_lstm_multivariate = {k: v for k, v in best_params_lstm_multivariate.items()
                        if k not in ("learning_rate", "batch_size")}

model_lstm_multivariate = LSTMSsimple(**best_params_lstm_multivariate)
ruta_model_lstm_multivariate = r"modelo_lstm_multivariado.pth"
model_lstm_multivariate.load_state_dict(torch.load(ruta_model_lstm_multivariate))
print(best_params_lstm_multivariate)
print(model_lstm_multivariate)

##CNNLSTM
ruta_params_cnnlstm_multivariate = r"hiperparametros_cnnlstm_multivariado.json"
with open(ruta_params_cnnlstm_multivariate, "r") as f_cnnlstm_multi: 
    best_params_cnnlstm_multivariate = json.load(f_cnnlstm_multi)

best_params_cnnlstm_multivariate = {k: v for k, v in best_params_cnnlstm_multivariate.items()
                        if k not in ("learning_rate", "batch_size")}

model_cnnlstm_multivariate = CNNLSTM(**best_params_cnnlstm_multivariate)
ruta_model_cnnlstm_multivariate = r"modelo_cnnlstm_multivariate.pth"
model_cnnlstm_multivariate.load_state_dict(torch.load(ruta_model_cnnlstm_multivariate))
print(best_params_cnnlstm_multivariate)
print(model_cnnlstm_multivariate)

##CNN_LSTMBi_Attention
ruta_params_cnnlstmbi_multivariate = r"hiperparametros_cnnlstmbi_multivariado.json"
with open(ruta_params_cnnlstmbi_multivariate, "r") as f_cnnlstmbi_multi:
    best_params_cnnlstmbi_multivariate = json.load(f_cnnlstmbi_multi)

best_params_cnnlstmbi_multivariate = {k: v for k, v in best_params_cnnlstmbi_multivariate.items()
                        if k not in ("learning_rate", "batch_size")}

model_cnnlstmbi_multivariate = CNN_LSTMBi_Attention(**best_params_cnnlstmbi_multivariate)
ruta_model_cnnlstmbi_multivariate = r"modelo_cnnlstmbi_multivariate.pth"
model_cnnlstmbi_multivariate.load_state_dict(torch.load(ruta_model_cnnlstmbi_multivariate))

print(best_params_cnnlstmbi_multivariate)
print(model_cnnlstmbi_multivariate)

##Modelos con regimen

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

inertias = []
K_range = range(1, 10)

for k in K_range:
    # Entrenamos el K-means SOLAMENTE con el set de entrenamiento
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

kmeans_final = KMeans(n_clusters=3, random_state=1, n_init=10)

kmeans_final.fit(train_scaled_reg)

regimen_train = kmeans_final.predict(train_scaled_reg)
regimen_val   = kmeans_final.predict(val_scaled_reg)
regimen_test  = kmeans_final.predict(test_scaled_reg)


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

#Etiquetas discretas en Y
plt.yticks([0,1,2],
           ["Régimen 0",
            "Régimen 1",
            "Régimen 2"])

plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()


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

def inv_yeojohnson_scalar(y, lmbda, eps=1e-8):
    y = np.array(y)
    x = np.zeros_like(y, dtype=float)
    pos = y >= 0
    neg = ~pos
    if abs(lmbda) > eps:
        x[pos] = np.power(np.maximum(y[pos] * lmbda + 1, 0), 1/lmbda) - 1
    else:
        x[pos] = np.exp(y[pos]) - 1
    if abs(lmbda - 2) > eps:
        x[neg] = 1 - np.power(np.maximum(-(2 - lmbda) * y[neg] + 1, 0), 1/(2 - lmbda))
    else:
        x[neg] = 1 - np.exp(-y[neg])
    return x

##Concatenamos train+val
train_varx = pd.concat([train_final, val_final], ignore_index=True)
test_varx = test_final
print(f"Estos son los datos para el modelo VARX: \n {train_varx}")
print(f"Esta es la dimensión de los datos: {train_varx.shape}")

print(f"Este es el conjunto de prueba para VARX: \n {test_varx}")
print(f"Esta es la dimensión de prueba: {test_varx.shape}")

n_train_varx = len(train_varx)  #80% (train_final + val_final concatenados)
idx_train_varx = price_data.index[:n_train_varx]

vars_to_yeo_reg = ["close", "VIX", "RSI"]
train_varx_yeo, dict_lambdas_varx = aplicar_yeojohnson(train_varx, vars_to_yeo_reg)
print(f"Estos son los datos con Yeo johnson: \n  {train_varx_yeo} y estos son sus lambda: {dict_lambdas_varx}")

##Guardamos la serie en escala yeo johnson pero pre
train_varx_yeo_indexed = train_varx_yeo.copy()
train_varx_yeo_indexed.index = idx_train_varx  #asignar fechas reales


vars_to_diff_yeo = ["close", "DXY"]
train_varx_diff = aplicar_diff(train_varx_yeo, vars_to_diff_yeo)
print(f"Estos son los datos diferenciados: \n {train_varx_diff}")

#Ajustar modelo final con el orden óptimo
var_endo = ["close", "VIX", "RSI", "DXY"]
var_exo = ["Regimen_Rt"]
best_p = 2
model_varx  = VAR(train_varx_diff[var_endo], exog=train_varx_diff[var_exo])
varx_fitted = model_varx.fit(best_p)

##Cargamos los modelos profundos con regimen
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

X_train_full_reg = torch.cat([X_train_reg, X_val_reg], dim = 0)
y_train_full_reg = torch.cat([y_train_reg, y_val_reg], dim = 0)
y_indices_full_train_reg = np.concatenate([y_indices_train_reg, y_indices_val_reg])

print(len(X_train_full_reg))
print(len(X_train_reg) + len(X_val_reg))

##Cargamos LSTM con reg
ruta_params_lstm_regimen = r"hiperparametros_lstm_multivariado_regimen.json"
with open(ruta_params_lstm_regimen, "r") as f_lstm_reg:
    best_params_lstm_reg = json.load(f_lstm_reg)

best_params_lstm_reg = {k: v for k, v in best_params_lstm_reg.items()
                        if k not in ("learning_rate", "batch_size")}

model_lstm_reg = LSTMSsimple(**best_params_lstm_reg)
ruta_model_lstm_reg = r"modelo_lstm_multivariado_regimen.pth"
model_lstm_reg.load_state_dict(torch.load(ruta_model_lstm_reg))

print(best_params_lstm_reg)
print(model_lstm_reg)

##Cargamos el modelo CNNLSTM con régimen
ruta_params_cnnlstm_regimen = r"hiperparametros_cnnlstm_multivariado_regimen.json"
with open(ruta_params_cnnlstm_regimen, "r") as f_cnnlstm_reg: 
    best_params_cnnlstm_reg = json.load(f_cnnlstm_reg)

best_params_cnnlstm_reg = {k: v for k, v in best_params_cnnlstm_reg.items()
                        if k not in ("learning_rate", "batch_size")}

model_cnnlstm_reg = CNNLSTM(**best_params_cnnlstm_reg)
ruta_model_cnnlstm_reg = r"modelo_cnnlstm_multivariate_regimen.pth"
model_cnnlstm_reg.load_state_dict(torch.load(ruta_model_cnnlstm_reg))

print(best_params_cnnlstm_reg)
print(model_cnnlstm_reg)

##Cargamos el modelo CNNLSTMBI con régimen
ruta_params_cnnlstmbi_regimen = r"hiperparametros_cnnlstmbi_multivariado_regimen.json"
with open(ruta_params_cnnlstmbi_regimen, "r") as f_cnnlstmbi_reg: 
    best_params_cnnlstmbi_reg = json.load(f_cnnlstmbi_reg)

best_params_cnnlstmbi_reg = {k: v for k, v in best_params_cnnlstmbi_reg.items()
                            if k not in ("learning_rate", "batch_size")}

model_cnnlstmbi_reg = CNN_LSTMBi_Attention(**best_params_cnnlstmbi_reg)
ruta_model_cnnlstmbi_reg = r"modelo_cnnlstmbi_multivariado_regimen.pth"
model_cnnlstmbi_reg.load_state_dict(torch.load(ruta_model_cnnlstmbi_reg))

print(best_params_cnnlstmbi_reg)
print(model_cnnlstmbi_reg)

##Preparamos para diebold mariano

##Generación de predicciones y alineación
print("\n--- Generando predicciones de todos los modelos ---")

#Referencia maestra de fechas
fechas_maestras = price_data.index[y_indices_test]
dict_predicciones = {}

##Arima
print("Prediciendo ARIMA...")
arima_f_diff = model_final.forecast(steps=len(test_close))
arima_idx = price_data.index[-len(arima_f_diff):]
arima_f_box = train_close["train_boxcox"].iloc[-1] + arima_f_diff.cumsum()
arima_p_real = pd.Series(inv_boxcox(arima_f_box, lambda_optim_close), index=arima_idx)
dict_predicciones['ARIMA'] = arima_p_real.loc[fechas_maestras].values

##VAR
print("Prediciendo VAR estándar...")
#Usamos el var_model entrenado anteriormente
var_f_diff = var_model.forecast(y=train_data.values[-p_opt:], steps=len(test_raw))
var_idx = price_data.index[-len(var_f_diff):]

var_f_box = train_box['close'].iloc[-1] + var_f_diff[:, 0].cumsum()
var_p_real = pd.Series(inv_boxcox(var_f_box, dict_lanbda['close']), index=var_idx)
dict_predicciones['VAR'] = var_p_real.loc[fechas_maestras].values

#VARX
print("Prediciendo VARX con régimen...")
varx_f_diff = varx_fitted.forecast(y=train_varx_diff[var_endo].values[-best_p:], steps=len(test_varx), exog_future=test_varx[var_exo])
varx_idx = price_data.index[-len(varx_f_diff):]
varx_f_yeo = train_varx_yeo['close'].iloc[-1] + varx_f_diff[:, 0].cumsum()
##Invertir Yeo Johnson
varx_f_yeo_inv = inv_yeojohnson_scalar(varx_f_yeo, dict_lambdas_varx['close'])
#Invertir Z score
dummy_varx = np.zeros((len(varx_f_yeo_inv), scaler_reg.n_features_in_))
dummy_varx[:, 0] = varx_f_yeo_inv
varx_p_real = pd.Series(
    scaler_reg.inverse_transform(dummy_varx)[:, 0],
    index=varx_idx
)
dict_predicciones['VARX'] = varx_p_real.loc[fechas_maestras].values


##Modelos profundos
dl_models = {
    'LSTM Uni': (model_lstm_univariate, X_test_lstm, 'uni'),
    'LSTM Multi': (model_lstm_multivariate, X_test, 'multi'),
    'CNNLSTM Multi': (model_cnnlstm_multivariate, X_test, 'multi'),
    'CNNLSTMBi-Att Multi': (model_cnnlstmbi_multivariate, X_test, 'multi'),
    'LSTM Reg': (model_lstm_reg, X_test_reg, 'multi_reg'),
    'CNNLSTM Reg': (model_cnnlstm_reg, X_test_reg, 'multi_reg'),
    'CNNLSTMBi-Att Reg': (model_cnnlstmbi_reg, X_test_reg, 'multi_reg')
}

for name, (model, X_in, s_type) in dl_models.items():
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_in, dtype=torch.float32) if isinstance(X_in, np.ndarray) else X_in).numpy()
    
    if s_type == 'uni':
        dict_predicciones[name] = scaler_lstm.inverse_transform(preds).flatten()
    elif s_type == 'multi':
        dummy = np.zeros((len(preds), 4)); dummy[:, target_indices] = preds
        dict_predicciones[name] = scaler1.inverse_transform(dummy)[:, 0]
    else: # multi_reg
        dummy = np.zeros((len(preds), 4)); dummy[:, target_indices_reg] = preds
        dict_predicciones[name] = scaler_reg.inverse_transform(dummy)[:, 0]

##matriz de comparación final: Diebold Mariano
actual_values = price_data['close'].iloc[y_indices_test].values

print("\n---Calculando test de Diebold-Mariano---")
df_pvalues, df_stats = matriz_diebold_mariano(actual_values, dict_predicciones)

##Heatmap
plt.figure(figsize=(14, 12))
sns.heatmap(df_pvalues, annot=True, cmap="coolwarm", fmt=".3f", vmin=0, vmax=0.05)
plt.show()
