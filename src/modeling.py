import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import os

class ModelTrainer:
    def __init__(self, input_path, model_output_dir, top_vars, target='aprobacion_boric'):
        self.input_path = input_path
        self.model_output_dir = model_output_dir
        self.top_vars = top_vars
        self.target = target

        os.makedirs(model_output_dir, exist_ok=True)

    def train(self):
        df = pd.read_csv(self.input_path)
        df = df.dropna(subset=self.top_vars + [self.target])

        if df.empty:
            print("⚠️ No hay datos suficientes para entrenar el modelo.")
            return

        print(f"📊 Observaciones con aprobación conocida: {df.shape[0]}")

        # Ordenar por fecha para división temporal
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # Marcar in_sample vs out_of_sample
        split_index = int(0.8 * len(df))
        df["in_sample"] = True
        df.loc[split_index:, "in_sample"] = False

        # Guardar dataset con marca
        df.to_csv(self.input_path, index=False)
        print("💾 Dataset actualizado con columna 'in_sample'")

        # Entrenamiento
        df_train = df[df["in_sample"] == True]
        df_test = df[df["in_sample"] == False]

        X_train = df_train[self.top_vars]
        y_train = df_train[self.target]

        X_test = df_test[self.top_vars]
        y_test = df_test[self.target]

        # Escalado
        scaler_X = MinMaxScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)

        scaler_y = MinMaxScaler()
        y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

        # Modelo
        xgb_model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

        xgb_model.fit(X_train_scaled, y_train_scaled, eval_set=[(X_test_scaled, y_test_scaled)],
                      early_stopping_rounds=20, verbose=False)

        y_pred = xgb_model.predict(X_test_scaled)
        y_pred_inv = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()

        # Métricas
        mae = mean_absolute_error(y_test, y_pred_inv)
        r2 = r2_score(y_test, y_pred_inv)
        print(f"⚡ XGBoost - MAE: {mae:.4f}, R²: {r2:.4f}")

        # Guardar artefactos
        joblib.dump(xgb_model, os.path.join(self.model_output_dir, 'modelo_xgb.pkl'))
        joblib.dump(scaler_X, os.path.join(self.model_output_dir, 'scaler_X.pkl'))
        joblib.dump(scaler_y, os.path.join(self.model_output_dir, 'scaler_y.pkl'))
        joblib.dump(self.top_vars, os.path.join(self.model_output_dir, 'feature_names.pkl'))
        print("💾 Modelo y escaladores guardados correctamente.")
