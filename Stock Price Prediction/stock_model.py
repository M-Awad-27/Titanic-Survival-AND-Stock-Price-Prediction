"""
stock_model.py
──────────────
Stock Price Prediction – Model Training & Evaluation.

Implements two forecasting approaches:
  1. **Linear Regression** – Baseline model on flattened sequence windows.
  2. **LSTM (PyTorch)** – Deep learning model for sequential time-series data.

Both models predict the next-day Close price.
All data loading / preprocessing is delegated to ``stock_data_loader.py``.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib

matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt

from stock_data_loader import load_stock_data


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  LSTM Network Definition
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class StockLSTM(nn.Module):
    """
    Simple stacked-LSTM regressor.

    Architecture:
      Input  →  LSTM (``num_layers`` layers)  →  FC  →  scalar output
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Predictor Class
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class StockPredictor:
    """
    Wraps both Linear Regression and LSTM training/evaluation workflows.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        lr: float = 0.001,
        epochs: int = 30,
        batch_size: int = 32,
    ):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

        self.lstm_model: StockLSTM | None = None
        self.lr_model: LinearRegression | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Metrics ──────────────────────────────────────────────────────────

    @staticmethod
    def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Returns MSE, RMSE, MAE, MAPE (%), and R² for the given arrays."""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        return {"MSE": mse, "RMSE": rmse, "MAE": mae, "MAPE_%": mape, "R2": r2}

    # ── Linear Regression ────────────────────────────────────────────────

    def train_linear_regression(
        self, X_train, X_test, y_train, y_test, target_scaler
    ) -> dict:
        """
        Trains a Linear Regression model on **flattened** sequence windows.
        """
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)

        self.lr_model = LinearRegression()
        self.lr_model.fit(X_train_flat, y_train.ravel())

        y_train_pred_s = self.lr_model.predict(X_train_flat).reshape(-1, 1)
        y_test_pred_s = self.lr_model.predict(X_test_flat).reshape(-1, 1)

        # Inverse-transform to original price scale
        y_train_pred = target_scaler.inverse_transform(y_train_pred_s)
        y_test_pred = target_scaler.inverse_transform(y_test_pred_s)
        y_train_orig = target_scaler.inverse_transform(y_train)
        y_test_orig = target_scaler.inverse_transform(y_test)

        metrics = self._compute_metrics(y_test_orig, y_test_pred)

        return {
            "predictions_train": y_train_pred,
            "predictions_test": y_test_pred,
            "actual_train": y_train_orig,
            "actual_test": y_test_orig,
            "metrics": metrics,
        }

    # ── LSTM ─────────────────────────────────────────────────────────────

    def train_lstm(
        self, X_train, X_test, y_train, y_test, target_scaler, progress_callback=None
    ) -> dict:
        """
        Trains a PyTorch LSTM and returns predictions + metrics.
        """
        X_tr_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_tr_t = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        X_te_t = torch.tensor(X_test, dtype=torch.float32).to(self.device)

        loader = DataLoader(
            TensorDataset(X_tr_t, y_tr_t),
            batch_size=self.batch_size,
            shuffle=False,
        )

        input_dim = X_train.shape[2]
        self.lstm_model = StockLSTM(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
        ).to(self.device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.lstm_model.parameters(), lr=self.lr)

        # ── Training loop ────────────────────────────────────────────────
        self.lstm_model.train()
        epoch_losses = []

        for epoch in range(self.epochs):
            total_loss = 0.0
            for bx, by in loader:
                optimizer.zero_grad()
                out = self.lstm_model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * bx.size(0)

            avg_loss = total_loss / len(loader.dataset)
            epoch_losses.append(avg_loss)

            if progress_callback:
                progress_callback(epoch + 1, self.epochs, avg_loss)
            elif (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch [{epoch + 1:3d}/{self.epochs}]  Loss: {avg_loss:.6f}")

        # ── Evaluation ───────────────────────────────────────────────────
        self.lstm_model.eval()
        with torch.no_grad():
            y_train_pred_s = self.lstm_model(X_tr_t).cpu().numpy()
            y_test_pred_s = self.lstm_model(X_te_t).cpu().numpy()

        y_train_pred = target_scaler.inverse_transform(y_train_pred_s)
        y_test_pred = target_scaler.inverse_transform(y_test_pred_s)
        y_train_orig = target_scaler.inverse_transform(y_train)
        y_test_orig = target_scaler.inverse_transform(y_test)

        metrics = self._compute_metrics(y_test_orig, y_test_pred)

        return {
            "predictions_train": y_train_pred,
            "predictions_test": y_test_pred,
            "actual_train": y_train_orig,
            "actual_test": y_test_orig,
            "epoch_losses": epoch_losses,
            "metrics": metrics,
        }

    # ── Visualisation ────────────────────────────────────────────────────

    @staticmethod
    def save_predictions_plot(
        actual, lr_pred, lstm_pred, dates, save_path="data/stock_predictions.png"
    ):
        """
        Creates a comparison plot: Actual vs Linear Regression vs LSTM predictions.
        """
        plt.figure(figsize=(14, 7))
        plt.plot(dates, actual, label="Actual Close", color="black", lw=1.5, alpha=0.85)
        plt.plot(dates, lr_pred, label="Linear Regression", color="#2563eb", ls="--", alpha=0.75)
        plt.plot(dates, lstm_pred, label="LSTM", color="#f59e0b", alpha=0.75)

        plt.title("Stock Price Prediction — Test Set Comparison", fontsize=14)
        plt.xlabel("Date")
        plt.ylabel("Price ($)")
        plt.legend()
        plt.grid(True, ls=":", alpha=0.5)
        plt.tight_layout()

        if os.path.dirname(save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"\n[stock_model] Plot saved to {save_path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    SEQ_LENGTH = 10
    EPOCHS = 20

    # ── 1. Load & preprocess data ────────────────────────────────────────
    csv_path = os.path.join("data", "AAPL_2020-01-01_2023-01-01.csv")

    # Download if not cached
    if not os.path.exists(csv_path):
        from stock_data_loader import download_stock_data
        csv_path = download_stock_data("AAPL", "2020-01-01", "2023-01-01")

    (
        X_train, X_test,
        y_train, y_test,
        train_dates, test_dates,
        feature_scaler, target_scaler,
    ) = load_stock_data(csv_path=csv_path, seq_length=SEQ_LENGTH)

    print(f"\nTrain: X={X_train.shape}  y={y_train.shape}")
    print(f"Test:  X={X_test.shape}  y={y_test.shape}")

    # ── 2. Train Linear Regression ───────────────────────────────────────
    predictor = StockPredictor(epochs=EPOCHS, hidden_dim=64, num_layers=2)

    print(f"\n{'=' * 50}")
    print("  LINEAR REGRESSION")
    print(f"{'=' * 50}")
    lr_res = predictor.train_linear_regression(
        X_train, X_test, y_train, y_test, target_scaler
    )
    print("  Test Metrics:")
    for k, v in lr_res["metrics"].items():
        print(f"    {k:8s}: {v:.4f}")

    # ── 3. Train LSTM ────────────────────────────────────────────────────
    print(f"\n{'=' * 50}")
    print("  LSTM (PyTorch)")
    print(f"{'=' * 50}")
    lstm_res = predictor.train_lstm(
        X_train, X_test, y_train, y_test, target_scaler
    )
    print("  Test Metrics:")
    for k, v in lstm_res["metrics"].items():
        print(f"    {k:8s}: {v:.4f}")

    # ── 4. Comparison summary ────────────────────────────────────────────
    print(f"\n{'=' * 50}")
    print("  MODEL COMPARISON (Test Set)")
    print(f"{'=' * 50}")
    print(f"  {'Metric':<10} {'Lin. Reg.':>12}  {'LSTM':>12}")
    print(f"  {'-' * 36}")
    for k in lr_res["metrics"]:
        lr_v = lr_res["metrics"][k]
        lstm_v = lstm_res["metrics"][k]
        print(f"  {k:<10} {lr_v:>12.4f}  {lstm_v:>12.4f}")

    # ── 5. Save comparison plot ──────────────────────────────────────────
    predictor.save_predictions_plot(
        actual=lstm_res["actual_test"].ravel(),
        lr_pred=lr_res["predictions_test"].ravel(),
        lstm_pred=lstm_res["predictions_test"].ravel(),
        dates=test_dates,
        save_path="data/AAPL_predictions_comparison.png",
    )
