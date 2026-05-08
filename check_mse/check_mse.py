import re
import numpy as np

log_text = """
Horizon 0 : Unnormed MSE : 3.82596, RMSE : 1.94828, MAE : 1.16609, MAPE: 0.02346
Horizon 1 : Unnormed MSE : 7.09088, RMSE : 2.64586, MAE : 1.47495, MAPE: 0.03094
Horizon 2 : Unnormed MSE : 10.54275, RMSE : 3.21984, MAE : 1.71806, MAPE: 0.03744
Horizon 3 : Unnormed MSE : 14.08219, RMSE : 3.71677, MAE : 1.93527, MAPE: 0.04333
Horizon 4 : Unnormed MSE : 17.62823, RMSE : 4.15379, MAE : 2.10934, MAPE: 0.04856
Horizon 5 : Unnormed MSE : 20.99080, RMSE : 4.53035, MAE : 2.25904, MAPE: 0.05317
Horizon 6 : Unnormed MSE : 24.13515, RMSE : 4.85502, MAE : 2.40075, MAPE: 0.05761
Horizon 7 : Unnormed MSE : 27.00556, RMSE : 5.13384, MAE : 2.52302, MAPE: 0.06152
Horizon 8 : Unnormed MSE : 30.31397, RMSE : 5.43754, MAE : 2.66392, MAPE: 0.06613
Horizon 9 : Unnormed MSE : 33.51767, RMSE : 5.71655, MAE : 2.80448, MAPE: 0.07047
Horizon 10 : Unnormed MSE : 36.32576, RMSE : 5.94940, MAE : 2.91700, MAPE: 0.07399
Horizon 11 : Unnormed MSE : 38.35821, RMSE : 6.11016, MAE : 2.97099, MAPE: 0.07615
"""

pattern = re.compile(
    r"Horizon\s+(\d+)\s*:\s*"
    r"Unnormed MSE\s*:\s*([\d.]+),\s*"
    r"RMSE\s*:\s*([\d.]+),\s*"
    r"MAE\s*:\s*([\d.]+),\s*"
    r"MAPE\s*:\s*([\d.]+)"
)

mse_list = []
rmse_list = []
mae_list = []
mape_list = []

for match in pattern.finditer(log_text):
    horizon = int(match.group(1))
    mse = float(match.group(2))
    rmse = float(match.group(3))
    mae = float(match.group(4))
    mape = float(match.group(5))

    mse_list.append(mse)
    rmse_list.append(rmse)
    mae_list.append(mae)
    mape_list.append(mape)

print(f"Total Horizons: {len(mse_list)}")
print(f"Average Unnormed MSE: {np.mean(mse_list):.5f}")
print(f"Average RMSE: {np.mean(rmse_list):.5f}")
print(f"Average MAE: {np.mean(mae_list):.5f}")
print(f"Average MAPE: {np.mean(mape_list):.5f}")