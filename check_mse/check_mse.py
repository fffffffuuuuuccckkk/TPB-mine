import re
import numpy as np

log_text = """
Horizon 0 : Unnormed MSE : 3.53100, RMSE : 1.87098, MAE : 1.09772, MAPE: 0.02107
Horizon 1 : Unnormed MSE : 6.57820, RMSE : 2.54638, MAE : 1.38586, MAPE: 0.02753
Horizon 2 : Unnormed MSE : 9.64392, RMSE : 3.08005, MAE : 1.62527, MAPE: 0.03389
Horizon 3 : Unnormed MSE : 12.66754, RMSE : 3.52809, MAE : 1.82927, MAPE: 0.03996
Horizon 4 : Unnormed MSE : 15.64037, RMSE : 3.91792, MAE : 2.00159, MAPE: 0.04511
Horizon 5 : Unnormed MSE : 18.53787, RMSE : 4.26186, MAE : 2.15745, MAPE: 0.04987
Horizon 6 : Unnormed MSE : 21.01981, RMSE : 4.53531, MAE : 2.27967, MAPE: 0.05378
Horizon 7 : Unnormed MSE : 23.43905, RMSE : 4.78790, MAE : 2.40140, MAPE: 0.05763
Horizon 8 : Unnormed MSE : 26.07579, RMSE : 5.04872, MAE : 2.57481, MAPE: 0.06217
Horizon 9 : Unnormed MSE : 28.98372, RMSE : 5.32034, MAE : 2.73698, MAPE: 0.06668
Horizon 10 : Unnormed MSE : 31.80252, RMSE : 5.56901, MAE : 2.87419, MAPE: 0.07061
Horizon 11 : Unnormed MSE : 34.07523, RMSE : 5.76035, MAE : 2.96594, MAPE: 0.07358
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