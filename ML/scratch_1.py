
from sklearn.model_selection import ParameterGrid
from params import ridge_param_grid, elasticnet_param_grid, lst_params_grid

# Loop through the hyperparameter combinations and log results in separate runs
for params in ParameterGrid(lst_params_grid):
    print(params['learning_rate']
          )