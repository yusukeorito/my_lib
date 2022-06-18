model_params_regression = {
    "task_type": "CPU",
                
    "loss_function":"RMSE",
    "eval_metric":"RMSE",

    "learning_rate": 0.05,
    "depth": 6,
    
    "random_state": 2022,
    "verbose": -1,
    "n_jobs": 8,
}


model_params_binary = {
    "task_type": "CPU",

    "loss_function": "Logloss",
    "eval_metric":"AUC", 

    "learning_rate": 0.05,
    "depth": 6,


    "random_state": 2022,
    "verbose": -1,
    "n_jobs": 8,
    }


model_params_multiclass = {
    "boosting_type": "gbdt",

    "num_class": 9, 
    "objective": "multiclass", #タスクによって変更する
    "metric": "None", #custom lossを使うときはここをnoneにしておく

    "learning_rate": 0.05,
    "depth": 6,

    "random_state": 2022,
    "verbose": -1,
    "n_jobs": 8,
}

fit_params = {
    "num_boost_round": 10000,
    "early_stopping_rounds": 100,
    "verbose_eval": 100,
}


        




