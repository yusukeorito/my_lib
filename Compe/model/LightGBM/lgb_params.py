model_params_regression = {
    "boosting_type": "gbdt",
                
    "objective": "regression", 
    "metric":"rmse",

    "learning_rate": 0.05,
    "max_depth": 12,
    "num_leaves":31,
    
    "reg_lambda": 1.0,
    "reg_alpha": 1.,
    
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    "subsample_freq": 1,
    
    "random_state": 2022,
    "verbose": -1,
    "n_jobs": 8,
}


model_params_binary = {
    "boosting_type": "gbdt",

    
    "objective": "binary", 
    "metric":"binary_logloss", 

    "learning_rate": 0.05,
    "max_depth": 12,

    "reg_lambda": 1.,
    "reg_alpha": .1,

    "colsample_bytree": .5,
    "min_child_samples": 10,
    "subsample_freq": 3,
    "subsample": .8,

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
    "max_depth": 12,

    "reg_lambda": 1.,
    "reg_alpha": .1,

    "colsample_bytree": .5,
    "min_child_samples": 10,
    "subsample_freq": 3,
    "subsample": .8,

    "random_state": 2022,
    "verbose": -1,
    "n_jobs": 8,
}

fit_params = {
    "num_boost_round": 10000,
    "early_stopping_rounds": 100,
    "verbose_eval": 100,
}

