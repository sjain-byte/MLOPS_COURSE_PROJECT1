from scipy.stats import uniform,randint

LIGHTGM_PARAMS = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(5, 50),
    'learning_rate': uniform(0.01, 0.2),
    'num_leaves': randint(20, 100),
    'boosting_type': ['gbdt', 'dart','goss'],
}

RANDOM_SEARCH_PARAMS = {
    'n_iter':4,
    'n_jobs':-1,
    'cv': 2,
    'verbose': 2,
    'random_state': 42,
    'scoring': 'accuracy', 
}

