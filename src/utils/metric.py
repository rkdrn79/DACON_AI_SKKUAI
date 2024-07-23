import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from typing import Dict

#original code
def expected_calibration_error(y_true, y_prob, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    bin_totals = np.histogram(y_prob, bins=np.linspace(0, 1, n_bins + 1), density=False)[0]
    non_empty_bins = bin_totals > 0
    bin_weights = bin_totals / len(y_prob)
    bin_weights = bin_weights[non_empty_bins]
    prob_true = prob_true[:len(bin_weights)]
    prob_pred = prob_pred[:len(bin_weights)]
    ece = np.sum(bin_weights * np.abs(prob_true - prob_pred))
    return ece

#custom
def auc_brier_ece_custom(answer_array:np.array, 
                         pred_array:np.array) -> Dict:
    
    auc_score_false = roc_auc_score(answer_array[:,0], pred_array[:,0]) #false가 0
    auc_score_true = roc_auc_score(answer_array[:,1], pred_array[:,1]) #true가 0
    mean_auc = np.mean([auc_score_true,auc_score_false])
    
    brier_score_false = mean_squared_error(answer_array[:,0], pred_array[:,0]) #false가 0
    brier_score_true = mean_squared_error(answer_array[:,1], pred_array[:,1]) #false가 0
    mean_brier = np.mean([brier_score_false,brier_score_true])

    ece_score_false = expected_calibration_error(answer_array[:,0], pred_array[:,0]) #false가 0
    ece_score_true = expected_calibration_error(answer_array[:,1], pred_array[:,1]) #false가 0
    mean_ece = np.mean([ece_score_false,ece_score_true])
    
    combined_score = 0.5 * (1 - mean_auc) + 0.25 * mean_brier + 0.25 * mean_ece
    
    metrics = {
        "auc_score_false": auc_score_false,
        "auc_score_true": auc_score_true,
        "mean_auc": mean_auc,
        "brier_score_false": brier_score_false,
        "brier_score_true": brier_score_true,
        "mean_brier": mean_brier,
        "ece_score_false": ece_score_false,
        "ece_score_true": ece_score_true,
        "mean_ece": mean_ece,
        "combined_score": combined_score
    }

    return metrics