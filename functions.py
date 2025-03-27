import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tabulate import tabulate

def create_regression_table(data, outcome, predictors_list, control_sets, significance_level=0.05):
    """
    Creates a regression table similar to the one in the image.

    Args:
        data (pd.DataFrame): The dataset.
        outcome (str): The name of the outcome variable.
        predictors_list (list): A list of predictors to include in all models.
        control_sets (list of lists): A list of lists, where each inner list represents a
                                      set of control variables to include in the regression.
        significance_level (float): The significance level for marking coefficients.

    Returns:
        str: A string representing the formatted table.
    """

    results = {}
    for i, control_vars in enumerate(control_sets):
        all_vars = predictors_list + control_vars
        X = data[all_vars]
        X = sm.add_constant(X)  # Add a constant term
        y = data[outcome]

        model = sm.OLS(y, X).fit(cov_type= "HC3")

        for predictor in predictors_list + control_vars + ['const']:
            if predictor not in results:
                results[predictor] = {}
            if predictor == 'const':
                predictor_name = 'Constant'
            else:
                predictor_name = predictor
            if i not in results[predictor]:
                results[predictor][i] = {}

            if predictor in model.params:
                results[predictor][i]['coef'] = model.params[predictor]
                results[predictor][i]['se'] = model.bse[predictor]
                p_value = model.pvalues[predictor]
                if p_value < 0.01:
                    results[predictor][i]['sig'] = '***'
                elif p_value < 0.05:
                    results[predictor][i]['sig'] = '**'
                elif p_value < 0.10:
                    results[predictor][i]['sig'] = '*'
                else:
                    results[predictor][i]['sig'] = ''
            else:
                results[predictor][i]['coef'] = ''
                results[predictor][i]['se'] = ''
                results[predictor][i]['sig'] = ''

        if i == 0:
            observations = model.nobs
            r_squared = model.rsquared

    table_data = []
    headers = ["Variables"] + [f"({i+1})" for i in range(len(control_sets))]

    for predictor, values in results.items():
        row = [predictor]
        for i in range(len(control_sets)):
            coef = values.get(i, {}).get('coef', '')
            se = values.get(i, {}).get('se', '')
            sig = values.get(i, {}).get('sig', '')
            if coef != '':
                row.append(f"{coef:.3f}{sig}\n({se:.3f})")
            else:
                row.append('')
        table_data.append(row)

    table_data.append(["Observations"] + [observations for _ in range(len(control_sets))])
    table_data.append(["R-squared"] + [r_squared for _ in range(len(control_sets))])
    table = tabulate(table_data, headers=headers, tablefmt="pipe")
    return table

def randomization_inference(df, treatment_col, y_col, percentile_input = 5, n_permutations = 1000):
    # Number of permutations that I will run to carry out the randomization inference
    num_permutations = n_permutations

    df_notreatment = df.copy()
    df_notreatment = df_notreatment.drop([treatment_col], axis = 1)

    n_subjects = df[treatment_col].sum()

    reg_coefficients = []

    for i in range(num_permutations):
        # Creates a permuted version of the DataFrame by randomly shuffling the rows without replacement
        permuted_df = df_notreatment.sample(frac=1, replace=False)

        # I assign to this variable the number of units to which I will have to assign the treatment in the MC simulation 
        # n_treatment = number of subjects that received treatment
        n_treatment = n_subjects

        # I assign to the first "n_treatment" rows of the permuted dataframe the treatment
        treatment_df = permuted_df.copy().iloc[:n_treatment]
        treatment_df["treated"] =  1

        # I leave the remaining entries untreated
        no_treatment_df = permuted_df.copy().iloc[n_treatment:]
        no_treatment_df["treated"] = 0

        # I concatenate the treatment and non-treatment dataframe
        # I do this, so that then I can run my function to estimate the coefficient of the treatment dummy
        complete_df = pd.concat([treatment_df, no_treatment_df], axis = 0)

        # I estimate the treatment coefficient
        reg = sm.OLS(complete_df[y_col], sm.add_constant(complete_df.drop([y_col], axis = 1))).fit(cov_type="HC3")
        treatment_coefficient = reg.params['treated']

        reg_coefficients.append(treatment_coefficient)

    percentile_lower_regcoeff = np.percentile(reg_coefficients,percentile_input)
    percentile_higher_regcoeff = np.percentile(reg_coefficients,100-percentile_input)

    return (percentile_lower_regcoeff, percentile_higher_regcoeff, reg_coefficients)

        