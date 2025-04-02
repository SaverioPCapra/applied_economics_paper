import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tabulate import tabulate
from scipy import stats
import plotly.graph_objects as go

from tabulate import tabulate
import statsmodels.api as sm
import pandas as pd

from tabulate import tabulate
import statsmodels.api as sm
import pandas as pd

def create_regression_table(data, outcome, predictors_list, control_sets, significance_level=0.05, robust=None):
    """
    Creates a regression table similar to the one in the image.

    Args:
        data (pd.DataFrame): The dataset.
        outcome (str): The name of the outcome variable.
        predictors_list (list): A list of predictors to include in all models.
        control_sets (list of lists): A list of lists, where each inner list represents a
                                        set of control variables to include in the regression.
        significance_level (float): The significance level for marking coefficients.
        robust (str, optional): If "yes", use robust regression (RLM). Defaults to None.

    Returns:
        str: A string representing the formatted table.
    """

    results = {}
    observations = {}  # Store observations for each model
    r_squared = {} #Store r-squared for each model
    for i, control_vars in enumerate(control_sets):
        all_vars = predictors_list + control_vars
        X = data[all_vars]
        X = sm.add_constant(X)  # Add a constant term
        y = data[outcome]

        if robust == "yes":
            model = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()
            r_squared[i] = "N/A (Robust)"
        else:
            model = sm.OLS(y, X).fit(cov_type="HC3")
            r_squared[i] = model.rsquared

        observations[i] = model.nobs
        
        if isinstance(model, sm.regression.linear_model.OLSResults):
            r_squared[i] = model.rsquared
        else:
            r_squared[i] = "N/A (Robust)"

        for predictor in predictors_list + control_vars + ['const']:
            if predictor not in results:
                results[predictor] = {}
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

    table_data.append(["Observations"] + [observations[i] for i in range(len(control_sets))])
    table_data.append(["R-squared"] + [r_squared[i] for i in range(len(control_sets))])
    table = tabulate(table_data, headers=headers, tablefmt="latex")
    return table

def randomization_inference(df, treatment_col, y_col, percentile_input = 5, n_permutations = 1000, robust = None):
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
        if robust == "yes":
            reg = sm.RLM(complete_df[y_col], sm.add_constant(complete_df.drop([y_col], axis = 1)),M=sm.robust.norms.HuberT()).fit()
        else:
            reg = sm.OLS(complete_df[y_col], sm.add_constant(complete_df.drop([y_col], axis = 1))).fit(cov_type="HC3")
        
        treatment_coefficient = reg.params['treated']

        reg_coefficients.append(treatment_coefficient)

    percentile_lower_regcoeff = np.percentile(reg_coefficients,percentile_input)
    percentile_higher_regcoeff = np.percentile(reg_coefficients,100-percentile_input)

    return (percentile_lower_regcoeff, percentile_higher_regcoeff, reg_coefficients)

# Note: this code what generated by Gemini, under a prompt, it was not our original creation

def check_balance(data, treatment_col, covariates, significance_level=0.05):
    """
    Checks balance between treatment and control groups for given covariates.

    Args:
        data (pd.DataFrame): The dataset.
        treatment_col (str): The name of the treatment column (binary: 0 or 1).
        covariates (list): A list of covariate names.
        significance_level (float): The significance level for balance tests.

    Returns:
        pd.DataFrame: A DataFrame showing balance test results.
    """

    results = []
    for covariate in covariates:
        treatment_group = data[data[treatment_col] == 1][covariate]
        control_group = data[data[treatment_col] == 0][covariate]

        if pd.api.types.is_numeric_dtype(data[covariate]):
            # Numeric covariate: t-test
            t_stat, p_value = stats.ttest_ind(treatment_group, control_group, equal_var=False)  # Welch's t-test
            test_type = "Welch's t-test"
            difference = treatment_group.mean() - control_group.mean()
            std_diff = difference / np.sqrt(0.5*(treatment_group.var() + control_group.var()))
        else:
            # Categorical covariate: chi-squared test
            contingency_table = pd.crosstab(data[covariate], data[treatment_col])
            chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
            test_type = "Chi-squared test"
            difference = None
            std_diff = None

        results.append({
            'Covariate': covariate,
            'Test': test_type,
            'p-value': p_value,
            'Significant': p_value < significance_level,
            'Difference': difference,
            'Standardized Difference': std_diff,
        })

    return pd.DataFrame(results)

def plot_2d_clusters(x_pca_rendered, centroids, title="Graph"):
    """
    Plots clusters in 2D using Plotly.

    Args:
        x_pca_rendered (pd.DataFrame): DataFrame with PCA components and cluster labels.
        centroids (np.ndarray): Array of cluster centroids.
        title (str): Title of the plot.
    """

    fig = go.Figure()

    colors = ['red', 'pink', 'green', 'yellow', 'blue']

    for i, cluster in enumerate(x_pca_rendered['cluster'].unique()):
        cluster_data = x_pca_rendered[x_pca_rendered['cluster'] == cluster]
        fig.add_trace(go.Scatter(
            x=cluster_data[0],
            y=cluster_data[1],
            mode='markers',
            marker=dict(color=colors[i]),
            name=f'Cluster {cluster + 1}'
        ))

    fig.add_trace(go.Scatter(
        x=centroids[:, 0],
        y=centroids[:, 1],
        mode='markers',
        marker=dict(size=10, color='black'),
        name='Centroids'
    ))

    fig.update_layout(title=title,
                      xaxis_title='Principal Component 1',
                      yaxis_title='Principal Component 2')

    fig.show()

def plot_3d_clusters(x_pca_rendered, centroids, title="Graph"):
    """
    Plots clusters in 3D using Plotly.

    Args:
        x_pca_rendered (pd.DataFrame): DataFrame with PCA components and cluster labels.
        centroids (np.ndarray): Array of cluster centroids.
        title (str): Title of the plot.
    """

    fig = go.Figure()

    colors = ['red', 'pink', 'green', 'yellow', 'blue']

    for i, cluster in enumerate(x_pca_rendered['cluster'].unique()):
        cluster_data = x_pca_rendered[x_pca_rendered['cluster'] == cluster]
        fig.add_trace(go.Scatter3d(
            x=cluster_data[0],
            y=cluster_data[1],
            z=cluster_data[2],
            mode='markers',
            marker=dict(color=colors[i]),
            name=f'Cluster {cluster + 1}'
        ))

    fig.add_trace(go.Scatter3d(
        x=centroids[:, 0],
        y=centroids[:, 1],
        z=centroids[:, 2],
        mode='markers',
        marker=dict(size=10, color='black'),
        name='Centroids'
    ))

    fig.update_layout(title=title, scene=dict(
        xaxis_title='Principal Component 1',
        yaxis_title='Principal Component 2',
        zaxis_title='Principal Component 3'
    ))

    fig.show()