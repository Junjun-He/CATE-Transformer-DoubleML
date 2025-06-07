import numpy as np
import pandas as pd
import patsy
import doubleml as dml
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm

# Data reading and preliminary cleaning
file_path = "data/pension.xlsx"
df = pd.read_excel(file_path, header=0)
df['net_tfa'] = df['net_tfa'] / 10000
df['inc'] = np.log(df['inc'])
df = df[df['inc'].notna() & np.isfinite(df['inc'])]
df = df.dropna(subset=['net_tfa', 'e401'])

# Feature engineering
# Set up a model according to regression formula with polynomials
features = df.copy()[['marr', 'twoearn', 'db', 'pira', 'hown']]

poly_dict = {'age': 6,
             'inc': 8,
             'educ': 4,
             'fsize': 2}
for key, degree in poly_dict.items():
    poly = PolynomialFeatures(degree, include_bias=False)
    data_transf = poly.fit_transform(df[[key]])
    x_cols = poly.get_feature_names_out([key])
    data_transf = pd.DataFrame(data_transf, columns=x_cols)

    features = pd.concat((features, data_transf),
                          axis=1, sort=False)
model_data = pd.concat((df.copy()[['net_tfa', 'e401']], features.copy()),
                        axis=1, sort=False)
model_data = model_data.dropna()
# Initialize DoubleMLData (data-backend of DoubleML)
data_dml = dml.DoubleMLData(
    model_data,
    y_col='net_tfa',
    d_cols='e401',
)

# First stage estimation
ml_l = RandomForestRegressor(
    n_estimators=1500,
    max_depth=25,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=0.8,
    bootstrap=True,
    random_state=123,
    verbose=0,
    n_jobs=-1
)

ml_m = RandomForestRegressor(
    n_estimators=1500,
    max_depth=25,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=0.8,
    bootstrap=True,
    random_state=123,
    verbose=0,
    n_jobs=-1
)

ml_g = RandomForestRegressor(
    n_estimators=1500,
    max_depth=25,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=0.8,
    bootstrap=True,
    random_state=123,
    verbose=0,
    n_jobs=-1
)

dml_plr_forest = dml.DoubleMLPLR(data_dml, ml_l=ml_l, ml_m=ml_m, ml_g=ml_g, n_folds=2, score="IV-type")
np.random.seed(123)
dml_plr_forest.fit(store_predictions=True)
print(dml_plr_forest.summary)
dml_plr_forest = dml.DoubleMLPLR(data_dml, ml_l=ml_l, ml_m=ml_m, ml_g=ml_g, n_folds=2, score="IV-type")
np.random.seed(123)
dml_plr_forest.fit(store_predictions=True)
print(dml_plr_forest.summary)

# Spline fitting function (automatic selection of node number)
def least_squares_splines(Z, Y, max_knot=9, norder=3, nderiv=0):
    Z = np.asarray(Z).reshape(-1)
    Y = np.asarray(Y).reshape(-1)
    cv_errors = []
    degree = norder - 1
    for knot_num in range(2, max_knot + 1):
        breaks = np.quantile(Z, np.linspace(0, 1, knot_num + 1))
        boundary_knots = [breaks[0], breaks[-1]]
        internal_knots = breaks[1:-1]
        Z_spline = patsy.dmatrix(
            f"bs(z, knots=knots, degree={degree}, include_intercept=False, lower_bound={boundary_knots[0]}, upper_bound={boundary_knots[1]})",
            {"z": Z, "knots": internal_knots},
            return_type='dataframe'
        )
        model = sm.OLS(Y, Z_spline).fit()
        residuals = model.resid
        hat_diag = model.get_influence().hat_matrix_diag
        cv_residuals = residuals / (1 - hat_diag)
        cv_errors.append(np.sum(cv_residuals ** 2))
    best_knot_num = np.argmin(cv_errors) + 2
    best_breaks = np.quantile(Z, np.linspace(0, 1, best_knot_num + 1))
    internal_knots = best_breaks[1:-1]
    boundary_knots = [best_breaks[0], best_breaks[-1]]
    Z_spline_best = patsy.dmatrix(
        f"bs(z, knots=knots, degree={degree}, include_intercept=False, "
        f"lower_bound={boundary_knots[0]}, upper_bound={boundary_knots[1]})",
        {"z": Z, "knots": internal_knots},
        return_type='dataframe'
    )
    model_best = sm.OLS(Y, Z_spline_best).fit()
    return {
        'cv_knot': best_knot_num,
        'fit': model_best,
        'breaks': best_breaks,
        'internal_knots': internal_knots,
        'Z_spline': Z_spline_best,
        'design_info': Z_spline_best.design_info
    }

# CATE analysis: estimation of heterogeneous treatment effects
Z = data_dml.data['inc']
Y = data_dml.data['net_tfa']
spline_result = least_squares_splines(Z.values, Y.values)
spline_basis = spline_result['Z_spline']
design_info = spline_result['design_info']
internal_knots = spline_result['internal_knots']

# Estimate CATE and output confidence intervals
cate_rf = dml_plr_forest.cate(spline_basis)
print(cate_rf)

# CATE effects on the prediction grid
z_seq = np.linspace(Z.min(), Z.max(), 100)
spline_grid = patsy.build_design_matrices(
    [design_info],
    {"z": z_seq, "knots": internal_knots}
)[0]
spline_grid = pd.DataFrame(spline_grid, columns=spline_basis.columns)

df_cate_rf = cate_rf.confint(spline_grid, level=0.95, joint=True, n_rep_boot=2000)
df_cate_rf['z'] = z_seq

# Visualization of CATE and confidence intervals
plt.figure(figsize=(10, 7.5))
plt.plot(df_cate_rf['z'], df_cate_rf['effect'], label='Estimated Effect')
plt.fill_between(df_cate_rf['z'], df_cate_rf['2.5 %'], df_cate_rf['97.5 %'],
                 color='b', alpha=0.3, label='95% Confidence Interval')
plt.legend()
plt.title('CATE with 95% Joint Confidence Interval')
plt.xlabel('Log(Income)')
plt.ylabel('Treatment Effect on net_tfa')
plt.grid(True)
plt.show()