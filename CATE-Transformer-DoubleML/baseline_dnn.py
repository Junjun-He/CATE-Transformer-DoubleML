import numpy as np
import pandas as pd
import patsy
import doubleml as dml
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
import os
import random
import torch
import torch.nn as nn
from skorch import NeuralNetRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skorch.dataset import ValidSplit
from skorch.callbacks import Callback
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import warnings

# Set global random seed
SEED = 123
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Data reading and preliminary cleaning
file_path = "data/pension.xlsx"
df = pd.read_excel(file_path, header=0)
df['net_tfa'] = df['net_tfa'] / 10000
df = df[df['inc'] > 0]
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
model_data = model_data.astype(np.float32)
# Initialize DoubleMLData (data-backend of DoubleML)
data_dml = dml.DoubleMLData(
    model_data,
    y_col='net_tfa',
    d_cols='e401',
)

# Callback for gradient clipping to prevent exploding gradients
class GradientClippingByNorm(Callback):
    def __init__(self, max_norm=0.5):
        self.max_norm = max_norm
    def on_backward_end(self, net, **kwargs):
        torch.nn.utils.clip_grad_norm_(
            net.module_.parameters(),
            max_norm=self.max_norm
        )

# Custom early stopping callback based on validation loss
class EarlyStopping(Callback):
    def __init__(self, patience=5):
        self.patience = patience
    def on_train_begin(self, net, **kwargs):
        self.best_loss = float('inf')
        self.no_improve_epochs = 0
    def on_epoch_end(self, net, **kwargs):
        current_loss = net.history[-1, 'valid_loss']
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.no_improve_epochs = 0
        else:
            self.no_improve_epochs += 1
            if self.no_improve_epochs >= self.patience:
                net.history.record('stop_early', True)
                raise KeyboardInterrupt

# Learning rate warm-up + cosine annealing scheduler
class WarmupCosineScheduler(Callback):
    def __init__(self):
        pass

    def on_train_begin(self, net, **kwargs):
        self.scheduler = self.get_scheduler(net.optimizer_)

    def on_epoch_end(self, net, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The epoch parameter in `scheduler.step`",
                category=UserWarning,
                module="torch.optim.lr_scheduler"
            )
            self.scheduler.step()
        current_lr = self.scheduler.get_last_lr()[0]
        net.history.record('lr', current_lr)

    def get_scheduler(self, optimizer):
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1e-4,
            end_factor=1.0,
            total_iters=20    # warmup for first 20 epochs
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=180    # rest of epochs follow cosine schedule
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[20]
        )

    # To support pickling by skorch
    def __getstate__(self):
        state = self.__dict__.copy()
        if 'scheduler' in state:
            del state['scheduler']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

# Define deep neural network using PyTorch
class DNNRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.GELU(),

            nn.Linear(64, 32),
            nn.GELU(),

            nn.Linear(32, 1)    # Output layer (linear activation)
        )

    def forward(self, x):
        x = x.to(torch.float32)
        return self.net(x).squeeze(dim=1)

# Determine input feature dimension
input_dim = data_dml.data.drop(columns=['net_tfa', 'e401']).shape[1]

# Build base estimator with skorch
base_net = NeuralNetRegressor(
    module=DNNRegressor,
    module__input_dim=input_dim,
    max_epochs=200,
    lr=1e-4,
    optimizer=torch.optim.AdamW,
    optimizer__weight_decay=1e-4,
    batch_size=256,
    callbacks=[
        GradientClippingByNorm(max_norm=0.5),
        WarmupCosineScheduler(),
        EarlyStopping(patience=5)
    ],
    train_split=ValidSplit(cv=0.2, stratified=False, random_state=SEED),
    verbose=0,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Wrap base network in scikit-learn pipeline with normalization
ml_l = make_pipeline(StandardScaler(), base_net)
ml_m = make_pipeline(StandardScaler(), base_net)
ml_g = make_pipeline(StandardScaler(), base_net)

# Construct DoubleML partially linear regression model
dml_plr_dnn = dml.DoubleMLPLR(data_dml, ml_l=ml_l, ml_m=ml_m, ml_g=ml_g, n_folds=2, score="IV-type")

np.random.seed(123)
dml_plr_dnn.fit(store_predictions=True)
print(dml_plr_dnn.summary)

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
cate_dnn = dml_plr_dnn.cate(spline_basis)
print(cate_dnn)

# CATE effects on the prediction grid
z_seq = np.linspace(Z.min(), Z.max(), 100)
spline_grid = patsy.build_design_matrices(
    [design_info],
    {"z": z_seq, "knots": internal_knots}
)[0]
spline_grid = pd.DataFrame(spline_grid, columns=spline_basis.columns)

df_cate_dnn = cate_dnn.confint(spline_grid, level=0.95, joint=True, n_rep_boot=2000)
df_cate_dnn['z'] = z_seq

# Visualization of CATE and confidence intervals
plt.figure(figsize=(10, 7.5))
plt.plot(df_cate_dnn['z'], df_cate_dnn['effect'], label='Estimated Effect')
plt.fill_between(df_cate_dnn['z'], df_cate_dnn['2.5 %'], df_cate_dnn['97.5 %'],
                 color='b', alpha=0.3, label='95% Confidence Interval')
plt.legend()
plt.title('CATE with 95% Joint Confidence Interval')
plt.xlabel('Log(Income)')
plt.ylabel('Treatment Effect on net_tfa')
plt.grid(True)
plt.show()