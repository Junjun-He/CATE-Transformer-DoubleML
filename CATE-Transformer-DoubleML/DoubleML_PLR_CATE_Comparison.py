import numpy as np
import pandas as pd
import patsy
import doubleml as dml
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
import os
import random
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skorch import NeuralNetRegressor
import torch
import torch.nn as nn
from skorch.dataset import ValidSplit
from skorch.callbacks import Initializer, Callback
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

# First stage estimation
# Random Forest
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

#DNN
class GradientClippingByNorm(Callback):
    def __init__(self, max_norm=0.5):
        self.max_norm = max_norm
    def on_backward_end(self, net, **kwargs):
        torch.nn.utils.clip_grad_norm_(
            net.module_.parameters(),
            max_norm=self.max_norm
        )

# 添加早停回调
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
            total_iters=20
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=180
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[20]
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        if 'scheduler' in state:
            del state['scheduler']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

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

            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = x.to(torch.float32)
        return self.net(x).squeeze(dim=1)

input_dim = data_dml.data.drop(columns=['net_tfa', 'e401']).shape[1]

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

ml_l = make_pipeline(StandardScaler(), base_net)
ml_m = make_pipeline(StandardScaler(), base_net)
ml_g = make_pipeline(StandardScaler(), base_net)

dml_plr_dnn = dml.DoubleMLPLR(data_dml, ml_l=ml_l, ml_m=ml_m, ml_g=ml_g, n_folds=2, score="IV-type")
np.random.seed(123)
dml_plr_dnn.fit(store_predictions=True)
print(dml_plr_dnn.summary)

#transformer
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, feature_num, embed_dim):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, feature_num, embed_dim))
        nn.init.normal_(self.pos_embedding, mean=0, std=0.02)

    def forward(self, x):
        return x + self.pos_embedding[:, :x.size(1), :]

class ResidualTransformer(nn.Module):
    def __init__(self, encoder_layer, num_layers, d_model):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoder(encoder_layer, num_layers=1)
            for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer, norm in zip(self.layers, self.layer_norms):
            residual = x
            x = layer(x)
            x = norm(x + residual)
        return x

class TransformerRegressor(nn.Module):
    def __init__(self, feature_num, embed_dim=128, num_heads=4, ff_dim=512, num_layers=4, dropout=0.2):
        super().__init__()

        self.feature_embedding = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim)
        )

        self.pos_encoder = LearnablePositionalEncoding(feature_num, embed_dim)
        self.input_norm = nn.LayerNorm(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )

        self.transformer = ResidualTransformer(encoder_layer, num_layers=num_layers, d_model=embed_dim)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.output_layer = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.feature_embedding(x)
        x = self.pos_encoder(x)
        x = self.input_norm(x)
        x = self.transformer(x)
        x = x.permute(0, 2, 1)
        x = self.pool(x).squeeze(-1)
        return self.output_layer(x)

def safe_kaiming_normal_(tensor):
    if tensor.dim() >= 2:
        nn.init.kaiming_normal_(tensor, mode='fan_out')

class CustomNeuralNetRegressor(NeuralNetRegressor):
    def train_step(self, batch, **fit_params):
        X, y = batch
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        y = y.float()
        return super().train_step((X, y), **fit_params)

    def get_loss(self, y_pred, y_true, *args, **kwargs):
        if y_true.ndim == 1:
            y_true = y_true.view(-1, 1)
        return super().get_loss(y_pred, y_true, *args, **kwargs)

    def predict(self, X):
        preds = super().predict(X)
        return preds.reshape(-1)

class GradientClippingByNorm(Callback):
    def __init__(self, max_norm=1.0, norm_type=2):
        self.max_norm = max_norm
        self.norm_type = norm_type

    def on_backward_end(self, net, **kwargs):
        torch.nn.utils.clip_grad_norm_(
            net.module_.parameters(),
            max_norm=self.max_norm,
            norm_type=self.norm_type
        )

def get_skorch_transformer(feature_num):
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
                total_iters=20
            )
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=180
            )
            return SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[20]
            )

        def __getstate__(self):
            state = self.__dict__.copy()
            if 'scheduler' in state:
                del state['scheduler']
            return state

        def __setstate__(self, state):
            self.__dict__.update(state)

    class GradientMonitor(Callback):
        def on_backward_end(self, net, **kwargs):
            grads = []
            for param in net.module_.parameters():
                if param.grad is not None:
                    grads.append(param.grad.norm().item())
            net.history.record('grad_norm', np.mean(grads))

    class EarlyStopping(Callback):
        def __init__(self, patience=5, threshold=1e-4):
            self.patience = patience
            self.threshold = threshold
        def on_train_begin(self, net, **kwargs):
            self.best_loss = float('inf')
            self.epochs_no_improve = 0
        def on_epoch_end(self, net, **kwargs):
            current_loss = net.history[-1, 'valid_loss']
            if current_loss < self.best_loss - self.threshold:
                self.best_loss = current_loss
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1

            if self.epochs_no_improve >= self.patience:
                print(f"Early stopping at epoch {len(net.history)}")
                raise KeyboardInterrupt

    return CustomNeuralNetRegressor(
        module=TransformerRegressor,
        module__feature_num=feature_num,
        module__embed_dim=128,
        module__num_heads=4,
        module__num_layers=4,
        module__ff_dim=512,
        module__dropout=0.2,
        max_epochs=200,
        lr=1e-4,
        batch_size=256,
        optimizer=torch.optim.AdamW,
        optimizer__weight_decay=1e-4,
        callbacks=[
            Initializer('*.weight', fn=safe_kaiming_normal_),
            Initializer('*.bias', fn=nn.init.zeros_),
            GradientClippingByNorm(max_norm=0.5),
            WarmupCosineScheduler(),
            GradientMonitor(),
            EarlyStopping(patience=5)
        ],
        train_split=ValidSplit(cv=0.2, stratified=False, random_state=SEED),
        verbose=1,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )

input_dim = data_dml.data.drop(columns=['net_tfa', 'e401']).shape[1]

ml_l = make_pipeline(
    StandardScaler(),
    get_skorch_transformer(input_dim),
)
ml_m = make_pipeline(
    StandardScaler(),
    get_skorch_transformer(input_dim),
)
ml_g = make_pipeline(
    StandardScaler(),
    get_skorch_transformer(input_dim),
)

dml_plr_transformer = dml.DoubleMLPLR(data_dml, ml_l=ml_l, ml_m=ml_m, ml_g=ml_g, n_folds=2, score="IV-type")
np.random.seed(SEED)
dml_plr_transformer.fit(store_predictions=True)
print(dml_plr_transformer.summary)

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
#randomforest
cate_rf = dml_plr_forest.cate(spline_basis)
print(cate_rf)
z_seq = np.linspace(Z.min(), Z.max(), 100)
spline_grid = patsy.build_design_matrices(
    [design_info],
    {"z": z_seq, "knots": internal_knots}
)[0]
spline_grid = pd.DataFrame(spline_grid, columns=spline_basis.columns)

df_cate_rf = cate_rf.confint(spline_grid, level=0.95, joint=True, n_rep_boot=2000)
df_cate_rf['z'] = z_seq

#DNN
cate_dnn = dml_plr_dnn.cate(spline_basis)
print(cate_dnn)
z_seq = np.linspace(Z.min(), Z.max(), 100)
spline_grid = patsy.build_design_matrices(
    [design_info],
    {"z": z_seq, "knots": internal_knots}
)[0]
spline_grid = pd.DataFrame(spline_grid, columns=spline_basis.columns)

df_cate_dnn = cate_dnn.confint(spline_grid, level=0.95, joint=True, n_rep_boot=2000)
df_cate_dnn['z'] = z_seq

#transformer
cate_transformer = dml_plr_transformer.cate(spline_basis)
print(cate_transformer)
z_seq = np.linspace(Z.min(), Z.max(), 100)
spline_grid = patsy.build_design_matrices(
    [design_info],
    {"z": z_seq, "knots": internal_knots}
)[0]
spline_grid = pd.DataFrame(spline_grid, columns=spline_basis.columns)

df_cate_transformer = cate_transformer.confint(spline_grid, level=0.95, joint=True, n_rep_boot=2000)
df_cate_transformer['z'] = z_seq

# Visualization of CATE and confidence intervals
plt.figure(figsize=(10, 7.5))
plt.plot(df_cate_dnn['z'], df_cate_dnn['effect'], label='DNN', color='blue')
plt.fill_between(df_cate_dnn['z'], df_cate_dnn['2.5 %'], df_cate_dnn['97.5 %'], color='blue', alpha=0.2)

plt.plot(df_cate_rf['z'], df_cate_rf['effect'], label='Random Forest', color='green')
plt.fill_between(df_cate_rf['z'], df_cate_rf['2.5 %'], df_cate_rf['97.5 %'], color='green', alpha=0.2)

plt.plot(df_cate_transformer['z'], df_cate_transformer['effect'], label='Transformer', color='red')
plt.fill_between(df_cate_transformer['z'], df_cate_transformer['2.5 %'], df_cate_transformer['97.5 %'], color='red', alpha=0.2)

plt.legend()
plt.title('Comparison of CATE Estimates by Model')
plt.xlabel('Log(Income)')
plt.ylabel('Treatment Effect on net_tfa')
plt.grid(True)
plt.show()