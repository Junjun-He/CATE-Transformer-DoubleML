import numpy as np
import pandas as pd
import patsy
import doubleml as dml
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
    features = pd.concat((features, data_transf), axis=1, sort=False)

model_data = pd.concat((df.copy()[['net_tfa', 'e401']], features.copy()), axis=1, sort=False)
model_data = model_data.dropna()
model_data = model_data.astype(np.float32)

# Initialize DoubleMLData
data_dml = dml.DoubleMLData(
    model_data,
    y_col='net_tfa',
    d_cols='e401',
)

# Learnable Positional Encoding Module
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, feature_num, embed_dim):
        super().__init__()
        # Initialize a learnable positional embedding tensor
        self.pos_embedding = nn.Parameter(torch.zeros(1, feature_num, embed_dim))
        # Apply normal initialization to the positional embedding
        nn.init.normal_(self.pos_embedding, mean=0, std=0.02)

    def forward(self, x):
        # Add positional encoding to the input tensor
        return x + self.pos_embedding[:, :x.size(1), :]

# Custom Transformer Encoder with Residual Connections and Layer Normalization
class ResidualTransformer(nn.Module):
    def __init__(self, encoder_layer, num_layers, d_model):
        super().__init__()
        # Stack multiple TransformerEncoder blocks
        self.layers = nn.ModuleList([
            nn.TransformerEncoder(encoder_layer, num_layers=1)
            for _ in range(num_layers)
        ])
        # Layer normalization after each residual connection
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        # Apply each encoder layer with residual connection and layer norm
        for layer, norm in zip(self.layers, self.layer_norms):
            residual = x
            x = layer(x)
            x = norm(x + residual)    # Residual connection followed by normalization
        return x

# Transformer-based Regression Model
class TransformerRegressor(nn.Module):
    def __init__(self, feature_num, embed_dim=128, num_heads=4, ff_dim=512, num_layers=4, dropout=0.2):
        super().__init__()

        # Feature embedding: map scalar features to high-dimensional space
        self.feature_embedding = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim)
        )

        # Learnable positional encoding
        self.pos_encoder = LearnablePositionalEncoding(feature_num, embed_dim)
        # Layer normalization for input
        self.input_norm = nn.LayerNorm(embed_dim)

        # Define a single Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )

        # Stack multiple Transformer layers with residual connections
        self.transformer = ResidualTransformer(encoder_layer, num_layers=num_layers, d_model=embed_dim)

        # Adaptive average pooling to reduce sequence dimension
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Output regression head: MLP with GELU activation and dropout
        self.output_layer = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: [batch_size, feature_num]
        x = x.unsqueeze(-1)    # Reshape to [batch_size, feature_num, 1]
        x = self.feature_embedding(x)    # Embed each scalar feature
        x = self.pos_encoder(x)    # Add positional information
        x = self.input_norm(x)    # Normalize input
        x = self.transformer(x)    # Apply Transformer layers

        x = x.permute(0, 2, 1)    # Prepare for pooling: [B, C, T]
        x = self.pool(x).squeeze(-1)    # Global average pooling: [B, C]
        return self.output_layer(x)    # Final regression output: [B, 1]

# Safe Kaiming Normal Initialization (only applies to tensors with >=2 dims)
def safe_kaiming_normal_(tensor):
    if tensor.dim() >= 2:
        nn.init.kaiming_normal_(tensor, mode='fan_out')

# Custom NeuralNetRegressor wrapper for skorch
class CustomNeuralNetRegressor(NeuralNetRegressor):
    """
    A custom skorch wrapper to support specialized training steps
    or initialization for complex models like Transformer.
    """
    def train_step(self, batch, **fit_params):
        # Unpack batch into input X and target y
        X, y = batch
        # Reshape y if it's a 1D tensor to make it compatible with regression loss
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        y = y.float()
        # Delegate to parent train_step method
        return super().train_step((X, y), **fit_params)

    def get_loss(self, y_pred, y_true, *args, **kwargs):
        # Ensure y_true has the correct shape for loss computation
        if y_true.ndim == 1:
            y_true = y_true.view(-1, 1)
        return super().get_loss(y_pred, y_true, *args, **kwargs)

    def predict(self, X):
        # Override to return 1D predictions
        preds = super().predict(X)
        return preds.reshape(-1)

# Callback for gradient clipping by norm (to stabilize training)
class GradientClippingByNorm(Callback):
    def __init__(self, max_norm=1.0, norm_type=2):
        self.max_norm = max_norm
        self.norm_type = norm_type

    def on_backward_end(self, net, **kwargs):
        # Clip gradients of model parameters to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(
            net.module_.parameters(),
            max_norm=self.max_norm,
            norm_type=self.norm_type
        )

# Factory function to return a skorch-compatible Transformer regressor with callbacks
def get_skorch_transformer(feature_num):

    # Callback for learning rate warm-up and cosine decay schedule
    class WarmupCosineScheduler(Callback):
        def __init__(self):
            pass

        def on_train_begin(self, net, **kwargs):
            # Create scheduler after optimizer is initialized
            self.scheduler = self.get_scheduler(net.optimizer_)

        def on_epoch_end(self, net, **kwargs):
            # Step the scheduler at the end of each epoch
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="The epoch parameter in `scheduler.step`",
                    category=UserWarning,
                    module="torch.optim.lr_scheduler"
                )
                self.scheduler.step()
            # Record the current learning rate in training history
            current_lr = self.scheduler.get_last_lr()[0]
            net.history.record('lr', current_lr)

        def get_scheduler(self, optimizer):
            # Linear warm-up for the first 20 iterations
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=1e-4,
                end_factor=1.0,
                total_iters=20
            )
            # Cosine annealing for the rest of training
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=180    # Total number of epochs
            )
            # Combine warm-up and cosine scheduler
            return SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[20]
            )

        def __getstate__(self):
            # Remove non-serializable parts before pickling
            state = self.__dict__.copy()
            if 'scheduler' in state:
                del state['scheduler']
            return state

        def __setstate__(self, state):
            # Restore state after unpickling
            self.__dict__.update(state)

    # Monitoring gradients after backpropagation (e.g., for logging/debugging)
    class GradientMonitor(Callback):
        def on_backward_end(self, net, **kwargs):
            # Could be extended to log or visualize gradient stats
            grads = []
            for param in net.module_.parameters():
                if param.grad is not None:
                    grads.append(param.grad.norm().item())
            net.history.record('grad_norm', np.mean(grads))

    # Early stopping callback to halt training when validation loss stops improving
    class EarlyStopping(Callback):
        def __init__(self, patience=5, threshold=1e-4):
            self.patience = patience    # Number of epochs to wait after last improvement
            self.threshold = threshold    # Minimum change to qualify as an improvement
        def on_train_begin(self, net, **kwargs):
            self.best_loss = float('inf')    # Initialize best loss as infinity
            self.epochs_no_improve = 0     # Counter for epochs without improvement
        def on_epoch_end(self, net, **kwargs):
            current_loss = net.history[-1, 'valid_loss']    # Get latest validation loss
            # Check if loss has improved by at least the threshold
            if current_loss < self.best_loss - self.threshold:
                self.best_loss = current_loss
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1

            # If no improvement for 'patience' epochs, stop training
            if self.epochs_no_improve >= self.patience:
                print(f"Early stopping at epoch {len(net.history)}")
                raise KeyboardInterrupt    # Interrupt training

    # Return a fully configured CustomNeuralNetRegressor instance for Transformer
    return CustomNeuralNetRegressor(
        module=TransformerRegressor,
        module__feature_num=feature_num,    # Number of input features
        module__embed_dim=128,    # Embedding dimension for each feature
        module__num_heads=4,    # Number of attention heads
        module__num_layers=4,    # Number of transformer layers
        module__ff_dim=512,    # Hidden layer size in feedforward network
        module__dropout=0.2,    # Dropout rate
        max_epochs=200,    # Max number of training epochs
        lr=1e-4,    # Learning rate
        batch_size=256,    # Training batch size
        optimizer=torch.optim.AdamW,    # Optimizer (AdamW recommended for transformers)
        optimizer__weight_decay=1e-4,    # L2 regularization
        callbacks=[
            # Weight and bias initialization
            Initializer('*.weight', fn=safe_kaiming_normal_),
            Initializer('*.bias', fn=nn.init.zeros_),
            # Gradient clipping to stabilize training
            GradientClippingByNorm(max_norm=0.5),
            # Learning rate scheduling with warm-up and cosine annealing
            WarmupCosineScheduler(),
            # Gradient monitoring callback (to be defined)
            GradientMonitor(),
            # Early stopping
            EarlyStopping(patience=5)
        ],
        train_split=ValidSplit(cv=0.2, stratified=False, random_state=SEED),    # 80/20 validation split
        verbose=0,
        device='cuda' if torch.cuda.is_available() else 'cpu',    # Use GPU if available
    )

# Determine input feature dimension
input_dim = data_dml.data.drop(columns=['net_tfa', 'e401']).shape[1]

ml_l = make_pipeline(
    StandardScaler(),    # Standardize features
    get_skorch_transformer(input_dim),    # Transformer regressor wrapped by skorch
)
ml_m = make_pipeline(
    StandardScaler(),
    get_skorch_transformer(input_dim),
)
ml_g = make_pipeline(
    StandardScaler(),
    get_skorch_transformer(input_dim),
)

# Construct DoubleML partially linear regression model
dml_plr_transformer = dml.DoubleMLPLR(data_dml, ml_l=ml_l, ml_m=ml_m, ml_g=ml_g, n_folds=2, score="IV-type")

np.random.seed(SEED)
dml_plr_transformer.fit(store_predictions=True)
print(dml_plr_transformer.summary)

# Spline fitting function
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
            f"bs(z, knots=knots, degree={degree}, include_intercept=False, "
            f"lower_bound={boundary_knots[0]}, upper_bound={boundary_knots[1]})",
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

# CATE analysis
Z = data_dml.data['inc']
Y = data_dml.data['net_tfa']
spline_result = least_squares_splines(Z.values, Y.values)
spline_basis = spline_result['Z_spline']
design_info = spline_result['design_info']
internal_knots = spline_result['internal_knots']

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

plt.figure(figsize=(10, 7.5))
plt.plot(df_cate_transformer['z'], df_cate_transformer['effect'], label='Estimated Effect')
plt.fill_between(df_cate_transformer['z'], df_cate_transformer['2.5 %'], df_cate_transformer['97.5 %'],
                 color='b', alpha=0.3, label='95% Confidence Interval')
plt.legend()
plt.title('CATE with 95% Joint Confidence Interval')
plt.xlabel('Log(Income)')
plt.ylabel('Treatment Effect on net_tfa')
plt.grid(True)
plt.show()