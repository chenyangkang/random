import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from skorch import NeuralNetClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score 
from skorch.callbacks import EpochScoring
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from skorch.callbacks import LRScheduler
from torch.optim.lr_scheduler import OneCycleLR
from skorch.callbacks import EarlyStopping
from sklearn.calibration import CalibratedClassifierCV
from skorch.dataset import ValidSplit
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F
import pickle 
from imblearn.over_sampling import RandomOverSampler

# Combined Model for Detection and Occupancy
class CombinedModel(nn.Module):
    def __init__(self, input_dim_det, input_dim_occ, latent_size_det=8, latent_size_occ=64):
        super(CombinedModel, self).__init__()
        
        # Detection Sub-network
        self.fc1_det = nn.Linear(input_dim_det, latent_size_det)
        self.fc2_det = nn.Linear(latent_size_det, latent_size_det)
        self.fc_det_out = nn.Linear(latent_size_det, 1)
        
        # Occupancy Sub-network
        self.fc1_occ = nn.Linear(input_dim_occ, latent_size_occ)
        self.fc2_occ = nn.Linear(latent_size_occ, latent_size_occ)
        self.fc_occ_out = nn.Linear(latent_size_occ, 1)
        
        #
        self.temperature_det = nn.Parameter(torch.ones(1))
        self.temperature_occ = nn.Parameter(torch.ones(1))
        
        # Activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, X_det, X_occ):
        
        det_prob = self.predict_detection_probability(X_det)
        occ_prob = self.predict_occupancy_probability(X_occ)
        observation_outcome = occ_prob * det_prob
        return observation_outcome
    
    def predict_detection_probability(self, X_det):
        # Detection Pathway
        x_det = torch.relu(self.fc1_det(X_det))
        x_det = torch.relu(self.fc2_det(x_det))
        x_det = self.fc_det_out(x_det) / self.temperature_det
        det_prob = self.sigmoid(x_det)  # Detection probability
        return det_prob
    
    def predict_occupancy_probability(self, X_occ):
        # Occupancy Pathway
        x_occ = torch.relu(self.fc1_occ(X_occ))
        x_occ = torch.relu(self.fc2_occ(x_occ))
        x_occ = self.fc_occ_out(x_occ) / self.temperature_occ
        occ_prob = self.sigmoid(x_occ)  # Occupancy probability
        return occ_prob
        
class occupancy_ml_trainer():
    def __init__(self, batch_size=128, max_epochs=1000, 
                 latent_size_det=8, latent_size_occ=64,
                 verbose=1, 
                 no_mini_batch=False, validation=False, tolerance_epoch=5, tolerance_threashold=0.001, 
                 scoring='roc_auc',
                 val_split = 0.1,
                 balance_sampling=True,
                 temperature_tuning=False) -> None:
        
        self.batch_size = batch_size
        self.no_mini_batch = no_mini_batch
        self.max_epochs = max_epochs
        self.verbose = verbose
        self.validation = validation
        self.tolerance_epoch = tolerance_epoch
        self.tolerance_threashold = tolerance_threashold
        self.scoring = scoring
        self.latent_size_det=latent_size_det
        self.latent_size_occ=latent_size_occ
        self.val_split = val_split
        self.balance_sampling = balance_sampling
        self.temperature_tuning = temperature_tuning
        
        self.train_metric = EpochScoring(
            scoring=self.scoring,
            lower_is_better=False,
            on_train=True,  # Set to True to calculate AUC on the training set
            name=f'train_{self.scoring}'
        )

        self.valid_metric = EpochScoring(
            scoring=self.scoring,
            lower_is_better=False,
            name=f'valid_{self.scoring}'
        )

        monitor_metric = f'valid_{self.scoring}' if self.validation else f'train_{self.scoring}'
        self.early_stopping = EarlyStopping(
            monitor=monitor_metric,  # Monitor the validation AUC score
            patience=self.tolerance_epoch,          # Number of epochs with no improvement to wait before stopping
            threshold=self.tolerance_threashold,      # Minimum change to consider an improvement
            threshold_mode='rel', # Use a relative change (0.1% improvement) as the threshold
            lower_is_better=False # Higher AUC is better
        )

        self.X_detection_var_normalizer = MinMaxScaler()
        self.X_occupancy_var_normalizer = MinMaxScaler()

    def tune_temperatures(self, X_val, y_val):
        self.model.module_.eval()  # Ensure model is in evaluation mode
        criterion = nn.BCELoss()
        optimizer = torch.optim.LBFGS([self.model.module_.temperature_det, self.model.module_.temperature_occ], lr=0.01, max_iter=50)

        def closure():
            optimizer.zero_grad()
            total_loss = 0.0
            X_det, X_occ = X_val[self.detect_vars], X_val[[col for col in X_val.columns if col not in self.detect_vars]]
            
            # Convert to tensors and apply scaling
            X_det = torch.tensor(self.X_detection_var_normalizer.transform(X_det).astype('float32'))
            X_occ = torch.tensor(self.X_occupancy_var_normalizer.transform(X_occ).astype('float32'))
            y_true = torch.tensor(np.array(y_val).astype('float32')).view(-1, 1)

            # Calculate detection and occupancy probabilities
            det_prob = self.model.module_.predict_detection_probability(X_det)
            occ_prob = self.model.module_.predict_occupancy_probability(X_occ)
            combined_prob = det_prob * occ_prob

            # Calculate loss and backpropagate
            loss = criterion(combined_prob, y_true)
            loss.backward()
            total_loss += loss.item()
            return total_loss

        optimizer.step(closure)
        # print(f"Optimized temperatures: Det - {self.model.module_.temperature_det.item()}, Occ - {self.model.module_.temperature_occ.item()}")

    def upsampler(self, X, y):
        ros = RandomOverSampler(random_state=42)
        X, y = ros.fit_resample(X, np.where(y>0, 1, 0))
        return X, y


    def fit(self, X_train, y_train):
        
        y_train = np.where(y_train>0, 1, 0)
        
        if self.temperature_tuning:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=self.val_split, stratify=y_train, random_state=42
            )
        
        if self.balance_sampling:
            X_train, y_train = self.upsampler(X_train, y_train)
        
        self.detect_vars = [i for i in X_train.columns if i.startswith('detect_')]
        X_train_detection_var_df = X_train[self.detect_vars]
        X_train_detection_var_df = self.X_detection_var_normalizer.fit_transform(X_train_detection_var_df)
        X_train_occupancy_var_df = X_train[[i for i in X_train.columns if not i in self.detect_vars]]
        X_train_occupancy_var_df = self.X_occupancy_var_normalizer.fit_transform(X_train_occupancy_var_df)
        self.detect_var_size = X_train_detection_var_df.shape[1]
        self.occupancy_var_size = X_train_occupancy_var_df.shape[1]
        
        if self.no_mini_batch:
            self.batch_size = X_train.shape[0]
            
        self.lr_scheduler = LRScheduler(
            policy=OneCycleLR
            # max_lr=0.1,  # max learning rate for the cycle
            # steps_per_epoch=len(X_train_detection_var_df) // self.batch_size,  # batch size is 64
            # epochs=self.max_epochs
        )

        # Calculate class weights based on the distribution of the target labels
        class_weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=np.array(y_train).ravel())
        class_weights = torch.tensor(class_weights, dtype=torch.float32)

        # Wrap the model in skorch's NeuralNetClassifier
        self.model = NeuralNetClassifier(
            CombinedModel,
            module__input_dim_det=X_train_detection_var_df.shape[1],
            module__input_dim_occ=X_train_occupancy_var_df.shape[1],
            module__latent_size_det=self.latent_size_det,
            module__latent_size_occ=self.latent_size_occ,
            criterion= nn.BCELoss(weight=class_weights[1]/class_weights[0]),
            optimizer=optim.Adam,
            max_epochs=self.max_epochs,
            lr=0.001,
            batch_size=self.batch_size,
            iterator_train__shuffle=True,
            train_split=ValidSplit(cv=5, stratified=True) if self.validation else None,
            callbacks=[self.train_metric, 
                       EpochScoring(scoring='f1',lower_is_better=False,name=f'train_f1', on_train=True),
                       EpochScoring(scoring='roc_auc',lower_is_better=False,name=f'train_roc_auc', on_train=True),
                       self.valid_metric, 
                       EpochScoring(scoring='f1',lower_is_better=False,name=f'valid_f1', on_train=False),
                       EpochScoring(scoring='roc_auc',lower_is_better=False,name=f'valid_roc_auc', on_train=False),
                       EpochScoring(scoring='recall',lower_is_better=False,name=f'valid_recall', on_train=False),
                       EpochScoring(scoring='precision',lower_is_better=False,name=f'valid_precision', on_train=False),
                       self.early_stopping],
            verbose=self.verbose
        )

        # Train the model
        # model.fit(X_train_detection_var_df.astype('float32'), X_train_occupancy_var_df.astype('float32'), y_train.astype('float32'))
        self.model.fit(
            {"X_det": torch.tensor(np.array(X_train_detection_var_df), dtype=torch.float32), 
            "X_occ": torch.tensor(np.array(X_train_occupancy_var_df), dtype=torch.float32)},
            torch.tensor(np.array(np.where(y_train>0, 1, 0)).reshape(-1,1), dtype=torch.float32)
        )
        
        # Fine-tune temperatures using the split-off validation set
        if self.temperature_tuning:
            self.tune_temperatures(X_val, y_val)
        
        return self
    
    def predict_detection_probability(self, X_det):
        
        if not X_det.shape[1] == self.detect_var_size:
            raise ValueError(f'Input predictor shape is different from training data!')
        
        for col in X_det.columns:
            if not col in self.detect_vars:
                raise ValueError(f'{col} not in self.detect_vars!')
            
        X_det = self.X_detection_var_normalizer.transform(X_det)
            
        with torch.no_grad():
            pred = self.model.module_.predict_detection_probability(torch.from_numpy(np.array(X_det).astype('float32')))
            
        pred = pred.detach().cpu().numpy().reshape(-1,1)
        return pred
    
    def predict_occupancy_probability(self, X_occ):
        
        if not X_occ.shape[1] == self.occupancy_var_size:
            raise ValueError(f'Input predictor shape is different from training data!')
        
        X_occ = self.X_occupancy_var_normalizer.transform(X_occ)
        
        with torch.no_grad():
            pred = self.model.module_.predict_occupancy_probability(torch.from_numpy(np.array(X_occ).astype('float32')))
            
        pred = pred.detach().cpu().numpy()
        return pred
    
    def predict_proba(self, X, stage='combined'):
        """Predicting probability
        
        Args:
            X:
                Prediciton set
            stage:
                One of 'combined', 'detection', or 'occupancy'.
        """
        
        X_detection_var_df = X[self.detect_vars]
        X_occupancy_var_df = X[[i for i in X.columns if not i in self.detect_vars]]
        
        if stage == 'combined':
            proba = self.predict_detection_probability(X_detection_var_df) * self.predict_occupancy_probability(X_occupancy_var_df)
        elif stage == 'detection':
            proba = self.predict_detection_probability(X_detection_var_df)
        elif stage == 'occupancy':
            proba = self.predict_occupancy_probability(X_occupancy_var_df)
        else:
            raise ValueError(f"stage must be one of 'combined', 'detection', or 'occupancy'.")
        
        # proba[:,0] = 1-proba[:,1].flatten()
        proba = np.concatenate([1-proba, proba], axis=1)
        return proba

    def predict(self, X):
        pred_proba = self.predict_proba(X)
        return np.where(pred_proba[:,1]>0.5, 1, 0)
    
    def save(self, path):
        with open(path,'wb') as f:
            pickle.dump(self, f)
            
    @staticmethod
    def load(path, fine_tuning=False):
        with open(path,'rb') as f:
            self = pickle.load(f)
        
        if fine_tuning:

            for param in self.model.module_.parameters():
                param.requires_grad = False

            # Unfreeze only the last layers for fine-tuning
            for param in self.model.module_.fc_det_out.parameters():
                param.requires_grad = True
            for param in self.model.module_.fc_occ_out.parameters():
                param.requires_grad = True
                
            self.model.module_.temperature_det.requires_grad = True
            self.model.module_.temperature_occ.requires_grad = True
        
        return self