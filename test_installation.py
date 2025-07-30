import torch
from contextualized.regression.lightning_modules import ContextualizedCorrelation
from contextualized.data import CorrelationDataModule
from lightning import Trainer

data = CorrelationDataModule(
    C_train=torch.randn(100, 50),
    C_val=torch.randn(50, 50),
    C_test=torch.randn(50, 50),
    C_predict=torch.randn(50, 50),
    X_train=torch.randn(100, 10),
    X_val=torch.randn(50, 10),
    X_test=torch.randn(50, 10),
    X_predict=torch.randn(50, 10),
    batch_size=4,
)
model = ContextualizedCorrelation(
    context_dim=50,
    x_dim=10,
    encoder_type='mlp',
)
trainer = Trainer(
    max_epochs=1,
    accelerator='auto',
    devices='auto',
)
trainer.fit(model, data)
trainer.test(model, data)
trainer.predict(model, data)
