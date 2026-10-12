from src.skeleton_toy import formal,dependency_test,attention_heatmap
from src.poseformer_lite import test

model, X_eval, Y_eval = formal()

attention_heatmap(
    model,
    X_eval
)