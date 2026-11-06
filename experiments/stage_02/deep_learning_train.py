"""
Alias entrypoint for Deep Learning Training suite.
Redirects to nuron_train.py.
"""
try:
    from experiments.stage_02.nuron_train import train_and_evaluate_all
except (ImportError, ModuleNotFoundError):
    from nuron_train import train_and_evaluate_all

if __name__ == "__main__":
    train_and_evaluate_all()

