from pathlib import Path

project_dir = Path(__file__).parent.parent

raw_data_dir = project_dir/'data/raw'
processed_data_dir = project_dir/'data/processed'
external_data_dir = project_dir/'data/external'

train_models_dir = project_dir/'models/trained_models'
hypertuned_models_dir = project_dir/'models/hypertuned_models'

total_features = ['age', 'balance', 'day', 'duration', 'pdays', 'marital', 'contact', 'poutcome', 'job', 'education', 'month', 'default', 'housing', 'loan', 'campaign', 'previous']
bin_features = ['default', 'housing', 'loan']
# ord_features = ['education', 'month', 'job']
ord_features = ['education', 'month']
# cat_features = ['marital', 'contact', 'poutcome']
cat_features = ['marital', 'contact', 'poutcome', 'job']
num_features = ['age', 'balance', 'day', 'duration', 'pdays']