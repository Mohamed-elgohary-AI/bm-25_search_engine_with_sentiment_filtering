uv run dvc stage add --force -n download_datasets \
    -p config/data/amzn.yaml:dataset_name,raw_data_path \
    -p config/data/sent140.yaml:dataset_name,raw_data_path \
    -o data/raw \
    "uv run python -m src.dataset"dd

uv run dvc stage add --force \
  -n train_bert \
  -d data/processed/reviews.parquet \
  -d config/model/bert.yaml \
  -d src/modeling/train.py \
  -o models/bert-sentiment \
  -p config/model/bert.yaml: \
  --cmd "uv run python src/modeling/train.py"