uv run dvc stage add --force -n download_datasets \
    -p config/data/amzn.yaml:dataset_name,raw_data_path \
    -p config/data/sent140.yaml:dataset_name,raw_data_path \
    --outs-persist data/raw \
    "uv run python -m src.dataset"