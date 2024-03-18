docker cp main.py b1722c480045:/workspace_2/code/
docker cp preprocessing_functions_Sophie.py b1722c480045:/workspace_2/code/
docker cp modeling_functions.py b1722c480045:/workspace_2/code/

docker cp ../data/harmonized.xlsx b1722c480045:/workspace_2/data/
docker cp ../data/first_72.xlsx b1722c480045:/workspace_2/data/
docker cp ../data/first_168.xlsx b1722c480045:/workspace_2/data/
docker cp ../data/bt_72_168.xlsx b1722c480045:/workspace_2/data/
# docker cp ../results/notes_sample_preprocessed.csv b1722c480045:/workspace_2/results/

# docker cp b1722c480045:/workspace/code/y_train_mrs_3patterns.csv .

# docker exec -it mimic_NLP_test bash