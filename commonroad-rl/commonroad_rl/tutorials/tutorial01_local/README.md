# Tutorial 01 local scripts

This folder contains standalone Python files equivalent to the code snippets in:
- `Tutorial 01 - Data Preprocessing.ipynb`

## Files

- `00_prepare.py`: environment and path checks
- `01_clone_dataset_converters.py`: clone/install dataset-converters
- `02_convert_highd_csv_to_xml.py`: convert highD raw CSV to XML
- `03_validate_xml.py`: validate XML against XSD
- `04_visualize_sample.py`: visualize one XML scenario
- `05_xml_to_pickle.py`: convert XML to pickle dataset
- `06_split_train_test.py`: split pickles into train/test
- `07_split_multi_env.py`: optional split for multi-env training
- `run_all.py`: run the main pipeline in sequence

## Usage

Run from anywhere:

```bash
python commonroad_rl/tutorials/tutorial01_local/00_prepare.py
python commonroad_rl/tutorials/tutorial01_local/01_clone_dataset_converters.py
python commonroad_rl/tutorials/tutorial01_local/02_convert_highd_csv_to_xml.py
python commonroad_rl/tutorials/tutorial01_local/03_validate_xml.py
python commonroad_rl/tutorials/tutorial01_local/05_xml_to_pickle.py
python commonroad_rl/tutorials/tutorial01_local/06_split_train_test.py
```

Optional:

```bash
python commonroad_rl/tutorials/tutorial01_local/04_visualize_sample.py --steps 80
python commonroad_rl/tutorials/tutorial01_local/07_split_multi_env.py --num-folders 5
```

Or execute the main non-visual pipeline:

```bash
python commonroad_rl/tutorials/tutorial01_local/run_all.py
```
