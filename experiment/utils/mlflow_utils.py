import os
from typing import Literal

import mlflow
import pandas as pd
from tqdm.auto import tqdm


def list_parent_runs(experiment_name: str) -> pd.DataFrame:
    client = mlflow.tracking.MlflowClient()
    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise ValueError(f"Эксперимент '{experiment_name}' не найден.")
        experiment_id = experiment.experiment_id
    except Exception as e:
        print(f"Ошибка: {e}")
        return pd.DataFrame()

    runs = client.search_runs(
        experiment_ids=[experiment_id], filter_string="tags.parent_run = 'True'", order_by=["start_time DESC"]
    )

    run_infos = []
    for run in runs:
        info = {
            "parent_run_id": run.info.run_id,
            "run_name": run.data.tags.get("mlflow.runName"),
            "exp_name": run.data.params.get("exp_name"),
            "dataset": run.data.params.get("dataset"),
            "model": run.data.params.get("model"),
            "optimizer": run.data.params.get("optimizer"),
            "n_runs": run.data.params.get("n_runs"),
        }
        run_infos.append(info)

    return pd.DataFrame(run_infos)


def load_results_for_analysis(
    parent_run_ids: list[str], prediction_type: Literal["auto", "onestep", "both"] = "both"
) -> pd.DataFrame:
    client = mlflow.tracking.MlflowClient()
    all_results = []

    artifact_map = {"auto": "predictions/preds_auto.csv", "onestep": "predictions/preds_onestep.csv"}

    types_to_load = artifact_map.keys() if prediction_type == "both" else [prediction_type]

    for parent_id in tqdm(parent_run_ids, desc="Обработка родительских запусков"):
        parent_run = client.get_run(parent_id)
        parent_params = parent_run.data.params

        child_runs = client.search_runs(
            experiment_ids=client.get_experiment(parent_run.info.experiment_id).experiment_id,
            filter_string=f"tags.'mlflow.parentRunId' = '{parent_id}'",
        )

        for run_number, child_run in enumerate(child_runs):
            for p_type in types_to_load:
                artifact_path = artifact_map[p_type]
                try:
                    local_dir = client.download_artifacts(child_run.info.run_id, os.path.dirname(artifact_path))
                    file_path = os.path.join(local_dir, os.path.basename(artifact_path))

                    if not os.path.exists(file_path):
                        continue

                    df_preds = pd.read_csv(file_path)
                    df_preds["timestep"] = df_preds.index

                    df_preds["parent_run_id"] = parent_id
                    df_preds["child_run_id"] = child_run.info.run_id
                    df_preds["run_number"] = run_number + 1
                    df_preds["prediction_type"] = p_type

                    for key, val in parent_params.items():
                        if key == "datasets":
                            continue
                        df_preds[key] = val

                    for key, val in child_run.data.metrics.items():
                        df_preds[key] = val

                    all_results.append(df_preds)
                except Exception:
                    pass

    if not all_results:
        print("No runs found")
        return pd.DataFrame()

    return pd.concat(all_results, ignore_index=True)
