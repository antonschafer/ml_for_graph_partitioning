import os
from pymongo import MongoClient
import numpy as np
import gridfs
from mongo.credentials import mongo_url, db_name
client = MongoClient(mongo_url)
db = client[db_name]
fs = gridfs.GridFS(db)


def get_run(run_id):
    try:
        return db.runs.find({"_id": run_id})[0]
    except:
        return None


def get_metrics(run_id):
    try:
        all_metrics = db.metrics.find({"run_id": run_id})
        return {metric["name"]: metric for metric in all_metrics}
    except:
        return None


def get_results(run_id, by="val loss"):
    metrics = get_metrics(run_id)
    steps = metrics[by]["steps"]
    by_vals = metrics[by]["values"]
    idx = np.argmin(by_vals)
    best_step = steps[idx]
    result = {}
    for m, vals in metrics.items():
        if m[:4] == 'test':
            found = False
            for step, val in zip(vals["steps"], vals["values"]):
                if step == best_step:
                    result[m] = val
                    found = True
                    break
            assert found
    return result


def get_model_files(run_id, save_dir):
    return get_artifacts(run_id=run_id, name_predicate=lambda x: x[-3:] == ".pt", save_dir=save_dir)


def get_artifacts(run_id, name_predicate=lambda _: True, save_dir=None):
    run_info = get_run(run_id)
    files = [x for x in run_info["artifacts"] if name_predicate(x["name"])]
    saved_fnames = []
    fname_to_content = {}
    for f_info in files:
        f_content = fs.get(f_info["file_id"]).read()
        if save_dir is None:
            fname_to_content[f_info["name"]] = f_content.decode("utf-8")
        else:
            f_name = "run_{}_".format(run_id) + f_info["name"]
            f_path = os.path.join(save_dir, f_name)
            with open(f_path, "wb") as f:
                f.write(f_content)
            saved_fnames.append(f_path)
    if save_dir is None:
        return fname_to_content
    else:
        return saved_fnames


def get_artifact(run_id, name, save_dir=None):
    return get_artifacts(run_id=run_id, name_predicate=lambda x: x==name, save_dir=save_dir)[0]
