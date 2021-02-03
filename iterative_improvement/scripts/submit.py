#!/usr/bin/env python3

import os
import re
from argparse import ArgumentParser
from git import Repo

parser = ArgumentParser(description="Submit on cscs")
parser.add_argument("--time", default="04:00:00")
parser.add_argument("--jobname", default="train")
args = parser.parse_args()

repo = Repo("../../")
if repo.is_dirty():
    raise Exception("Repo is dirty. Commit changes")

repo_status = repo.git.status(porcelain="v2", branch=True)
ahead_behind_match = re.search(r"#\sbranch\.ab\s\+(\d+)\s-(\d+)", repo_status)
if ahead_behind_match:
    ahead = int(ahead_behind_match.group(1))
    behind = int(ahead_behind_match.group(2))
    if not ahead == 0 and behind == 0:
        raise Exception("Push changes first")

time_exp = re.compile("^\\d\\d:\\d\\d:\\d\\d$")
timelimit = args.time
if not re.match(time_exp, timelimit):
    raise Exception("Invalid time")


with open("submit_template.txt", "r") as f_template:
    submit_str = f_template.read()

submit_str = submit_str.format(args.jobname, timelimit)

with open("submit.sh", "w") as submit_file:
    submit_file.write(submit_str)

os.system("scp submit.sh daint:/users/aschfer/")
os.system(
    "ssh daint 'cd Repositories/k-cut-ML; git pull; cd /users/aschfer; sbatch /users/aschfer/submit.sh; rm /users/aschfer/submit.sh'"
)
os.system("rm submit.sh")
