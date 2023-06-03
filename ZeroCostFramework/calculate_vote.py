import json
from itertools import combinations
import os
from tqdm import tqdm
from utils.util_functions import get_proxies


path = "experiment"

epochs = ["zero_cost_scores"]

# for i in range(10):
#     epochs.append(f"zero_cost_scores_{i}")

def init(names):
    with open(f"{path}/generated_architectures.json") as file:
        results = json.load(file)
        
    metrics = {}
    acc = []
    
    for key, value in results.items():
        if not all(format in value for format in epochs):
            continue
        if "val_acc" not in value:
            continue
        for e_n in names:
            # if value[e_n.split(".")[0]][e_n.split(".")[1]]["score"] < 0.2:
            #     continue
            if e_n not in metrics:
                metrics[e_n] = []
            score = value[e_n.split(".")[0]][e_n.split(".")[1]]["score"]
            if score == "Nan":
                metrics[e_n].append(float("nan"))
            else:
                metrics[e_n].append(score)
        acc.append(value["val_acc"])
    return (acc, metrics)

def vote(mets, gt):
    numpos = 0
    for m in mets:
        numpos += 1 if m > 0 else 0
    if numpos >= len(mets)/2:
        sign = +1
    else:
        sign = -1
    return sign*gt


def calc(acc, metrics, comb):
    num_pts = len(acc)
    tot=0
    right=0
    for i in range(num_pts):
        for j in range(num_pts):
            if i!=j:
                diff = acc[i] - acc[j]
                if diff == 0:
                    continue
                diffsyn = []
                for m in comb:
                    diffsyn.append(metrics[m][i] - metrics[m][j])
                same_sign = vote(diffsyn, diff)
                right += 1 if same_sign > 0 else 0
                tot += 1
    votes = right/tot
    return (comb, votes)
    
def get_all_combinations(names):
    list_combinations = []
    for n in range(5):
        list_combinations += list(combinations(names, n))
    return list_combinations

if __name__ == "__main__":
    print("STARTING...")
    proxies = get_proxies()
    print("INITIALIZING...")
    names = []
    for name in proxies:
        for epoch in epochs:
            names.append(f"{epoch}.{name}")
    acc, metrics = init(names)
    comb = get_all_combinations(names)
    
    D = {}
    print("CALCULATING...")
    for c in tqdm(comb):
        a, votes = calc(acc, metrics, c)
        D[str(a)] = votes
    print("SORTING...")
    D = dict(sorted(D.items(), key=lambda item: item[1], reverse=True))
    print("WRITING TO FILE...")
    with open(f"{path}/vote_combinations.json", "w") as file:
        json.dump(D, file)
