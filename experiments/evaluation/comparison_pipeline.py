import os
import json
import csv
import networkx as nx

# ============================================================
# Configuration
# ============================================================

GAMMAS = [0.75]

# ============================================================
# TD computation
# ============================================================

def compute_td(ranking):
    return sum(
        r["severity"]
        for r in ranking
        if r["severity"] > 0
    )

# ============================================================
# Baseline traversal
# ============================================================

def baseline_remediation(ranking, target):
    cumulative = 0.0
    selected = []

    for r in ranking:
        sev = r["severity"]

        if sev <= 0:
            continue

        selected.append(r["type"])
        cumulative += sev

        if cumulative >= target:
            break

    m_i = len(selected)
    return selected, m_i, cumulative

# ============================================================
# Cluster severity
# ============================================================

def compute_cluster_severity(cluster_results_snapshot, severity_map):

    cluster_info = []

    for c in cluster_results_snapshot:

        types = c["cluster"]   # <-- correct key

        sev = sum(
            severity_map.get(t, 0.0)
            for t in types
        )

        cluster_info.append({
            "types": types,
            "severity": sev
        })

    cluster_info.sort(key=lambda x: -x["severity"])
    print(
    "[VERIFY CLUSTERS]",
    [(len(c["types"]), round(c["severity"],2)) for c in cluster_info[:3]]
)

    return cluster_info

# ============================================================
# Proposed traversal
# ============================================================

def proposed_remediation(cluster_info, target):
    cumulative = 0.0
    selected_types = []
    k_i = 0

    for cluster in cluster_info:
        k_i += 1
        selected_types.extend(cluster["types"])
        cumulative += cluster["severity"]

        if cumulative >= target:
            break

    selected_types = list(dict.fromkeys(selected_types))
    return selected_types, k_i, cumulative

# ============================================================
# Cohesion metrics
# ============================================================

def compute_cohesion(G, selected_types):

    sub = G.subgraph(set(selected_types)).copy()

    wcc = nx.number_weakly_connected_components(sub)

    comps = list(nx.weakly_connected_components(sub))
    lwcc = max(len(c) for c in comps) if comps else 0

    density = nx.density(sub) if sub.number_of_nodes() > 1 else 0.0

    return wcc, lwcc, density

# ============================================================
# Dependency graph loader
# ============================================================

def load_dependency_graph(gephi_dir, snapshot_index):

    graph_dir = os.path.join(gephi_dir, "debt_subgraphs")

    files = sorted([
        f for f in os.listdir(graph_dir)
        if f.endswith(".gexf")
    ])

    if snapshot_index >= len(files):
        print(f"[WARNING] Snapshot index {snapshot_index} exceeds graph files.")
        return None

    path = os.path.join(graph_dir, files[snapshot_index])
    G = nx.read_gexf(path)

    print(f"[GRAPH] Loaded {files[snapshot_index]}  nodes={G.number_of_nodes()} edges={G.number_of_edges()}")

    return G

# ============================================================
# Main comparison routine
# ============================================================

def run_comparison(type_results_path,
                   cluster_results_path,
                   gephi_dir,
                   output_csv):

    with open(type_results_path) as f:
        type_results = json.load(f)

    with open(cluster_results_path) as f:
        cluster_results = json.load(f)

    rows = []

    for i, snapshot in enumerate(sorted(type_results.keys())):

        ranking = sorted(
            type_results[snapshot],
            key=lambda x: -x["severity"]
        )

        severity_map = {
            r["type"]: r["severity"]
            for r in ranking
        }

        TD = compute_td(ranking)

        if TD == 0:
            continue

        G = load_dependency_graph(gephi_dir, i)

        if G is None:
            continue

        if snapshot not in cluster_results:
            continue

        cluster_info = compute_cluster_severity(
            cluster_results[snapshot],
            severity_map
        )
                
        for gamma in GAMMAS:

            target = gamma * TD

            if len(cluster_info) > 0:

                first_cluster_sev = cluster_info[0]["severity"]
                first_cluster_size = len(cluster_info[0]["types"])

                print(
                    f"[VERIFY] {snapshot} | "
                    f"TD={TD:.2f} | "
                    f"gamma={gamma} | "
                    f"target={target:.2f} | "
                    f"first_cluster_severity={first_cluster_sev:.2f} | "
                    f"cluster_size={first_cluster_size}"
                )

                if first_cluster_sev >= target:
                    print("[VERIFY RESULT] First cluster alone satisfies target")

            base_types, m_i, base_cov = baseline_remediation(
                ranking,
                target
            )

            prop_types, k_i, prop_cov = proposed_remediation(
                cluster_info,
                target
            )

            WCC, LWCC, Den = compute_cohesion(G, prop_types)

            baseline_coverage = base_cov / TD
            proposed_coverage = prop_cov / TD

            baseline_type_count = len(base_types)
            proposed_type_count = len(prop_types)

            rows.append({
                "snapshot": snapshot,
                "gamma": gamma,
                "TD": TD,
                "target": target,
                "m_i_types": m_i,
                "k_i_clusters": k_i,
                "baseline_types": baseline_type_count,
                "proposed_types": proposed_type_count,
                "coverage_baseline": baseline_coverage,
                "coverage_proposed": proposed_coverage,
                "prop_WCC": WCC,
                "prop_LWCC": LWCC,
                "prop_density": Den
            })

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "snapshot",
                "gamma",
                "TD",
                "target",
                "m_i_types",
                "k_i_clusters",
                "baseline_types",
                "proposed_types",
                "coverage_baseline",
                "coverage_proposed",
                "prop_WCC",
                "prop_LWCC",
                "prop_density"
             ]
        )
        writer.writeheader()
        writer.writerows(rows)

# ============================================================
# Entry point
# ============================================================

def main():

    output_dir = "comparison results"
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------
    # MenuCal
    # -------------------------
    run_comparison(
        type_results_path="baseline type analysis/menucal/type_results.json",
        cluster_results_path="proposed cluster analysis/menucal/cluster_results.json",
        gephi_dir="proposed cluster analysis/menucal/gephi",
        output_csv=os.path.join(output_dir, "comparison_menucal.csv")
    )

    # -------------------------
    # NutriCompass
    # -------------------------
    run_comparison(
        type_results_path="baseline type analysis/nutricompass/type_results.json",
        cluster_results_path="proposed cluster analysis/nutricompass/cluster_results.json",
        gephi_dir="proposed cluster analysis/nutricompass/gephi",
        output_csv=os.path.join(output_dir, "comparison_nutricompass.csv")
    )


if __name__ == "__main__":
    main()