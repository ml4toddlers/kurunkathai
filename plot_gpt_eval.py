import json
import argparse
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Plot rating metrics for two models from a JSON file.")
    parser.add_argument("json_file", help="Path to the JSON file containing rating data")
    args = parser.parse_args()

    with open(args.json_file, "r") as f:
        entries = json.load(f)

    datasets = ["test", "validation"]
    models = ["gpt-3.5-turbo", "gpt-4o-2024-08-06"]
    metrics = ["Creativity", "Consistency", "Grammar", "Plot"]

    iters = [entry["Iter"] for entry in entries]

    data_dict = {
        dataset: {
            model: {metric: [] for metric in metrics}
            for model in models
        }
        for dataset in datasets
    }

    for entry in entries:
        for dataset in datasets:
            for model in models:
                for metric in metrics:
                    data_dict[dataset][model][metric].append(
                        entry["ratings"][dataset][model][metric]
                    )

    linestyles = {"gpt-3.5-turbo": "-", "gpt-4o-2024-08-06": "--"}
    markers = {"Creativity": "o", "Consistency": "s", "Grammar": "^", "Plot": "d"}

    for dataset in datasets:
        fig, ax = plt.subplots(figsize=(8, 6))
        for model in models:
            for metric in metrics:
                ax.plot(
                    iters,
                    data_dict[dataset][model][metric],
                    label=f"{model} - {metric}",
                    linestyle=linestyles[model],
                    marker=markers[metric]
                )
        ax.set_title(dataset.capitalize() + " Set")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Rating")
        ax.legend(fontsize="small", loc="best")
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(args.json_file.replace(".json","") +  f"_{dataset}_ratings.png")
        plt.close(fig)

if __name__ == "__main__":
    main()
