from pathlib import Path
import csv
import json

class TrainingLogger:
    def __init__(
        self,
        save_dir,
        mode="overwrite",
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.csv_path = self.save_dir / "training_log.csv"
        self.history_path = self.save_dir / "history.json"

        self.fields = [
            "epoch",
            "stage",
            "keep_ratio",
            "mask_ratio",
            "hard_mask_ratio",
            "soft_mask_ratio",
            "mask_temperature",
            "loss",
            "attack_loss",
            "clean_attack_loss",
            "attack_gain",
            "retention_loss",
            "semantic_loss",
            "stealth_loss",
            "reconstruction_loss",
            "gradient_loss",
            "budget_loss",
            "diversity_loss",
            "support_entropy",
            "clean_accuracy",
            "adv_accuracy",
            "accuracy_drop",
            "learning_rate",
        ]

        if mode == "overwrite":
            self.history = []

            with open(
                self.csv_path,
                "w",
                newline="",
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=self.fields,
                )
                writer.writeheader()

            with open(
                self.history_path,
                "w",
            ) as file:
                json.dump(
                    self.history,
                    file,
                    indent=4,
                )

        else:
            if self.history_path.exists():
                with open(
                    self.history_path,
                    "r",
                ) as file:
                    self.history = json.load(file)
            else:
                self.history = []

            if not self.csv_path.exists():
                with open(
                    self.csv_path,
                    "w",
                    newline="",
                ) as file:
                    writer = csv.DictWriter(
                        file,
                        fieldnames=self.fields,
                    )
                    writer.writeheader()

    def log(
        self,
        metrics,
    ):
        row = {
            key: metrics[key]
            for key in self.fields
        }

        self.history.append(
            row.copy()
        )

        with open(
            self.csv_path,
            "a",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=self.fields,
            )
            writer.writerow(row)

        with open(
            self.history_path,
            "w",
        ) as file:
            json.dump(
                self.history,
                file,
                indent=4,
            )
