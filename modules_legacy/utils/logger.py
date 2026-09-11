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

        self.csv_path = (
            self.save_dir
            / "training_log.csv"
        )

        self.history_path = (
            self.save_dir
            / "history.json"
        )

        self.history = []
        self.fields = None

        if mode == "overwrite":
            self.csv_path.unlink(
                missing_ok=True
            )

            self.history_path.unlink(
                missing_ok=True
            )

        elif self.history_path.exists():
            with open(
                self.history_path,
                "r",
            ) as file:
                self.history = json.load(
                    file
                )

            if self.history:
                self.fields = list(
                    self.history[0].keys()
                )
                
    def log(
        self,
        metrics,
    ):
        row = {}

        for key, value in metrics.items():
            if hasattr(value, "item"):
                value = value.item()

            row[key] = value

        if self.fields is None:
            self.fields = list(
                row.keys()
            )

        write_header = (
            not self.csv_path.exists()
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

            if write_header:
                writer.writeheader()

            writer.writerow(row)

        self.history.append(
            row.copy()
        )

        with open(
            self.history_path,
            "w",
        ) as file:
            json.dump(
                self.history,
                file,
                indent=4,
            )