from pathlib import Path
import csv
import json

class TrainingLogger:
    def __init__(
        self,
        save_dir,
        mode="overwrite",
    ):
        """
        Args:
            save_dir:
                Directory for training_log.csv
                and history.json.

            mode:
                "overwrite":
                    Start a completely new log.
                    Existing log files are replaced.

                "append":
                    Keep existing history and
                    continue logging.
        """

        if mode not in {
            "overwrite",
            "append",
        }:
            raise ValueError(
                "mode must be "
                "'overwrite' or 'append'."
            )

        self.mode = mode

        self.save_dir = Path(
            save_dir
        )

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

        self.fields = [
            "epoch",
            "loss",
            "attack_loss",
            "semantic_loss",
            "tv_loss",
            "clean_accuracy",
            "adv_accuracy",
            "accuracy_drop",
            "learning_rate",
        ]

        # ==================================================
        # NEW TRAINING RUN
        # ==================================================

        if self.mode == "overwrite":

            self.history = []

            # Replace CSV completely
            with open(
                self.csv_path,
                "w",
                newline="",
            ) as f:

                writer = csv.DictWriter(
                    f,
                    fieldnames=self.fields,
                )

                writer.writeheader()

            # Replace JSON completely
            with open(
                self.history_path,
                "w",
            ) as f:

                json.dump(
                    self.history,
                    f,
                    indent=4,
                )

        # ==================================================
        # RESUME TRAINING
        # ==================================================

        else:

            # Load previous history if available
            if self.history_path.exists():

                try:

                    with open(
                        self.history_path,
                        "r",
                    ) as f:

                        self.history = (
                            json.load(f)
                        )

                except (
                    json.JSONDecodeError,
                    OSError,
                ):

                    self.history = []

            else:

                self.history = []

            # Create CSV header only if missing
            if not self.csv_path.exists():
                with open(
                    self.csv_path,
                    "w",
                    newline="",
                ) as f:

                    writer = (
                        csv.DictWriter(
                            f,
                            fieldnames=self.fields,
                        )
                    )

                    writer.writeheader()

    def log(
        self,
        metrics,
    ):
        # Only keep expected fields
        row = {
            key: metrics[key]
            for key in self.fields
        }

        # ==========================================
        # In-memory history
        # ==========================================

        self.history.append(
            row.copy()
        )

        # ==========================================
        # Append epoch to CSV
        # ==========================================

        with open(
            self.csv_path,
            "a",
            newline="",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=self.fields,
            )

            writer.writerow(
                row
            )

        # ==========================================
        # Replace JSON with updated full history
        # ==========================================

        with open(
            self.history_path,
            "w",
        ) as f:
            json.dump(
                self.history,
                f,
                indent=4,
            )