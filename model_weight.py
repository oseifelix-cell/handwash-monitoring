from pathlib import Path
from src.models.lstm_model import HandwashLSTM

# Project root
project_root = Path(__file__).resolve().parent
output_file = project_root / "model_weight_report.txt"

# Ensemble configs
configs = [
    {"hidden": 128, "dropout": 0.3},
    {"hidden": 112, "dropout": 0.4},
    {"hidden": 144, "dropout": 0.35},
    {"hidden": 128, "dropout": 0.45},
    {"hidden": 96,  "dropout": 0.5},
]

total_params = 0
total_bytes = 0

lines = []
lines.append("=" * 60)
lines.append("HANDWASH LSTM MODEL WEIGHT REPORT")
lines.append("=" * 60 + "\n")

for i, c in enumerate(configs, 1):
    model = HandwashLSTM(
        input_size=63,
        hidden_size=c["hidden"],
        num_classes=9,
        num_layers=2,
        dropout=c["dropout"]
    )

    params = sum(p.numel() for p in model.parameters())
    bytes_ = sum(p.numel() * p.element_size() for p in model.parameters())

    total_params += params
    total_bytes += bytes_

    lines.append(f"Model {i}:")
    lines.append(f"  Hidden size   : {c['hidden']}")
    lines.append(f"  Parameters    : {params:,}")
    lines.append(f"  Model size    : {bytes_ / (1024**2):.2f} MB\n")

lines.append("-" * 60)
lines.append(f"Ensemble total parameters : {total_params:,}")
lines.append(f"Ensemble total size       : {total_bytes / (1024**2):.2f} MB")
lines.append("=" * 60)

output_file.write_text("\n".join(lines), encoding="utf-8")

print(f"Report saved to: {output_file}")
