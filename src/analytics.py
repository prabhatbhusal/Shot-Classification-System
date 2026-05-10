import matplotlib
matplotlib.use("Agg")          # no display needed — saves to file
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter


class Analytics:
    """
    Takes the list of detected shots and produces:
      - A printed summary table
      - A bar chart saved as PNG
    """

    def __init__(self, shots: list):
        self.shots = shots
        self.df    = pd.DataFrame(shots) if shots else pd.DataFrame()

    # ── public ───────────────────────────────────────────────────────────

    def print_summary(self):
        print("=" * 45)
        print("          SHOT ANALYTICS SUMMARY")
        print("=" * 45)

        if self.df.empty:
            print("  No shots detected.")
            print("=" * 45)
            return

        total = len(self.df)
        print(f"  Total shots detected : {total}")
        print()

        # Shots by type
        print("  By shot type:")
        for shot_type, count in self.df["shot_type"].value_counts().items():
            pct = count / total * 100
            bar = "█" * int(pct / 5)
            print(f"    {shot_type:<12} {count:>4}  {pct:5.1f}%  {bar}")

        print()

        # Shots by player
        print("  By player:")
        for pid, count in self.df["player_id"].value_counts().items():
            print(f"    Player {pid:<3}  {count:>4} shots")

        # Time info
        duration = self.df["timestamp"].max() - self.df["timestamp"].min()
        rate     = total / max(duration, 1) * 60
        print()
        print(f"  Match span   : {self.df['timestamp'].min():.1f}s "
              f"– {self.df['timestamp'].max():.1f}s")
        print(f"  Avg shot rate: {rate:.1f} shots/min")
        print("=" * 45)

    def save_chart(self, path="output/shot_classification_chart.png"):
        if self.df.empty:
            return

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("Shot Classification System", fontsize=16, fontweight="bold")

        # ── Chart 1: shot type counts ─────────────────────────────────
        type_counts = self.df["shot_type"].value_counts()
        colors      = ["#4CAF50", "#2196F3", "#FF5722", "#9C27B0", "#FF9800"]
        axes[0].bar(type_counts.index, type_counts.values,
                    color=colors[:len(type_counts)])
        axes[0].set_title("Shots by Type")
        axes[0].set_xlabel("Shot Type")
        axes[0].set_ylabel("Count")
        axes[0].tick_params(axis="x", rotation=30)

        # ── Chart 2: shots per player ─────────────────────────────────
        player_counts = self.df["player_id"].value_counts().sort_index()
        axes[1].bar(
            [f"Player {p}" for p in player_counts.index],
            player_counts.values,
            color="#03A9F4",
        )
        axes[1].set_title("Shots by Player")
        axes[1].set_xlabel("Player")
        axes[1].set_ylabel("Count")

        # ── Chart 3: shot timeline ────────────────────────────────────
        shot_colors = {
            "forehand" : "#4CAF50",
            "backhand" : "#2196F3",
            "smash"    : "#FF5722",
            "volley"   : "#9C27B0",
            "serve"    : "#FF9800",
        }
        for _, row in self.df.iterrows():
            color = shot_colors.get(row["shot_type"], "grey")
            axes[2].scatter(row["timestamp"], row["player_id"],
                            c=color, s=40, alpha=0.7)

        # Legend
        handles = [
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=c, markersize=8, label=t)
            for t, c in shot_colors.items()
        ]
        axes[2].legend(handles=handles, loc="upper right", fontsize=7)
        axes[2].set_title("Shot Timeline")
        axes[2].set_xlabel("Time (s)")
        axes[2].set_ylabel("Player ID")

        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Chart saved → {path}")
