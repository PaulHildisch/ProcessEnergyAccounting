import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Rectangle

class HandlerMultiColor(HandlerBase):
    """Custom legend handler for grouping top processes into a single multi-color block."""
    def create_artists(self, legend, orig_handle, x0, y0, width, height, fontsize, trans):
        # Default to a gray fallback if no colors are attached
        colors = getattr(orig_handle, "_legend_colors", ["#cccccc"])
        n = len(colors)
        artists = []
        for i, color in enumerate(colors):
            rect = Rectangle(
                (x0 + i * width / n, y0),
                width / n,
                height,
                facecolor=color,
                transform=trans,
                lw=0,
            )
            artists.append(rect)
        return artists


class AttributionPlotter:
    def __init__(self, df, time_col="_time", energy_col="attributed_dynamic_Ws"):
        """
        Initializes the plotter with the dataframe output from the SHAP attributor.
        """
        self.df = df.copy()
        self.time_col = time_col
        self.energy_col = energy_col
        
        # Merge process instances like wrk_99 and wrk_246 into one base name
        # ist his correct?
        self.df["base_name"] = self.df["process_name"].str.replace(r"_\d+$", "", regex=True).str.strip()
        self.df.loc[self.df["base_name"] == "", "base_name"] = "unknown"
        
        # Create PID label
        self.df["pid_label"] = self.df["process_name"] + " (" + self.df["pid"].astype(str) + ")"

    def plot_top_processes(self, top_n=8, save_path=None):
        """Plots the stacked area chart grouped by base process name."""
        # Aggregate and Pivot
        agg = self.df.groupby([self.time_col, "base_name"])[self.energy_col].sum().reset_index()
        pivot = agg.pivot(index=self.time_col, columns="base_name", values=self.energy_col)#.fillna(0)
        # pivot = pivot.ffill(limit=2)
        # pivot = pivot.fillna(0)

        # Find Top N by peak (max)
        # top_processes = pivot.max().sort_values(ascending=False).head(top_n).index
        top_processes = pivot.sum().sort_values(ascending=False).head(top_n).index
        pivot_top = pivot[top_processes].copy()
        print("Top processes: ", top_processes)

        # Group remainder into "Other"
        if len(pivot.columns) > top_n:
            pivot_top["Other"] = pivot.drop(columns=top_processes).sum(axis=1)

        pivot_top_clipped = pivot_top.clip(lower=0)

        # Plotting
        fig, ax = plt.subplots(figsize=(7.2, 3.4))
        pivot_top_clipped.plot.area(ax=ax, alpha=0.8, linewidth=0, legend=False)

        ax.set_xlabel("Time", fontsize=12, labelpad=4)
        ax.set_ylabel("Attributed Dynamic Power (Ws)", fontsize=12, labelpad=4)
        ax.tick_params(axis="both", labelsize=12)

        # Setup custom multi-color legend
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        top_colors = colors[:top_n]
        other_color = colors[len(pivot_top_clipped.columns) % 10 - 1] if "Other" in pivot_top_clipped.columns else "#cccccc"

        top_handle = Rectangle((0, 0), 1, 1)
        top_handle._legend_colors = top_colors
        
        handles = [top_handle]
        labels = [f"Top {top_n} processes"]
        

        if "Other" in pivot_top_clipped.columns:
            other_handle = Rectangle((0, 0), 1, 1)
            other_handle._legend_colors = [other_color]
            handles.append(other_handle)
            labels.append("Other")

        ax.legend(
            handles=handles,
            labels=labels,
            handler_map={top_handle: HandlerMultiColor(), other_handle: HandlerMultiColor()} if "Other" in pivot_top_clipped.columns else {top_handle: HandlerMultiColor()},
            loc="upper right",
            bbox_to_anchor=(0.97, 0.97),
            fontsize=10.5,
            frameon=True,
            framealpha=0.9,
            ncol=1,
            handlelength=2.5,
            labelspacing=0.5,
        )

        ax.set_title("Per-Process Attributed Power Over Time", fontsize=13, pad=6)
        plt.tight_layout(pad=0.5)
        
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
        #plt.show()

    def plot_top_processes_new(self, top_n=8, save_path=None):
        """Plots the stacked area chart grouped by base process name."""
        # Aggregate and Pivot
        agg = self.df.groupby([self.time_col, "base_name"])[self.energy_col].sum().reset_index()
        pivot = agg.pivot(index=self.time_col, columns="base_name", values=self.energy_col)

        # Find Top N by peak (max)
        top_processes = pivot.sum().sort_values(ascending=False).head(top_n).index
        pivot_top = pivot[top_processes].copy()
        print("Top processes: ", top_processes)

        # Group remainder into "Other"
        if len(pivot.columns) > top_n:
            pivot_top["Other"] = pivot.drop(columns=top_processes).sum(axis=1)

        pivot_top_clipped = pivot_top.clip(lower=0)

        # Plotting
        #fig, ax = plt.subplots(figsize=(7.2, 3.4))
        fig, ax = plt.subplots(figsize=(12, 5))
        
        # NOTE: legend=True is now set here!
        pivot_top_clipped.plot.area(ax=ax, alpha=0.8, linewidth=0, legend=True)

        ax.set_xlabel("Time", fontsize=12, labelpad=4)
        ax.set_ylabel("Attributed Dynamic Power (Ws)", fontsize=12, labelpad=4)
        ax.tick_params(axis="both", labelsize=11)

        # Setup standard legend with actual process names
        # ax.legend(
        #     title="Top Processes",
        #     loc="center left",
        #     bbox_to_anchor=(1.02, 0.5), # Moves it just outside the right edge
        #     fontsize=10.5,
        #     frameon=True,
        #     framealpha=0.9,
        #     ncol=1
        # )
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=9, frameon=True)

        ax.set_title("Per-Process Attributed Power Over Time", fontsize=13, pad=6)
        plt.tight_layout(pad=0.5)
        
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)

    def plot_top_pids(self, top_n=20, save_path=None):
        """Plots the stacked area chart separated strictly by individual PIDs."""
        agg = self.df.groupby([self.time_col, "pid_label"])[self.energy_col].sum().reset_index()
        pivot = agg.pivot(index=self.time_col, columns="pid_label", values=self.energy_col).fillna(0)

        # top_pids = pivot.max().sort_values(ascending=False).head(top_n).index
        top_pids = pivot.sum().sort_values(ascending=False).head(top_n).index
        pivot_top = pivot[top_pids].copy()
        
        if len(pivot.columns) > top_n:
            pivot_top["Other"] = pivot.drop(columns=top_pids).sum(axis=1)
            
        pivot_top_clipped = pivot_top.clip(lower=0)

        fig, ax = plt.subplots(figsize=(12, 5))
        pivot_top_clipped.plot.area(ax=ax, alpha=0.8, linewidth=0)
        
        ax.set_xlabel("Time", fontsize=12, labelpad=4)
        ax.set_ylabel("Attributed Dynamic Power (Ws)", fontsize=12, labelpad=4)
        ax.set_title(f"Per-PID Attributed Power Over Time (Top {top_n} by Peak)", fontsize=13, pad=6)
        ax.tick_params(axis="both", labelsize=11)
        
        # Standard legend for PID view (too many for a single grouped color block)
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=9, frameon=True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
        #plt.show()