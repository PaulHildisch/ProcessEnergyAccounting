import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Rectangle

#These classes are used to create the attribution plots for the per process/pid attribtuion
#Heavily based on the existing code from the original paper


class HandlerMultiColor(HandlerBase):
  
    def create_artists(self, legend, orig_handle, x0, y0, width, height, fontsize, trans):
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
        self.df = df.copy()
        self.time_col = time_col
        self.energy_col = energy_col
        
        #Group process instaces by base name
        self.df["base_name"] = self.df["process_name"].str.replace(r"_\d+$", "", regex=True).str.strip()
        self.df.loc[self.df["base_name"] == "", "base_name"] = "unknown"
        
        #Create a column that combines the pid with the process name for better identification
        self.df["pid_label"] = self.df["process_name"] + " (" + self.df["pid"].astype(str) + ")"


    #Plotting the top processes the same way as the original paper as a stacked chart grouped by the basename
    def plot_top_processes(self, top_n=8, save_path=None, display = False):
        agg = self.df.groupby([self.time_col, "base_name"])[self.energy_col].sum().reset_index()
        pivot = agg.pivot(index=self.time_col, columns="base_name", values=self.energy_col)#.fillna(0)

        #Choose processes either by max or by sum
        #top_processes = pivot.max().sort_values(ascending=False).head(top_n).index
        top_processes = pivot.sum().sort_values(ascending=False).head(top_n).index
        pivot_top = pivot[top_processes].copy()
        print("Top processes: ", top_processes)

        # Group remainder into "Other"
        if len(pivot.columns) > top_n:
            pivot_top["Other"] = pivot.drop(columns=top_processes).sum(axis=1)
        pivot_top_clipped = pivot_top.clip(lower=0)

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
        if display:
            plt.show()

    #Very similar to the previous function but also show the names of the aggregated top processes
    def plot_top_processes_new(self, top_n=8, save_path=None):
        agg = self.df.groupby([self.time_col, "base_name"])[self.energy_col].sum().reset_index()
        pivot = agg.pivot(index=self.time_col, columns="base_name", values=self.energy_col)

        #Choose top processes either by max value (highest peak) or by max sum total
        top_processes = pivot.sum().sort_values(ascending=False).head(top_n).index
        #top_processes = pivot.max().sort_values(ascending=False).head(top_n).index
        pivot_top = pivot[top_processes].copy()
        print("Top processes: ", top_processes)

        # Group remainder into "Other"
        if len(pivot.columns) > top_n:
            pivot_top["Other"] = pivot.drop(columns=top_processes).sum(axis=1)

        pivot_top_clipped = pivot_top.clip(lower=0)

        #fig, ax = plt.subplots(figsize=(7.2, 3.4))
        fig, ax = plt.subplots(figsize=(12, 5))
        pivot_top_clipped.plot.area(ax=ax, alpha=0.8, linewidth=0, legend=True)
        ax.set_xlabel("Time", fontsize=12, labelpad=4)
        ax.set_ylabel("Attributed Dynamic Power (Ws)", fontsize=12, labelpad=4)
        ax.tick_params(axis="both", labelsize=11)
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=10, frameon=True)
        ax.set_title("Per-Process Attributed Power Over Time", fontsize=13, pad=6)
        plt.tight_layout(pad=0.5)        
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)



    #Plot top pids instead of basenames -> more granular view
    def plot_top_pids(self, top_n=20, save_path=None, display = False):
        agg = self.df.groupby([self.time_col, "pid_label"])[self.energy_col].sum().reset_index()
        pivot = agg.pivot(index=self.time_col, columns="pid_label", values=self.energy_col).fillna(0)

        #Choose either max oder sum -> highest peak vs total highest energy
        #top_pids = pivot.max().sort_values(ascending=False).head(top_n).index
        top_pids = pivot.sum().sort_values(ascending=False).head(top_n).index
        pivot_top = pivot[top_pids].copy()
        
        if len(pivot.columns) > top_n:
            pivot_top["Other"] = pivot.drop(columns=top_pids).sum(axis=1)
            
        pivot_top_clipped = pivot_top.clip(lower=0)

        fig, ax = plt.subplots(figsize=(12, 5))
        pivot_top_clipped.plot.area(ax=ax, alpha=0.8, linewidth=0)
        
        ax.set_xlabel("Time", fontsize=12, labelpad=4)
        ax.set_ylabel("Attributed Dynamic Power (Ws)", fontsize=12, labelpad=4)
        ax.set_title(f"Per-PID Attributed Power Over Time (Top {top_n})", fontsize=13, pad=6)
        ax.tick_params(axis="both", labelsize=11)
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=10, frameon=True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)

        if display:
            plt.show()