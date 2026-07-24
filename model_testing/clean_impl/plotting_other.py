import matplotlib.pyplot as plt


class Plotter:

    def __init__(self,y_pred,y_test,t_test, alg_name="rf", window_start = None, window_end= None):
        self.t_test = t_test
        self.y_test = y_test
        self.y_pred = y_pred
        self.alg_name = alg_name
        self.window_start = window_start if window_start is not None else 0
        #check this
        self.window_end = window_end if window_end is not None else len(t_test)
        
    
    def _init_sub_plots(self, label_actual="Actual Energy", label_model="Predicted Energy"):
        self.fig, self.ax = plt.subplots(figsize=(7.2, 3.4))
        self.ax.plot(
            self.t_test[self.window_start : self.window_end],
            self.y_test[self.window_start : self.window_end],
            label = label_actual,
            linewidth=2.0
        )

        if self.alg_name.lower() =="rf":
            label_model = "Predicted (Random Forest)"
        elif self.alg_name.lower() == "ebm":
            label_model = "Predicted (EBM)"
        elif self.alg_name.lower() == "mlp":
            label_model = "Predicted (MLP)"
        elif self.alg_name.lower() == "mlp_optuna":
            label_model = "Predicted (MLP optuna)"
        elif self.alg_name.lower() == "ffnn":
            label_model = "Predicted (FFNN)"
        elif self.alg_name.lower() == "cnn_1d":
            label_model = "Predicted (CNN-1D)"
        elif self.alg_name.lower() == "lstm":
            label_model = "Predicted (LSTM)"
        else:
            label_model = "Predicted Energy"
        
        self.ax.plot(
            self.t_test[self.window_start : self.window_end],
            self.y_pred[self.window_start : self.window_end],
            label=label_model,
            linestyle="--",
            linewidth=2.0,
        )

    def _set_labels_title_legend(self, x_label="Time", y_label="Interval Energy (Ws)", title="Actual vs. Predicted Interval energy"):
        self.ax.set_xlabel(x_label, fontsize=12, labelpad=4)
        self.ax.set_ylabel(y_label, fontsize=12, labelpad=4)
        self.ax.tick_params(axis="both", labelsize=12)
        self.ax.legend(
            loc="upper right",
            bbox_to_anchor=(0.97, 0.97),
            fontsize=10.5,
            frameon=True,
            framealpha=0.9,
            handlelength=1.8,
            labelspacing=0.4,
        )
        self.ax.set_title(title, fontsize=13, pad=6)

    def plot_and_save(self, path="./", name="actual_vs_predicted_interval_energy"):
        self._init_sub_plots()
        self._set_labels_title_legend()
        plt.tight_layout(pad=0.5)
        plt.savefig(path + name +".pdf", bbox_inches="tight")
        plt.savefig(path + name + ".png", bbox_inches="tight", dpi=300)
        #plt.show()
        plt.close(self.fig)

    def plot_only(self, path="./"):
        self._init_sub_plots()
        self._set_labels_title_legend()
        plt.tight_layout(pad=0.5)
        plt.show()

def plot_dataset(t, y,name, window_start = None, window_end= None):
    window_start = window_start if window_start is not None else 0
    window_end = window_end if window_end is not None else len(t)
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.plot(
            t[window_start : window_end],
            y[window_start : window_end],
            label = "Actual Energy",
            linewidth=2.0
        )
    x_label="Time"
    y_label="Interval Energy (Ws)"
    title="Actual Interval energy"
    ax.set_xlabel(x_label, fontsize=12, labelpad=4)
    ax.set_ylabel(y_label, fontsize=12, labelpad=4)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(
            loc="upper right",
            bbox_to_anchor=(0.97, 0.97),
            fontsize=10.5,
            frameon=True,
            framealpha=0.9,
            handlelength=1.8,
            labelspacing=0.4,
        )
    ax.set_title(title, fontsize=13, pad=6)
    plt.tight_layout(pad=0.5)
    plt.savefig( name + ".png", bbox_inches="tight", dpi=300)
    #plt.show()
    plt.close(fig)