from typing import Protocol

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# import umap
from matplotlib_venn import venn2, venn2_circles, venn3, venn3_circles
from plotnine import *
from sklearn import mixture
from sklearn.cluster import KMeans

plt.style.use("seaborn-whitegrid")


class PlotTheme(Protocol):
    def __call__(self, y_axis_text_size, strip_text_size, fig_size, legend_position):
        ...


def plot_theme_classic(
    y_axis_text_size, strip_text_size, fig_size, legend_position="right"
):
    plot_theme = theme_classic()
    plot_theme += theme(
        # axis_text_x=element_blank(),
        axis_text_y=element_text(size=y_axis_text_size),
        axis_ticks=element_blank(),
        strip_text=element_text(size=strip_text_size, weight="bold"),
        strip_background=element_blank(),
        subplots_adjust={"wspace": 0.35, "hspace": 0.35},
        figure_size=fig_size,
        legend_position=legend_position,
    )
    return plot_theme


def plot_theme_classic2(x_axis_text_size, y_axis_text_size, strip_text_size, fig_size):
    plot_theme = theme_classic()
    plot_theme += theme(
        axis_text_x=element_text(size=x_axis_text_size),  # element_blank(),
        axis_text_y=element_text(size=y_axis_text_size),
        axis_ticks=element_blank(),
        strip_text=element_text(size=strip_text_size, weight="bold"),
        strip_background=element_blank(),
        subplots_adjust={"wspace": 0.35, "hspace": 0.35},
        figure_size=fig_size,
        legend_position="none",
    )
    return plot_theme


def plot_theme_minimal(
    x_axis_text_size,
    y_axis_text_size,
    strip_text_size,
    legend_position="right",
    fig_size=(6, 5),
):
    plot_theme = theme_minimal()
    plot_theme += theme(
        axis_text=element_text(size=14, color="black"),
        axis_text_x=element_text(size=x_axis_text_size),
        axis_text_y=element_text(size=y_axis_text_size),
        axis_ticks=element_blank(),
        strip_text=element_text(size=strip_text_size, weight="bold"),
        strip_background=element_blank(),
        subplots_adjust={"wspace": 0.35, "hspace": 0.35},
        figure_size=fig_size,
        legend_position=legend_position,
    )
    return plot_theme


class VolcanoPlot(object):
    def __init__(self, df, p_thres, plot_theme: PlotTheme = plot_theme_minimal):
        self.df = df
        self.plot_theme = plot_theme
        self.transform(p_thres)

    def transform(self, p_thres):
        self.df = self.df.rename(columns={"p.value": "P"})
        if isinstance(p_thres, dict):
            self.df = self.df.assign(
                is_sig=lambda row: row[list(p_thres.keys())[0]]
                < list(p_thres.values())[0]
            )
        else:
            self.df = self.df.assign(
                is_sig=lambda row: row.P < p_thres,
            )  # noqa: E501

        self.df = self.df.assign(
            log_p=lambda row: -np.log10(row.P),
            has_id=lambda row: ~row.ID.isna() * row.is_sig,
        )
        self.color = "is_sig"
        self.x_lab = ""  # "Log2 fold change"
        self.y_lab = ""  # "-log(P-value)"

    def draw(
        self,
        x="FC_G2_G1",
        y="log_p",
        point_fill=("#999999", "darkred"),
        point_alpha=0.9,
        point_size=5,
        x_axis_text_size=8,
        y_axis_text_size=8,
        strip_text_size=8,
        fig_size=(6, 5),
    ):
        p = (
            ggplot(self.df, aes(x, y, fill=self.color, size=self.color))
            + geom_point(
                aes(color="has_id"), alpha=point_alpha, size=point_size
            )  # aes(color="has_id"), color="white"
            + scale_color_manual(["white", "black"])
            + scale_fill_manual(point_fill)
            + xlab(self.x_lab)
            + ylab(self.y_lab)
            + self.plot_theme(
                x_axis_text_size, y_axis_text_size, strip_text_size, fig_size
            )
        )
        return p  ## noqa WPS331


class DotPlot(object):
    def __init__(
        self, dot_data: pd.DataFrame, plot_theme: PlotTheme = plot_theme_classic2
    ):
        self.dot_data = dot_data
        self.plot_theme = plot_theme

    def load_config(self, configs=None):
        if configs is None:
            configs = {}
        self.configs = configs

    def group_medians(self, group_col):
        return (
            self.dot_data.groupby(["metabolite", group_col])["level"]
            .median()
            .reset_index()
        )

    def labeller_funcb(self, col):
        return "\n" * 2 + f"{col.split(';')[0]}" + "\n" * 1

    def labeller_func(self, col):
        # return ";\n".join(col.split(";"))
        return "\n" * 2 + f"{col.split(';')[0]}" + "\n" * 2

    def draw(
        self,
        ncol=3,
        x="group",
        y="level",
        facet_by="metabolite",
        x_axis_text_size=8,
        y_axis_text_size=8,
        strip_text_size=8,
        fig_size=(10, 5),
        group_names=["Control", "Ulcerative colitis"],
        custom_colors={"control": "#DEB4B2", "uc": "#313342"},
    ):
        plot = (
            ggplot(self.dot_data, aes(x=x, y=y, fill=x))
            + stat_summary(
                ymin=self.group_medians(x)["level"],
                ymax=self.group_medians(x)["level"],
                geom="errorbar",
                width=0.4,
            )
            + geom_jitter(
                aes(fill=x), height=0, width=0.1, alpha=0.8, size=3, color="black"
            )
            + scale_fill_manual(values=custom_colors)
            + facet_wrap(
                f"~{facet_by}",
                ncol=ncol,
                scales="free",
                labeller=labeller(cols=self.labeller_func),
            )
            + scale_x_discrete(labels=group_names)
            + scale_y_continuous(limits=(0, None))
            + labs(x="", y="")
            + self.plot_theme(
                x_axis_text_size=x_axis_text_size,
                y_axis_text_size=y_axis_text_size,
                strip_text_size=strip_text_size,
                fig_size=fig_size,
            )
        )
        return plot


class PairedLinePlot(object):
    def __init__(
        self, dot_data: pd.DataFrame, plot_theme: PlotTheme = plot_theme_minimal
    ):
        self.dot_data = dot_data
        self.plot_theme = plot_theme

    def load_config(self, configs=None):
        if configs is None:
            configs = {}
        self.configs = configs

    def draw(
        self,
        ncol=3,
        x="Visit",
        y="level",
        facet_by="metabolite",
        group="animal",
        y_axis_text_size=8,
        strip_text_size=8,
        fig_size=(12, 5),
    ):
        plot = (
            ggplot(self.dot_data, aes(x=x, y=y))
            + geom_point(aes(fill='Progressive'), alpha=0.8, size=3)  # "#540D15"
            + geom_line(aes(group=group))
            + facet_wrap(f"~{facet_by}", ncol=ncol, scales="free")
            + scale_fill_manual(values={"0": "#DEB4B2", "1": "#313342"})
            + theme_minimal()
            + theme(
                axis_text_x=element_blank(),
                axis_text_y=element_text(size=y_axis_text_size),
                axis_ticks=element_blank(),
                strip_text=element_text(size=strip_text_size, weight="bold"),
                strip_background=element_blank(),
                subplots_adjust={"wspace": 0.35},
                figure_size=fig_size,
            )
        )
        return plot
