# imports 
import sys
import os
import re
import math
import csv
from datetime import datetime
import pyarrow as pa
import pyarrow.parquet as pq
import duckdb
import pandas as pd
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mplcursors
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtWidgets import *
from PySide6.QtCore import Qt, QDateTime, QTimer, QThread, Signal, QObject

# to supress warnings
os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.wayland*=false")

# max only 4 graphs to display
max_graphs = 4

# parameter headings
cat_keywords = [
    "Core Input Values",    
    "Core Settings Values",
    "Core Output before evaluating",
    "Core Output after evaluating",
]

# colors used to plot
colors = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
] 

# pyarrow schema for the output parquet file written during streaming
OUTPUT_SCHEMA = pa.schema([
    ("time", pa.string()),
    ("category", pa.string()),
    ("parameter_name", pa.string()),
    ("value", pa.string()),
])

# to extract timestamp from message column
def extract_msg_time(msg: str) -> str | None:
    m = re.match(r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)",msg)
    if not m:
        return None
    return m.group(1)

# to assign colors for each plot
def color_for_index(i: int) -> str:
    return colors[i % len(colors)]

# background thread for heavy processing so UI doesn't freeze
class LoadWorker(QThread):
    # to send progress info
    progress = Signal(int, str)
    # to send error info
    error = Signal(str)
    # finished now emits: parameter_categories, number_of_files, output_parquet_path
    finished = Signal(object, int, str)

    # constructor
    def __init__(self, stream_path: str, start_dt: datetime, end_dt: datetime, output_parquet_path: str, parent: QObject | None = None):
        super().__init__(parent)
        self.stream_path = stream_path
        self.start_dt = start_dt
        self.end_dt = end_dt
        self.output_parquet_path = output_parquet_path

    # automatically executed when worker.start() is called
    def run(self):
        try:
            self.execute()
        except Exception as exc:
            self.error.emit(f"Unexpected error: {exc}")

    # to load parquet files 
    def execute(self):
        self.progress.emit(4, "Scanning for parquet files…")

        # list to store required files acc to start and stop
        all_files: list[str] = []
        for root, _dirs, files in os.walk(self.stream_path):
            for f in files:
                if not f.endswith(".parquet"):
                    continue
                try:
                    ts = f.replace(".parquet", "")
                    file_dt = datetime.strptime(ts, "%Y%m%d_%H%M%S_%f")
                    if self.start_dt <= file_dt <= self.end_dt:
                        all_files.append(os.path.join(root, f))
                except Exception:
                    pass
        all_files.sort()

        # if no parquet file found
        if not all_files:
            self.error.emit("No parquet files found in the selected directory.")
            return

        self.progress.emit(8, f"Found {len(all_files)} file(s). Reading first file for parameters…")

        try:
            con = duckdb.connect()
            
            # loading parameters to tree from first parquet file
            first_file_df = con.execute("""
                SELECT message
                FROM read_parquet(?)
            """, [all_files[0]]).fetch_df()

            parameter_categories = extract_all_parameters(first_file_df)
            del first_file_df
            import gc
            gc.collect()

            self.progress.emit(15, "Streaming parquet files to output…")

            # to delete old output parquet file if is exists
            if os.path.exists(self.output_parquet_path):
                os.remove(self.output_parquet_path)

            # write data to new parquet file
            pq_writer = pq.ParquetWriter(
                self.output_parquet_path, OUTPUT_SCHEMA, compression="snappy",)

            total_written_rows = 0

            # progress bar info
            for file_idx, parquet_file in enumerate(all_files):
                pct = 15 + int((file_idx + 1) / len(all_files) * 75)  
                self.progress.emit(
                    pct,
                    f"Streaming file {file_idx + 1}/{len(all_files)}: "
                    f"{os.path.basename(parquet_file)}"
                )
                
                # read parquet files one by one
                file_df = con.execute("""
                    SELECT message
                    FROM read_parquet(?)
                """, [parquet_file]).fetch_df()

                if file_df.empty:
                    continue

                # lists to store required values from parquet file
                time_list: list[str] = []
                cat_list: list[str] = []
                param_list: list[str] = []
                value_list: list[str] = []

                # read message column
                msg_series = file_df["message"].astype(str)

                for cat in cat_keywords:
                    mask = msg_series.str.contains(cat, regex=False, na=False)
                    if not mask.any():
                        continue

                    sub_msg = msg_series[mask]

                    for msg in sub_msg:
                        msg_time = extract_msg_time(msg)
                        if msg_time is None:
                            continue
                        pos = msg.find(cat)
                        if pos == -1:
                            continue
                        after = msg[pos + len(cat):]
                        sep_idx = after.find("::")
                        if sep_idx == -1:
                            continue
                        pairs_str = after[sep_idx + 2:]

                        # regex to match [parameter, value]
                        for param, value in re.findall(r'\[([^,\]]+),\s*([^\]]+)\]', pairs_str):
                            pname = param.strip()
                            vstr  = value.strip()
                            if not pname:
                                continue
                            time_list.append(msg_time)
                            cat_list.append(cat)
                            param_list.append(pname)
                            value_list.append(vstr)

                # writing to parquet file
                if time_list:
                    batch = pa.table(
                        {
                            "time": pa.array(time_list, type=pa.string()),
                            "category": pa.array(cat_list, type=pa.string()),
                            "parameter_name": pa.array(param_list, type=pa.string()),
                            "value": pa.array(value_list, type=pa.string()),
                        },
                        schema=OUTPUT_SCHEMA,
                    )
                    pq_writer.write_table(batch)
                    total_written_rows += len(time_list)
                
                # to free memory
                del file_df, time_list, cat_list, param_list, value_list, msg_series
                gc.collect()
            pq_writer.close()

        except Exception as exc:
            self.error.emit(f"Failed to stream parquet files:\n{exc}")
            return

        if total_written_rows == 0:
            self.error.emit("No matching data rows found in the selected time range.")
            return

        self.progress.emit(100, "Done…")
        
        self.finished.emit(parameter_categories, len(all_files), self.output_parquet_path)

# to extract parameters from first parquet file 
def extract_all_parameters(df: pd.DataFrame) -> dict[str, list]:
    seen: dict[str, dict[str, None]] = {cat: {} for cat in cat_keywords}
    found_cats: set[str] = set()

    for msg in df["message"].astype(str):
        for cat in cat_keywords:
            if cat in found_cats:
                continue
            pos = msg.find(cat)
            if pos == -1:
                continue
            after = msg[pos + len(cat):]
            sep_idx = after.find("::")
            if sep_idx == -1:
                continue
            pairs_str = after[sep_idx + 2:]

            # regex to match [parameter, value]
            for param, _ in re.findall(r'\[([^,\]]+),\s*([^\]]+)\]', pairs_str):
                pname = param.strip()
                if pname:
                    seen[cat][pname] = None
            found_cats.add(cat)

        if len(found_cats) == len(cat_keywords):
            break

    return {cat: list(seen[cat]) for cat in cat_keywords}

# to load a single parameter's data 
def query_parameter_data(output_parquet_path: str, cat: str, param_name: str) -> pd.DataFrame:
    try:
        con = duckdb.connect()
        df = con.execute("""
            SELECT time, value
            FROM read_parquet(?)
            WHERE category = ? AND parameter_name = ?
            ORDER BY time
        """, [output_parquet_path, cat, param_name]).fetch_df()
        con.close()

        if df.empty:
            return pd.DataFrame(columns=["time", "value"])

        df["time"] = pd.to_datetime(df["time"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"]).reset_index(drop=True)
        return df[["time", "value"]]

    except Exception as exc:
        print(f"query_parameter_data error ({cat}, {param_name}): {exc}")
        return pd.DataFrame(columns=["time", "value"])

# export to excel/csv
def build_wide_export_df(output_parquet_path: str, params: list[tuple[str, str]]) -> pd.DataFrame:
    # disambiguate parameter names that repeat across different categories
    name_counts: dict[str, int] = {}
    for _, param_name in params:
        name_counts[param_name] = name_counts.get(param_name, 0) + 1

    col_labels: list[str] = []
    series_map: dict[str, pd.Series] = {}
    all_times: set = set()

    for cat, param_name in params:
        label = param_name if name_counts[param_name] == 1 else f"{param_name} ({cat})"
        col_labels.append(label)

        df = query_parameter_data(output_parquet_path, cat, param_name)
        if df.empty:
            series_map[label] = pd.Series(dtype=object)
            continue

        s = df.set_index("time")["value"]
        series_map[label] = s
        all_times.update(s.index)

        del df  # free immediately after building the series

    if not all_times:
        return pd.DataFrame(columns=["timestamp"] + col_labels)

    sorted_times = sorted(all_times)
    # write timestamp as a full-precision string - otherwise excel's default datetime format truncates to whole seconds
    ts_strs = [str(t) for t in sorted_times]

    out = pd.DataFrame({"timestamp": ts_strs})
    for label in col_labels:
        s = series_map[label]
        out[label] = [s.get(t, "-") for t in sorted_times]

    return out

# to create graph widget — title bar, remove button, toolbar, graph
class GraphWidget(QWidget):
    max_points_per_series = 5000

    # constructor
    def __init__(self, series: list, on_remove_callback, graph_index: int, output_parquet_path: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.series = series  # list of (cat, param_name)  
        self.on_remove_callback = on_remove_callback
        self.graph_index = graph_index
        self.output_parquet_path = output_parquet_path

        self.setMinimumHeight(220)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background-color: white;")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(0)
        
        # title bar
        title_bar = QWidget()
        title_bar.setStyleSheet("background-color: #f0f0f0; border-radius: 4px 4px 0 0;")
        title_bar.setFixedHeight(34)
        tb_layout = QHBoxLayout(title_bar)
        tb_layout.setContentsMargins(10, 0, 6, 0)

        # display parameter names plotted at top of graph
        param_names = ", ".join(p for _, p in series[:3])
        if len(series) > 3:
            param_names += f" +{len(series) - 3} more"

        # graph label
        title_lbl = QLabel(
            f"<b>Graph {graph_index + 1}</b>  "
            f"<span style='color:#666;font-size:11px;'>{param_names}</span>"
        )
        title_lbl.setStyleSheet("color: #222; background: transparent;")
        tb_layout.addWidget(title_lbl, 1)

        # export csv button — exports time, parameter name, value for all series in the graph
        export_btn = QPushButton("⬇ CSV")
        export_btn.setFixedSize(52, 22)
        export_btn.setToolTip("Export plotted data to CSV")
        export_btn.setStyleSheet(
            "QPushButton { background:#27ae60; color:white; border:none;"
            "  border-radius:3px; font-size:11px; font-weight:bold; }"
            "QPushButton:hover { background:#2ecc71; }"
        )
        export_btn.clicked.connect(self.on_export_csv)
        tb_layout.addWidget(export_btn)

        # remove graph button
        remove_btn = QPushButton("✕")
        remove_btn.setFixedSize(22, 22)
        remove_btn.setStyleSheet(
            "QPushButton { background:#c0392b; color:white; border:none;"
            "  border-radius:3px; font-weight:bold; }"
            "QPushButton:hover { background:#e74c3c; }"
        )
        remove_btn.clicked.connect(self.on_remove)
        tb_layout.addWidget(remove_btn)
        outer.addWidget(title_bar) 

        # create matplot graph area
        self.figure = Figure(constrained_layout=True)
        # figure background 
        self.figure.patch.set_facecolor("white")
        # matplotlib itself cannot be placed directly into a Qt window - so canvas
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # adds matplotlib tool bar
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.toolbar.setStyleSheet("background:#f0f0f0; color:#333;")

        outer.addWidget(self.toolbar)
        outer.addWidget(self.canvas, 1)

        self.plot()
    
    # when graph widget is closed
    def destroy_resources(self):
        try:
            plt.close(self.figure)
        except Exception:
            pass

    # to reduce no. of points being plotted
    @staticmethod
    def downsample(data_df: pd.DataFrame, max_points: int) -> pd.DataFrame:
        n = len(data_df)
        if n <= max_points:
            return data_df
        step = math.ceil(n / max_points) 
        return data_df.iloc[::step]

    # to plot graph
    def plot(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor("white")

        if not self.series:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, color="#888", fontsize=12)
            ax.set_axis_off()
            self.canvas.draw()
            return

        # load data on-demand — one series at a time
        prepared: list[tuple] = []
        for cat, param_name in self.series:
            data_df = query_parameter_data(self.output_parquet_path, cat, param_name)
            if data_df.empty:
                continue
            plot_df = self.downsample(data_df, self.max_points_per_series)
            max_abs = float(plot_df["value"].abs().max())
            prepared.append((cat, param_name, plot_df, max_abs))
            del data_df  # free immediately after downsampling

        if not prepared:
            ax.text(0.5, 0.5, "No numeric data available", ha="center", va="center", transform=ax.transAxes, color="#888", fontsize=12)
            ax.set_axis_off()
            self.canvas.draw()
            return

        # threshold to create right axis
        large_value = 1000.0
        all_maxes = [m for *_, m in prepared]
        pos_maxes = [m for m in all_maxes if m > 0]
        smallest_max = min(pos_maxes) if pos_maxes else 0.0

        primary: list[tuple] = []
        secondary: list[tuple] = []
        for cat, param_name, plot_df, max_abs in prepared:
            if smallest_max > 0 and max_abs >= smallest_max * large_value:
                secondary.append((cat, param_name, plot_df))
            else:
                primary.append((cat, param_name, plot_df))

        if not primary:
            primary, secondary = secondary, []

        ax2 = ax.twinx() if secondary else None

        lines:  list = []
        labels: list = []
        color_idx = 0
        line_style = ["-", "--", "-.", ":"]
        line_width = [2.2, 1.6, 1.2, 0.9]

        # plotting in left y axis
        for j, (cat, param_name, plot_df) in enumerate(primary):
            color = color_for_index(color_idx); color_idx += 1
            (line,) = ax.plot(
                plot_df["time"], plot_df["value"],
                linewidth=line_width[j % len(line_width)],
                linestyle=line_style[j % len(line_style)],
                label=param_name, color=color,
            )
            lines.append(line); labels.append(param_name)

        # plotting in right y axis
        for j, (cat, param_name, plot_df) in enumerate(secondary):
            color = color_for_index(color_idx); color_idx += 1
            label = f"{param_name} (right axis)"
            (line,) = ax2.plot(
                plot_df["time"], plot_df["value"],
                linewidth=line_width[j % len(line_width)],
                linestyle=line_style[(j + 1) % len(line_style)],
                label=label, color=color,
            )
            lines.append(line); labels.append(label)

        for spine in ax.spines.values():
            spine.set_edgecolor("#ccc")
        ax.set_xlabel("Time", color="#333", fontsize=9)
        ax.set_ylabel("Value", color="#333", fontsize=9)
        ax.tick_params(colors="#333", labelsize=8)
        ax.grid(True, color="#e0e0e0", linewidth=0.5)

        if ax2 is not None:
            for spine in ax2.spines.values():
                spine.set_edgecolor("#ccc")
            ax2.set_ylabel("Value (right axis)", color="#333", fontsize=9)
            ax2.tick_params(colors="#333", labelsize=8)

        # to display parameter name and value when hovering over graph
        cursor = mplcursors.cursor(lines, hover=mplcursors.HoverMode.Transient)

        @cursor.connect("add")
        def on_add(sel):
            idx = lines.index(sel.artist)
            sel.annotation.set_text(
                f"{labels[idx]}\nValue: {sel.target[1]:.3f}"
            )
            sel.annotation.get_bbox_patch().set_alpha(0.9)

        # date format in x axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(30); lbl.set_ha("right")

        # legend — placed below the time axis, all entries laid out in a single row
        if lines:
            self.figure.legend(
                lines, labels,
                loc="outside lower center",
                ncol=len(labels),
                fontsize=8,
                frameon=False,
            )

        self.canvas.draw()

    # to remove graph from graph widget
    def on_remove(self):
        self.on_remove_callback(self.graph_index)

    # to export graph data to csv file
    def on_export_csv(self):
        if not self.series or not self.output_parquet_path:
            QMessageBox.information(self, "Export", "No data to export.")
            return

        # save csv in the same directory as the output parquet file
        out_dir = os.path.dirname(self.output_parquet_path)
        save_path = os.path.join(out_dir, f"graph_{self.graph_index + 1}_export.csv")

        try:
            wide_df = build_wide_export_df(self.output_parquet_path, self.series)

            with open(save_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(list(wide_df.columns))

                # stream rows one at a time — avoids loading all into memory at once
                for row in wide_df.itertuples(index=False):
                    writer.writerow(list(row))

            QMessageBox.information(self, "Export Complete", f"Data exported to:\n{save_path}")
        except Exception as exc:
            QMessageBox.warning(self, "Export Error", f"Failed to export CSV:\n{exc}")

# graph area
class GraphPanel(QWidget):
    # constructor
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.outer = QVBoxLayout(self)
        self.outer.setContentsMargins(0, 0, 0, 0)
        self.outer.setSpacing(0)
        self.setStyleSheet("background-color: white;")

        top_bar = QWidget()
        top_bar.setStyleSheet("background: white;")
        top_bar.setFixedHeight(46)
        tb = QHBoxLayout(top_bar)
        tb.setContentsMargins(10, 6, 10, 6)
        tb.addStretch()

        # export excel button — exports checked left-panel parameters (timestamp + value columns)
        self.export_excel_btn = QPushButton("⬇ Export Excel")
        self.export_excel_btn.setFixedSize(120, 30)
        self.export_excel_btn.setToolTip("Export checked parameters to Excel")
        self.export_excel_btn.setStyleSheet(
            "QPushButton { background:#27ae60; color:white; border:none;"
            "  border-radius:5px; font-size:12px; font-weight:bold; }"
            "QPushButton:hover { background:#2ecc71; }"
        )
        tb.addWidget(self.export_excel_btn)
        tb.addSpacing(8)

        # add graph button
        self.add_btn = QPushButton("+")
        self.add_btn.setFixedSize(30, 30)
        self.add_btn.setStyleSheet(
            "QPushButton {"
            "  background: transparent; color: #333; border: 1px solid #aaa;"
            "  border-radius: 5px; font-size: 16px; font-weight: bold;"
            "}"
            "QPushButton:hover { background: #f0f0f0; }"
            "QPushButton:disabled { color: #ccc; border-color: #ddd; }"
        )
        tb.addWidget(self.add_btn)
        self.outer.addWidget(top_bar)

        # initial widget
        self.content = QWidget()
        self.content.setStyleSheet("background-color: white;")
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(6, 6, 6, 6)

        # initial placeholder text
        self.placeholder = QLabel(
            "Check parameters on the left, then click  +  to add a graph."
        )
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.placeholder.setStyleSheet("color: #888; font-size: 14px;")
        self.content_layout.addWidget(self.placeholder)

        self.container: QWidget | None = None
        self.active_widgets: list[GraphWidget] = []

        self.outer.addWidget(self.content, 1)

    # to rebuild graph area each time a graph is added
    def rebuild(self, graph_widgets: list[GraphWidget]):
        for w in self.active_widgets:
            try:
                w.destroy_resources()
            except Exception:
                pass
        self.active_widgets = []

        if self.container is not None:
            self.content_layout.removeWidget(self.container)
            self.container.deleteLater()
            self.container = None

        n = len(graph_widgets)
        self.active_widgets = list(graph_widgets)

        if n == 0:
            self.placeholder.setVisible(True)
            self.add_btn.setEnabled(True)
            return

        self.placeholder.setVisible(False)
        self.container = QWidget()
        grid = QGridLayout(self.container)
        grid.setSpacing(6)
        grid.setContentsMargins(0, 0, 0, 0)

        # if 1 graph is present
        if n == 1:
            grid.addWidget(graph_widgets[0], 0, 0, 1, 2)
            grid.setRowStretch(0, 1)
            grid.setColumnStretch(0, 1)

        # if 2 graph is present
        elif n == 2:
            grid.addWidget(graph_widgets[0], 0, 0)
            grid.addWidget(graph_widgets[1], 1, 0)
            grid.setRowStretch(0, 1); grid.setRowStretch(1, 1)
            grid.setColumnStretch(0, 1)

        # if 3 graph is present
        elif n == 3: 
            grid.addWidget(graph_widgets[0], 0, 0)
            grid.addWidget(graph_widgets[1], 0, 1)
            grid.addWidget(graph_widgets[2], 1, 0, 1, 2)
            grid.setRowStretch(0, 1); grid.setRowStretch(1, 1)
            grid.setColumnStretch(0, 1); grid.setColumnStretch(1, 1)

        # if 4 graph is present
        elif n == 4:
            grid.addWidget(graph_widgets[0], 0, 0)
            grid.addWidget(graph_widgets[1], 0, 1)
            grid.addWidget(graph_widgets[2], 1, 0)
            grid.addWidget(graph_widgets[3], 1, 1)
            grid.setRowStretch(0, 1); grid.setRowStretch(1, 1)
            grid.setColumnStretch(0, 1); grid.setColumnStretch(1, 1)

        self.content_layout.addWidget(self.container, 1)
        self.add_btn.setEnabled(n < max_graphs)

# main analysis window
class AnalysisWindow(QDialog):
    # progress bar styles
    progress_style_blue = (
        "QProgressBar { border: 1px solid #ccc; border-radius: 3px; background: #e0e0e0; }"
        "QProgressBar::chunk { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
        "  stop:0 #1f77b4, stop:1 #4fa3d1); border-radius: 3px; }"
    )
    progress_style_green = (
        "QProgressBar { border: 1px solid #ccc; border-radius: 3px; background: #e0e0e0; }"
        "QProgressBar::chunk { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
        "  stop:0 #27ae60, stop:1 #2ecc71); border-radius: 3px; }"
    )

    # search box styles
    search_style_default = (
        "QLineEdit { border: 1px solid #bbb; border-radius: 4px;"
        "  padding: 2px 6px; font-size: 12px; background: #fafafa; }"
        "QLineEdit:focus { border-color: #1f77b4; background: white; }"
    )
    search_style_error = (
        "QLineEdit { border: 1px solid #e74c3c; border-radius: 4px;"
        "  padding: 2px 6px; font-size: 12px; background: #fdf2f2; }"
        "QLineEdit:focus { border-color: #e74c3c; background: #fdf2f2; }"
    )

    # constructor
    def __init__(self, stream_path: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.stream_path = stream_path
        # title of window
        self.setWindowTitle("RTE Log Analyzer")
        self.resize(1400, 900)

        self.parameter_categories: dict[str, list] = {}
        self.output_parquet_path: str = ""   # path to processed_log.parquet

        self.graphs: list[list] = []         # list of list[(cat, param_name)]
        self.worker: LoadWorker | None = None

        self.search_timer: QTimer | None = None

        # track whether search filter is currently active
        self.search_active: bool = False

        self.last_load_status: str = "Ready"
        self.last_load_status_style: str = "color: #666; font-size: 12px; min-width: 200px;"

        self.build_ui()

    # main ui 
    def build_ui(self):
        layout = QVBoxLayout(self)

        # title of ui 
        title = QLabel("📊 RTE Log Analyzer")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # horizontal line
        layout.addWidget(self.hline())

        time_row = QHBoxLayout()

        # start date time box
        time_row.addWidget(QLabel("Start Time:"))
        self.start_time_edit = QDateTimeEdit()
        self.start_time_edit.setCalendarPopup(True)
        self.start_time_edit.setDisplayFormat("yyyy-MM-dd HH:mm:ss.zzz")
        self.start_time_edit.setDateTime(QDateTime.currentDateTime().addSecs(-3600))
        self.start_time_edit.setFixedWidth(220)
        time_row.addWidget(self.start_time_edit)
        time_row.addSpacing(20)

        # end date time box
        time_row.addWidget(QLabel("End Time:"))
        self.end_time_edit = QDateTimeEdit()
        self.end_time_edit.setCalendarPopup(True)
        self.end_time_edit.setDisplayFormat("yyyy-MM-dd HH:mm:ss.zzz")
        self.end_time_edit.setDateTime(QDateTime.currentDateTime())
        self.end_time_edit.setFixedWidth(220)
        time_row.addWidget(self.end_time_edit)
        time_row.addStretch()

        # load parameters button
        self.btn_load = QPushButton("Load Parameters")
        self.btn_load.setFixedWidth(140)
        self.btn_load.clicked.connect(self.on_load_clicked)
        time_row.addWidget(self.btn_load)
        layout.addLayout(time_row)

        # horizontal line
        layout.addWidget(self.hline())

        # splitter - left side: parameter names, right side: graphs
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter, 1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        # parameters label
        hdr = QHBoxLayout()
        lbl = QLabel("Parameters")
        lbl.setStyleSheet("font-weight: bold;")
        hdr.addWidget(lbl)

        # search box
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("🔍 Search…")
        self.search_bar.setFixedHeight(26)
        self.search_bar.setStyleSheet(self.search_style_default)
        self.search_bar.textChanged.connect(self.on_search_text_changed)
        self.search_bar.returnPressed.connect(self.on_search_enter)
        hdr.addWidget(self.search_bar)

        left_layout.addLayout(hdr)

        # tree widget to display parameters
        self.parameter_tree = QTreeWidget()
        self.parameter_tree.setHeaderHidden(True)
        left_layout.addWidget(self.parameter_tree)

        # graph area
        self.graph_panel = GraphPanel()
        self.graph_panel.add_btn.clicked.connect(self.add_graph)
        self.graph_panel.export_excel_btn.clicked.connect(self.on_export_excel_clicked)

        splitter.addWidget(left)
        splitter.addWidget(self.graph_panel)
        splitter.setSizes([320, 1080])

        bottom = QWidget()
        bottom.setStyleSheet("background:#f5f5f5; border-top:1px solid #ddd;")
        bottom.setFixedHeight(34)
        bl = QHBoxLayout(bottom)
        bl.setContentsMargins(10, 4, 10, 4)
        bl.setSpacing(10)

        # progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(16)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet(self.progress_style_blue)
        bl.addWidget(self.progress_bar, 1)

        # progress bar text label
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color:#666; font-size:12px; min-width:200px;")
        self.status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        bl.addWidget(self.status_label)

        layout.addWidget(bottom)

        self.build_initial_tree()

    # horizontal separator line
    @staticmethod
    def hline() -> QFrame:
        f = QFrame()
        f.setFrameShape(QFrame.HLine)
        f.setFrameShadow(QFrame.Sunken)
        return f

    # called when loading is in progress
    def set_progress(self, value: int, message: str):
        self.progress_bar.setStyleSheet(self.progress_style_blue)
        self.progress_bar.setValue(value)
        style = "color:#444; font-size:12px; min-width:200px;"
        self.status_label.setStyleSheet(style)
        self.status_label.setText(message)
        self.last_load_status = message
        self.last_load_status_style = style

    # called when loading fails
    def set_progress_error(self, message: str):
        self.progress_bar.setStyleSheet(self.progress_style_blue)
        self.progress_bar.setValue(0)
        style = "color:#c0392b; font-size:12px; font-weight:bold; min-width:200px;"
        self.status_label.setStyleSheet(style)
        self.status_label.setText(message)
        self.last_load_status = message
        self.last_load_status_style = style

    # called when loading completes
    def set_progress_success(self, message: str = "✓ Loaded successfully"):
        self.progress_bar.setStyleSheet(self.progress_style_green)
        self.progress_bar.setValue(100)
        style = "color:#27ae60; font-size:12px; font-weight:bold; min-width:200px;"
        self.status_label.setStyleSheet(style)
        self.status_label.setText(message)
        self.last_load_status = message
        self.last_load_status_style = style

    # used before starting a new load
    def reset_progress(self, message: str = "Ready"):
        self.progress_bar.setStyleSheet(self.progress_style_blue)
        self.progress_bar.setValue(0)
        style = "color:#666; font-size:12px; min-width:200px;"
        self.status_label.setStyleSheet(style)
        self.status_label.setText(message)
        self.last_load_status = message
        self.last_load_status_style = style

    # restores the last saved status message
    def restore_load_status(self):
        self.status_label.setText(self.last_load_status)
        self.status_label.setStyleSheet(self.last_load_status_style)

    # when load parameters is clicked
    def on_load_clicked(self):
        if self.worker is not None and self.worker.isRunning():
            return

        # get user input - start & stop date time
        start_qdt = self.start_time_edit.dateTime()
        end_qdt   = self.end_time_edit.dateTime()

        if start_qdt >= end_qdt:
            QMessageBox.warning(self, "Invalid Range", "Start time must be before end time.")
            return

        start_dt = self.start_time_edit.dateTime().toPython()
        end_dt   = self.end_time_edit.dateTime().toPython()
        
        # disable load parameters button when it is already loading
        self.btn_load.setEnabled(False)
        self.reset_progress("Starting…")

        # output parquet file 
        script_dir  = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, "processed_log.parquet")

        self.worker = LoadWorker(
            stream_path=self.stream_path,
            start_dt= start_dt,
            end_dt = end_dt,
            output_parquet_path=output_path,
            parent=self,
        )
        self.worker.progress.connect(self.set_progress)
        self.worker.error.connect(self.on_worker_error)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()

    # when thread emits error
    def on_worker_error(self, message: str):
        self.set_progress_error(message)
        self.btn_load.setEnabled(True)
        self.worker = None
        QMessageBox.warning(self, "Load Error", message)

    # when all parameters are finished loading
    def on_worker_finished(self, parameter_categories, n_files, output_parquet_path):
        self.set_progress(97, "Updating UI…")
        self.parameter_categories = parameter_categories
        self.output_parquet_path = output_parquet_path
        self.clear_all_graphs()
        self.graph_panel.rebuild([])
        self.rebuild_tree()

        # no. of parameters loaded
        n_params = sum(len(v) for v in self.parameter_categories.values())
        msg = f"✓ {n_params} parameters loaded from {n_files} file(s)"
        self.set_progress_success(msg)

        self.btn_load.setEnabled(True)
        self.worker = None

    # initial empty tree containing category names
    def build_initial_tree(self):
        self.parameter_tree.clear()
        self.tree_items: dict[str, QTreeWidgetItem] = {}
        for cat in cat_keywords:
            item = QTreeWidgetItem(self.parameter_tree, [cat])
            item.setFlags(item.flags() | Qt.ItemIsAutoTristate)
            self.tree_items[cat] = item
        self.parameter_tree.expandAll()

    # reconstructs tree after parquet files are loaded so that parameter names are now visible
    def rebuild_tree(self):
        self.clear_search_filter()
        self.reset_search_state()

        self.parameter_tree.clear()
        self.tree_items = {}

        for cat in cat_keywords:
            params = self.parameter_categories.get(cat, [])
            if not params:
                continue
            cat_item = QTreeWidgetItem(self.parameter_tree, [cat])
            cat_item.setFlags(
                cat_item.flags() | Qt.ItemIsAutoTristate | Qt.ItemIsUserCheckable
            )
            self.tree_items[cat] = cat_item

            seen_params: set[str] = set()
            for param in params:
                if param in seen_params:
                    continue
                seen_params.add(param)
                p_item = QTreeWidgetItem(cat_item, [param])
                p_item.setFlags(p_item.flags() | Qt.ItemIsUserCheckable)
                p_item.setCheckState(0, Qt.Unchecked)
        self.parameter_tree.expandAll()

    # finds all checked parameter in the tree and returns them
    def get_checked_params(self) -> list[tuple[str, str]]:
        result: list[tuple[str, str]] = []
        root = self.parameter_tree.invisibleRootItem()
        for i in range(root.childCount()):
            cat_item = root.child(i)
            cat = cat_item.text(0)
            for j in range(cat_item.childCount()):
                child = cat_item.child(j)
                if child.checkState(0) == Qt.Checked:
                    result.append((cat, child.text(0)))
        return result

    # to add graph
    def add_graph(self):
        # if more than 4 graphs added
        if len(self.graphs) >= max_graphs:
            QMessageBox.warning(self, "Maximum Graph Limit", "Only 4 graphs can be displayed at once.")
            return
        checked = self.get_checked_params()
        # if no parameters are checked
        if not checked:
            QMessageBox.information(self, "No Parameters Selected", "Check at least one parameter on the left.")
            return
        # store only (cat, param_name) tuples —
        self.graphs.append(list(checked))
        self.refresh_graph_panel()

        # uncheck all selected parameters so the next graph can start with a clean slate
        root = self.parameter_tree.invisibleRootItem()
        for i in range(root.childCount()):
            cat_item = root.child(i)
            for j in range(cat_item.childCount()):
                child = cat_item.child(j)
                if child.checkState(0) == Qt.Checked:
                    child.setCheckState(0, Qt.Unchecked)

    # export selected parameters to excel
    def on_export_excel_clicked(self):
        # if parameters are not loaded
        if not self.output_parquet_path:
            QMessageBox.information(self, "Export Excel", "Load parameters first.")
            return

        checked = self.get_checked_params()
        # if no parameters selected
        if not checked:
            QMessageBox.information(self, "No Parameters Selected", "Check at least one parameter on the left.")
            return

        # output excel sheet
        out_dir = os.path.dirname(self.output_parquet_path)
        save_path = os.path.join(out_dir, "parameters_export.xlsx")

        # to excel
        try:
            from collections import defaultdict

            # group selected parameters by category
            category_params = defaultdict(list)
            for cat, param_name in checked:
                category_params[cat].append((cat, param_name))
            with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
                for cat, params in category_params.items():
                    df = build_wide_export_df(self.output_parquet_path, params)
                    df.to_excel(writer, sheet_name=cat[:31], index=False)
            QMessageBox.information(self, "Export Complete", f"Data exported to:\n{save_path}")
            root = self.parameter_tree.invisibleRootItem()
            for i in range(root.childCount()):
                cat_item = root.child(i)
                for j in range(cat_item.childCount()):
                    child = cat_item.child(j)
                    child.setCheckState(0, Qt.Unchecked)
        except Exception as exc:
            QMessageBox.warning(self, "Export Error", f"Failed to export Excel:\n{exc}")

    # when graph is removed - refreshes ui
    def on_graph_removed(self, graph_index: int):
        if 0 <= graph_index < len(self.graphs):
            self.graphs.pop(graph_index)
        self.refresh_graph_panel()

    # clear all graphs when new dataset is loaded
    def clear_all_graphs(self):
        self.graphs.clear()

    # refresh graph area - new graph added, graph is removed, all graphs are cleared
    def refresh_graph_panel(self):
        widgets = [
            GraphWidget(
                series = series,
                on_remove_callback = self.on_graph_removed,
                graph_index = idx,
                output_parquet_path = self.output_parquet_path,
            )
            for idx, series in enumerate(self.graphs)
        ]
        self.graph_panel.rebuild(widgets)

    # reset the search box
    def reset_search_state(self):
        if self.search_timer is not None and self.search_timer.isActive():
            self.search_timer.stop()
        self.search_bar.setStyleSheet(self.search_style_default)

    # search filter — show only matching items, hide the rest 
    def apply_search_filter(self, text: str):
        text_lower = text.lower()
        root = self.parameter_tree.invisibleRootItem()
        match_count = 0

        for i in range(root.childCount()):
            cat_item = root.child(i)
            cat_has_match = False

            for j in range(cat_item.childCount()):
                child = cat_item.child(j)
                matches = text_lower in child.text(0).lower()
                child.setHidden(not matches)
                if matches:
                    cat_has_match = True
                    match_count += 1

            # hide the category header too if none of its children match
            cat_item.setHidden(not cat_has_match)
            if cat_has_match:
                cat_item.setExpanded(True)

        self.search_active = True
        
        # if no match
        if match_count == 0:
            self.search_bar.setStyleSheet(self.search_style_error)
            QTimer.singleShot(800, self.restore_search_bar_if_error)
            self.status_label.setText(f"No match for \"{text}\"")
            self.status_label.setStyleSheet(
                "color:#c0392b; font-size:12px; min-width:200px;"
            )
        # display no. of matches found
        else:
            self.search_bar.setStyleSheet(self.search_style_default)
            self.status_label.setText(f"{match_count} match(es) for \"{text}\"")
            self.status_label.setStyleSheet(
                "color:#1f77b4; font-size:12px; min-width:200px;"
            )
        
    # all tree items are visible when search is cleared
    def clear_search_filter(self):
        if not self.search_active:
            return
        root = self.parameter_tree.invisibleRootItem()
        for i in range(root.childCount()):
            cat_item = root.child(i)
            cat_item.setHidden(False)
            cat_item.setExpanded(True)
            for j in range(cat_item.childCount()):
                cat_item.child(j).setHidden(False)
        self.search_active = False

    # everytime character is typed by user
    def on_search_text_changed(self, text: str):
        self.search_bar.setStyleSheet(self.search_style_default)

        if not text.strip():
            # clear filter and restore full tree
            self.clear_search_filter()
            self.restore_load_status()
            if self.search_timer is not None:
                self.search_timer.stop()
            return

        # debounce: wait 300ms after user stops typing before filtering
        if self.search_timer is None:
            self.search_timer = QTimer(self)
            self.search_timer.setSingleShot(True)
            self.search_timer.timeout.connect(self.run_search_debounced)
        self.search_timer.start(300)

    # runs search when user clicks enter
    def on_search_enter(self):
        text = self.search_bar.text().strip()
        if not text:
            self.clear_search_filter()
            self.restore_load_status()
            return
        if self.search_timer is not None:
            self.search_timer.stop()
        self.apply_search_filter(text) 

    # after user starts typing for 300ms
    def run_search_debounced(self):
        text = self.search_bar.text().strip()
        if text:
            self.apply_search_filter(text)

    # restore after showing a temporary error
    def restore_search_bar_if_error(self):
        if self.search_bar.styleSheet() == self.search_style_error:
            self.search_bar.setStyleSheet(self.search_style_default)

# main function
def main():
    app = QApplication(sys.argv)
    
    # mention stream name here
    stream = "RTE_Stream_logs"

    # find file path
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    stream_path = os.path.join( 
        base_dir, "logs", "parquet",
        "redis=localhost_6379",
        f"stream={stream}"
    )

    # if no file path is found
    if not os.path.isdir(stream_path):
        QMessageBox.critical(None, "Path Error", f"Stream path not found:\n{stream_path}")
        sys.exit(1)

    # to open gui window
    window = AnalysisWindow(stream_path)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()