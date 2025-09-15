# -*- coding: utf-8 -*-
"""
Vibration Analysis Tool
Refactored: Kelvin T
Updated: Auto units & TW type, Temperature forced, no °C in dropdown, restore vibration units, grid added
Dynamic subplot resizing based on plot mode: TWF, FFT, TWF+FFT
Mouse hover tooltips showing x,y values
"""

import tkinter as tk
from tkinter import ttk, filedialog, font
import customtkinter
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import ast
from scipy.fft import rfft, rfftfreq
import mplcursors  # <-- For hover tooltips

# ------------------ Settings ------------------
customtkinter.set_appearance_mode("Dark")
customtkinter.set_default_color_theme("blue")

# ------------------ Root Window ------------------
root = customtkinter.CTk()
root.geometry("1000x400")
root.title("Vibration Analysis Tool")

# ------------------ Global Variables ------------------
df = pd.DataFrame()
current_unit = tk.StringVar(value="g")
current_tw_type = tk.StringVar(value="Acceleration")
current_plot_mode = tk.StringVar(value="TWF+FFT")  # Default selection

last_vibration_unit = "g"  # Keep last vibration unit to restore when returning from temperature

unit_factors = {"g": 1, "mg": 1000, "mm/s²": 9.81, "mm/s": 0.001, "°C": 1}

# ------------------ Frames ------------------
nav_frame = customtkinter.CTkFrame(root, height=50)
nav_frame.pack(side="top", fill="x", padx=5, pady=5)

left_frame = customtkinter.CTkFrame(root, width=250)
left_frame.pack(side="left", fill="y", padx=10, pady=10)

right_frame = customtkinter.CTkFrame(root)
right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

# ------------------ Navigation Bar ------------------
customtkinter.CTkButton(nav_frame, text="📂 Open DB", command=lambda: load_csv()).pack(side="left", padx=5)

# Plot Mode Switch
plot_mode_switch = customtkinter.CTkOptionMenu(
    nav_frame,
    values=["TWF","FFT","TWF+FFT"],
    variable=current_plot_mode,
    command=lambda v: plot_selected()
)
plot_mode_switch.pack(side="left", padx=5)

# Units Dropdown (only vibration units)
unit_dropdown = customtkinter.CTkOptionMenu(
    nav_frame,
    values=["g","mg","mm/s²","mm/s"],
    variable=current_unit,
    command=lambda v: set_last_vibration_unit()
)
unit_dropdown.pack(side="right", padx=10)

# ------------------ Treeview ------------------
tree = ttk.Treeview(left_frame, show="tree", height=25)
tree.pack(fill="y", expand=True, padx=5, pady=5)

bg_color = root._apply_appearance_mode(customtkinter.ThemeManager.theme["CTkFrame"]["fg_color"])
text_color = root._apply_appearance_mode(customtkinter.ThemeManager.theme["CTkLabel"]["text_color"])
selected_color = root._apply_appearance_mode(customtkinter.ThemeManager.theme["CTkButton"]["fg_color"])
tree_font = font.Font(family="Helvetica", size=12)

style = ttk.Style()
style.theme_use('default')
style.configure("Treeview", font=tree_font, background=bg_color,
                fieldbackground=bg_color, foreground=text_color, borderwidth=0)
style.map('Treeview', background=[('selected', bg_color)],
          foreground=[('selected', selected_color)])

# ------------------ Matplotlib Figure ------------------
fig = plt.figure(figsize=(10,6))
canvas = FigureCanvasTkAgg(fig, master=right_frame)
canvas.get_tk_widget().pack(fill="both", expand=True)
toolbar = NavigationToolbar2Tk(canvas, right_frame)
toolbar.update()

# ------------------ Functions ------------------
def load_csv():
    global df
    path = filedialog.askopenfilename(filetypes=[("DB Files","*.db")])
    if not path:
        return
    df = pd.read_csv(path)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    build_tree()

def build_tree():
    tree.delete(*tree.get_children())
    for c in sorted(df['company'].unique()):
        c_id = tree.insert('', 'end', text=c)
        for l in sorted(df[df['company']==c]['location'].unique()):
            l_id = tree.insert(c_id,'end',text=l)
            for m in sorted(df[(df['company']==c)&(df['location']==l)]['machine'].unique()):
                m_id = tree.insert(l_id,'end',text=m)
                for p in sorted(df[(df['company']==c)&(df['location']==l)&(df['machine']==m)]['point'].unique()):
                    p_id = tree.insert(m_id,'end',text=p)
                    for meas in sorted(df[(df['company']==c)&(df['location']==l)&(df['machine']==m)&(df['point']==p)]['measurement'].unique()):
                        tree.insert(p_id,'end',text=meas)

def set_last_vibration_unit():
    global last_vibration_unit
    if current_unit.get() != "°C":
        last_vibration_unit = current_unit.get()
    plot_selected()

def plot_selected(event=None):
    global last_vibration_unit
    if df.empty: return
    sel = tree.selection()
    if not sel: return
    path = []
    node = sel[0]
    while node:
        path.insert(0, tree.item(node,'text'))
        node = tree.parent(node)
    if len(path)!=5: return
    company, loc, mach, point, meas = path
    subset = df[(df['company']==company)&(df['location']==loc)&(df['machine']==mach)&(df['point']==point)&(df['measurement']==meas)]
    if subset.empty: return

    meas_lower = meas.lower()
    
    # --- Auto adjust unit and TW type ---
    if meas_lower in ["axial","horizontal","vertical"]:
        if current_unit.get() == "°C":
            current_unit.set(last_vibration_unit)
        if current_unit.get() in ["g","mg","mm/s²"]:
            current_tw_type.set("Acceleration")
        elif current_unit.get() == "mm/s":
            current_tw_type.set("Velocity")
        elif current_unit.get() == "mm":
            current_tw_type.set("Displacement")
    elif "temperature" in meas_lower:
        current_unit.set("°C")
        current_tw_type.set("Temperature")

    # --- Get data ---
    if 'values' in subset.columns:
        row = subset.iloc[0]
        y = np.array(ast.literal_eval(row['values'])) * unit_factors[current_unit.get()]
        sr = float(row['sampling_rate'])
        t = np.arange(len(y))/sr
    else:
        y = subset['value'].to_numpy() * unit_factors[current_unit.get()]
        t = subset['timestamp'].to_numpy()
        sr = 1/((t[1]-t[0]).total_seconds() if len(t)>1 else 1)

    # Adjust TW type for vibration
    tw_type = current_tw_type.get()
    if tw_type == "Velocity":
        y = np.cumsum(y)/sr
    elif tw_type == "Displacement":
        y = np.cumsum(np.cumsum(y))/sr**2

    # Clear figure
    fig.clf()
    mode = current_plot_mode.get()

    if mode == "TWF":
        ax = fig.add_subplot(111)
        ax.plot(t, y, label=f"{meas} [{current_unit.get()}] ({tw_type})", color='b')
        ax.set_title(f"{meas} - {tw_type} Time Waveform")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"Amplitude [{current_unit.get()}]")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        mplcursors.cursor(ax.get_lines(), hover=True)
    elif mode == "FFT" and meas_lower in ["axial","horizontal","vertical"]:
        ax = fig.add_subplot(111)
        xf = rfftfreq(len(y),1/sr)
        yf = np.abs(rfft(y))
        ax.plot(xf, yf, label=f"FFT [{current_unit.get()}]", color='r')
        ax.set_title(f"{meas} - FFT")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(f"Amplitude [{current_unit.get()}]")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        mplcursors.cursor(ax.get_lines(), hover=True)
    else:  # TWF+FFT
        ax_top = fig.add_subplot(211)
        ax_bottom = fig.add_subplot(212)
        ax_top.plot(t, y, label=f"{meas} [{current_unit.get()}] ({tw_type})", color='b')
        ax_top.set_title(f"{meas} - {tw_type} Time Waveform")
        ax_top.set_xlabel("Time (s)")
        ax_top.set_ylabel(f"Amplitude [{current_unit.get()}]")
        ax_top.grid(True, linestyle='--', alpha=0.6)
        ax_top.legend()

        if meas_lower in ["axial","horizontal","vertical"]:
            xf = rfftfreq(len(y),1/sr)
            yf = np.abs(rfft(y))
            ax_bottom.plot(xf, yf, label=f"FFT [{current_unit.get()}]", color='r')
            ax_bottom.set_title(f"{meas} - FFT")
            ax_bottom.set_xlabel("Frequency (Hz)")
            ax_bottom.set_ylabel(f"Amplitude [{current_unit.get()}]")
            ax_bottom.grid(True, linestyle='--', alpha=0.6)
            ax_bottom.legend()

        mplcursors.cursor(ax_top.get_lines() + ax_bottom.get_lines(), hover=True)

    fig.tight_layout()
    canvas.draw()

tree.bind("<<TreeviewSelect>>", plot_selected)

# ------------------ Run App ------------------
root.mainloop()
