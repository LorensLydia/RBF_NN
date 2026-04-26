#Модуль построения виджетов главного окна

import tkinter as tk
from tkinter import ttk
from typing import List


class WidgetFactory: #Фабрика виджетов главного окна

    @staticmethod
    def make_labeled_entry(parent: ttk.Frame,
                           label_text: str,
                           default_value: str,
                           row_index: int,
                           col_index: int,
                           width: int = 8) -> ttk.Entry: #Создать связку «метка + поле ввода» в указанной позиции grid
        ttk.Label(parent, text=label_text).grid(
            row=row_index, column=col_index, sticky=tk.W, padx=5
        )
        entry: ttk.Entry = ttk.Entry(parent, width=width)
        entry.insert(0, default_value)
        entry.grid(row=row_index, column=col_index + 1, padx=5)
        return entry

    @staticmethod
    def make_combobox(parent: ttk.Frame,
                      label_text: str,
                      options: List[str],
                      row_index: int,
                      col_index: int,
                      width: int = 28) -> ttk.Combobox: #Создать связку «метка + выпадающий список
        ttk.Label(parent, text=label_text).grid(
            row=row_index, column=col_index, sticky=tk.W, padx=5
        )
        combobox: ttk.Combobox = ttk.Combobox(
            parent, values=options, width=width, state="readonly"
        )
        combobox.current(0)
        combobox.grid(row=row_index, column=col_index + 1, padx=5)
        return combobox

    @staticmethod
    def make_text_area(parent: ttk.Frame,
                       width: int = 55,
                       height: int = 30) -> tk.Text: #Создать текстовую область со скроллбаром
        text_widget: tk.Text = tk.Text(
            parent, width=width, height=height,
            font=("Courier New", 9)
        )
        scroll: ttk.Scrollbar = ttk.Scrollbar(
            parent, command=text_widget.yview
        )
        text_widget.config(yscrollcommand=scroll.set)
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        return text_widget
