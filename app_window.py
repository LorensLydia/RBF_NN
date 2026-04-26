#Главное окно приложения
import tkinter as tk
from tkinter import ttk, messagebox
from typing import List, Optional, Tuple

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from activation_functions import ActivationFunctions
from boolean_function import BooleanFunction
from plot_renderer import PlotRenderer
from rbf_network import RbfNetwork
from result_formatter import ResultFormatter
from validation_exception import ValidationException
from widget_factory import WidgetFactory


class AppWindow: #Окно приложения - точка взаимодействия пользователя с программой

    WINDOW_TITLE: str = "ЛР № 4, вариант 10: F = x1*x2 + x3 + x4 (RBF)"
    WINDOW_SIZE: str = "1100x780"
    DEFAULT_EPOCHS_TEXT: str = "200"
    DEFAULT_ETA_TEXT: str = "0.3"
    COMPARE_OPTION: str = "Сравнить ФА 1 и ФА 2"

    def __init__(self) -> None:
        # Всегда инициализируем приватные поля
        self._root: tk.Tk = tk.Tk()
        self._txt_epochs: Optional[ttk.Entry] = None
        self._txt_eta: Optional[ttk.Entry] = None
        self._cmb_activation: Optional[ttk.Combobox] = None
        self._txa_results: Optional[tk.Text] = None
        self._lbl_status: Optional[ttk.Label] = None
        self._figure: Figure = Figure(figsize=(7, 4), dpi=100)
        self._canvas: Optional[FigureCanvasTkAgg] = None
        self._renderer: PlotRenderer = PlotRenderer(self._figure, None)
        self._boolean_function: BooleanFunction = BooleanFunction()
        self._formatter: ResultFormatter = ResultFormatter(
            self._boolean_function
        )

        self._build_ui()
        self._show_initial_info()

    # построение UI

    def _build_ui(self) -> None: #Сконструировать все элементы интерфейса
        self._root.title(self.WINDOW_TITLE)
        self._root.geometry(self.WINDOW_SIZE)
        self._build_controls_frame()
        self._build_main_area()
        self._build_status_label()

    def _build_controls_frame(self) -> None: #Панель ввода параметров и кнопок
        frm_controls: ttk.Frame = ttk.Frame(self._root, padding=10)
        frm_controls.pack(side=tk.TOP, fill=tk.X)

        self._txt_eta = WidgetFactory.make_labeled_entry(
            frm_controls, "Норма обучения η (0;1]:",
            self.DEFAULT_ETA_TEXT, row_index=0, col_index=0
        )
        self._txt_epochs = WidgetFactory.make_labeled_entry(
            frm_controls, "Макс. число эпох:",
            self.DEFAULT_EPOCHS_TEXT, row_index=0, col_index=2
        )

        activation_options: List[str] = [
            ActivationFunctions.get_name(code)
            for code in ActivationFunctions.AVAILABLE_CODES
        ]
        activation_options.append(self.COMPARE_OPTION)
        self._cmb_activation = WidgetFactory.make_combobox(
            frm_controls, "Функция активации:", activation_options,
            row_index=0, col_index=4
        )

        btn_run: ttk.Button = ttk.Button(
            frm_controls, text="Обучить", command=self._on_run_clicked
        )
        btn_run.grid(row=0, column=6, padx=10)

        btn_reset: ttk.Button = ttk.Button(
            frm_controls, text="Сброс", command=self._on_reset_clicked
        )
        btn_reset.grid(row=0, column=7, padx=5)

        lbl_variant: ttk.Label = ttk.Label(
            frm_controls,
            text=("Вариант 10: F = x1·x2 + x3 + x4   "
                  "(RBF: гауссовы скрытые нейроны, обучение Видроу-Хоффа)")
        )
        lbl_variant.grid(row=1, column=0, columnspan=8,
                         sticky=tk.W, padx=5, pady=(8, 0))

    def _build_main_area(self) -> None:
        """Основная область: слева текст, справа график."""
        frm_main: ttk.Frame = ttk.Frame(self._root)
        frm_main.pack(side=tk.TOP, fill=tk.BOTH, expand=True,
                      padx=10, pady=5)

        frm_left: ttk.Frame = ttk.Frame(frm_main)
        frm_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ttk.Label(frm_left, text="Результаты:").pack(anchor=tk.W, padx=2)

        frm_text_holder: ttk.Frame = ttk.Frame(frm_left)
        frm_text_holder.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self._txa_results = WidgetFactory.make_text_area(frm_text_holder)

        frm_right: ttk.Frame = ttk.Frame(frm_main)
        frm_right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True,
                       padx=(10, 0))
        self._canvas = FigureCanvasTkAgg(self._figure, master=frm_right)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._renderer.set_canvas(self._canvas)
        self._renderer.draw_empty()

    def _build_status_label(self) -> None: #Строка состояния
        self._lbl_status = ttk.Label(
            self._root,
            text="Введите параметры и нажмите «Обучить»",
            relief=tk.SUNKEN, anchor=tk.W, padding=5
        )
        self._lbl_status.pack(side=tk.BOTTOM, fill=tk.X)

    # обработчики событий

    def _on_run_clicked(self) -> None: #Обработчик кнопки «Обучить»
        try:
            eta, max_epochs = self._read_and_validate_inputs()
        except ValidationException as error:
            messagebox.showerror("Ошибка ввода", error.message)
            return

        selection: str = self._cmb_activation.get()
        if selection == self.COMPARE_OPTION:
            self._run_comparison(eta, max_epochs)
        else:
            self._run_single(eta, max_epochs, selection)

    def _on_reset_clicked(self) -> None: #Сброс полей и графика
        self._set_entry_text(self._txt_epochs, self.DEFAULT_EPOCHS_TEXT)
        self._set_entry_text(self._txt_eta, self.DEFAULT_ETA_TEXT)
        self._cmb_activation.current(0)
        self._show_initial_info()
        self._renderer.draw_empty()
        if self._lbl_status is not None:
            self._lbl_status.config(
                text="Параметры сброшены к значениям по умолчанию"
            )

    # сценарии запуска

    def _run_single(self,
                    eta: float,
                    max_epochs: int,
                    activation_name: str) -> None: #Обучить НС для одной выбранной ФА и показать результат
        activation_code: int = self._activation_code_from_name(
            activation_name
        )
        history, network = self._train_network(
            eta, max_epochs, activation_code
        )

        text: str = (self._formatter.format_static_info() + "\n"
                     + ResultFormatter.format_single_run(
                         network, history, activation_name))
        self._set_text_area(text)
        self._renderer.draw_single_history(history, activation_name)

        message: str = (f"{activation_name}    |    "
                        f"финальная E(k) = {history[-1]}    |    "
                        f"эпох = {len(history)}")
        if self._lbl_status is not None:
            self._lbl_status.config(text=message)

    def _run_comparison(self, eta: float, max_epochs: int) -> None: #Обучить НС для ФА 1 и ФА 2 и сравнить
        history_th, _ = self._train_network(
            eta, max_epochs, ActivationFunctions.THRESHOLD
        )
        history_sg, _ = self._train_network(
            eta, max_epochs, ActivationFunctions.RATIONAL_SIGMOID
        )

        text: str = (self._formatter.format_static_info() + "\n"
                     + ResultFormatter.format_comparison(
                         history_th, history_sg))
        self._set_text_area(text)
        self._renderer.draw_comparison(history_th, history_sg)

        message: str = (
            f"ФА 1: эпох = {len(history_th)}, E = {history_th[-1]}    |"
            f"    ФА 2: эпох = {len(history_sg)}, E = {history_sg[-1]}"
        )
        if self._lbl_status is not None:
            self._lbl_status.config(text=message)

    def _train_network(self,
                       eta: float,
                       max_epochs: int,
                       activation_code: int
                       ) -> Tuple[List[int], RbfNetwork]: #Создать и обучить сеть; вернуть историю и саму сеть
        network: RbfNetwork = RbfNetwork(
            centers=self._boolean_function.centers,
            learning_rate=eta,
            activation_code=activation_code
        )
        history: List[int] = network.train(
            self._boolean_function.get_input_vectors(),
            self._boolean_function.get_target_outputs(),
            max_epochs
        )
        return history, network

    # валидация и чтение полей

    def _read_and_validate_inputs(self) -> Tuple[float, int]: #Считать и проверить параметры из полей ввода
        try:
            eta_text: str = self._txt_eta.get().strip().replace(",", ".")
            eta: float = float(eta_text)
            max_epochs: int = int(self._txt_epochs.get().strip())
        except ValueError:
            raise ValidationException(
                "Норма обучения должна быть вещественной, "
                "число эпох — целым."
            )

        if not (RbfNetwork.MIN_LEARNING_RATE
                <= eta <= RbfNetwork.MAX_LEARNING_RATE):
            raise ValidationException(
                f"Норма обучения должна лежать в диапазоне "
                f"({RbfNetwork.MIN_LEARNING_RATE}; "
                f"{RbfNetwork.MAX_LEARNING_RATE}]."
            )

        if max_epochs <= 0:
            raise ValidationException(
                "Максимальное число эпох должно быть положительным."
            )

        return eta, max_epochs

    # вспомогательные методы

    @staticmethod
    def _set_entry_text(entry: ttk.Entry, text: str) -> None: #Установить значение в поле Entry
        entry.delete(0, tk.END)
        entry.insert(0, text)

    @staticmethod
    def _activation_code_from_name(name: str) -> int: #По удобочитаемому имени найти код ФА
        for code in ActivationFunctions.AVAILABLE_CODES:
            if ActivationFunctions.get_name(code) == name:
                return code
        raise ValidationException(f"Неизвестная ФА: {name}")

    def _set_text_area(self, text: str) -> None: #Полностью обновить содержимое текстовой области
        if self._txa_results is None:
            return
        self._txa_results.delete("1.0", tk.END)
        self._txa_results.insert(tk.END, text)

    def _show_initial_info(self) -> None: #Стартовое наполнение текстовой области
        self._set_text_area(self._formatter.format_static_info())

    # запуск

    def run(self) -> None: #Запустить главный цикл обработки событий Tkinter
        self._root.mainloop()
