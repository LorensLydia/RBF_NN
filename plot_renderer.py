#Модуль с классом, отвечающим за визуализацию результатов

from typing import List, Optional

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class PlotRenderer: #Отрисовщик кривой сходимости обучения и сравнения двух ФА
    def __init__(self,
                 figure: Figure,
                 canvas: Optional[FigureCanvasTkAgg]):
        # Всегда инициализируем приватные поля
        self._figure: Figure = figure
        self._canvas: Optional[FigureCanvasTkAgg] = canvas

    def set_canvas(self, canvas: FigureCanvasTkAgg) -> None:
        self._canvas = canvas

    def draw_empty(self) -> None:
        self._figure.clear()
        axes = self._figure.add_subplot(1, 1, 1)
        axes.text(0.5, 0.5,
                  "Нажмите «Обучить» — здесь появится график E(k)",
                  ha="center", va="center", fontsize=11)
        axes.set_xticks([])
        axes.set_yticks([])
        self._refresh_canvas()

    def draw_single_history(self,
                            error_history: List[int],
                            activation_name: str) -> None: #Нарисовать кривую сходимости одной ФА
        self._figure.clear()
        axes = self._figure.add_subplot(1, 1, 1)

        epochs_range = range(1, len(error_history) + 1)
        axes.plot(epochs_range, error_history, "b-o",
                  markersize=4, linewidth=1)
        axes.set_title(
            f"E(k) - суммарная ошибка по эпохам [{activation_name}]"
        )
        axes.set_xlabel("Номер эпохи k")
        axes.set_ylabel("E(k)")
        axes.grid(True)
        self._refresh_canvas()

#Сравнить процесс обучения для двух ФА варианта 10 на одном графике
    def draw_comparison(self,
                        history_threshold: List[int],
                        history_sigmoid: List[int]) -> None:
        self._figure.clear()
        axes = self._figure.add_subplot(1, 1, 1)

        threshold_x = range(1, len(history_threshold) + 1)
        sigmoid_x = range(1, len(history_sigmoid) + 1)
        axes.plot(threshold_x, history_threshold, "b-o",
                  markersize=4, linewidth=1, label="ФА 1: пороговая")
        axes.plot(sigmoid_x, history_sigmoid, "r-s",
                  markersize=4, linewidth=1,
                  label="ФА 2: рациональная сигмоида")

        axes.set_title("Сравнение сходимости двух ФА (вариант 10)")
        axes.set_xlabel("Номер эпохи k")
        axes.set_ylabel("E(k)")
        axes.grid(True)
        axes.legend()
        self._refresh_canvas()

    def _refresh_canvas(self) -> None: #Обновить отображение после изменения figure
        self._figure.tight_layout()
        if self._canvas is not None:
            self._canvas.draw()
