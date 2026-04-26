#Модуль форматирования текстовой сводки результатов лабораторной
from typing import List

from boolean_function import BooleanFunction
from rbf_network import RbfNetwork


class ResultFormatter: #Преобразование объектов-данных в человекочитаемые текстовые блоки для отображения в текстовой области главного окна
    def __init__(self, boolean_function: BooleanFunction):
        # Всегда инициализируем приватное поле
        self._boolean_function: BooleanFunction = boolean_function

    # статическая информация

    def format_static_info(self) -> str: #Информация, не зависящая от запуска
        lines: List[str] = []
        lines.append("ЛР4, вариант 10")
        lines.append("БФ: F(x1,x2,x3,x4) = x1*x2 + x3 + x4\n")

        lines.append("Таблица истинности:")
        lines.append("  x1 x2 x3 x4 | F")
        lines.append("  ------------+--")
        for row in self._boolean_function.truth_table:
            lines.append(
                f"   {row[0]}  {row[1]}  {row[2]}  {row[3]} | {row[4]}"
            )

        target: int = self._boolean_function.target_value
        lines.append("")
        lines.append(
            f"Центры RBF (точки, где F = {target}, "
            f"J = {self._boolean_function.neuron_count}):"
        )
        for index, center in enumerate(self._boolean_function.centers, 1):
            lines.append(f"  C^({index}) = {center}")
        return "\n".join(lines) + "\n"

    # результаты одиночного обучения
    @staticmethod
    def format_single_run(network: RbfNetwork,
                          history: List[int],
                          activation_name: str) -> str: #Сформировать блок с итогами одного запуска
        lines: List[str] = []
        lines.append(f"--- Результаты обучения ({activation_name}) ---")
        lines.append(f"Эпох до останова: {len(history)}")
        lines.append(f"Финальная E(k) = {history[-1]}")
        lines.append("")

        lines.append("Синаптические коэффициенты выходного нейрона:")
        lines.append(f"  v0 (bias) = {network.bias:+.4f}")
        for index, weight in enumerate(network.weights, 1):
            lines.append(f"  v{index}        = {weight:+.4f}")

        return "\n".join(lines) + "\n"

    # результаты сравнительного запуска

    @staticmethod
    def format_comparison(history_threshold: List[int],
                          history_sigmoid: List[int]) -> str: # Подробный вывод результатов сравнения двух ФА
        lines: List[str] = ["--- Сравнение ФА ---"]

        lines.append(
            f"ФА 1 (пороговая):  E_final = {history_threshold[-1]}, "
            f"эпох = {len(history_threshold)}"
        )
        lines.append(
            f"ФА 2 (сигмоида):   E_final = {history_sigmoid[-1]}, "
            f"эпох = {len(history_sigmoid)}"
        )
        return "\n".join(lines) + "\n"
