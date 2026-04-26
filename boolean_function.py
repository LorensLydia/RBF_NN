#исходные данные задачи: булева функция и таблица истинности

from typing import List, Tuple

class BooleanFunction:

    # Размерность входного пространства для всех вариантов ЛР № 4
    INPUT_DIMENSION: int = 4
    TRUTH_TABLE_SIZE: int = 2 ** INPUT_DIMENSION  # = 16

    def __init__(self) -> None:
        # Всегда инициализируем приватные поля
        self._truth_table: List[Tuple[int, int, int, int, int]] = [] #16 строк таблицы истинности, каждая строка - кортеж (х1, х2, х3, х4, F)
        self._centers: List[Tuple[int, int, int, int]] = [] #найденный минимальный набор центров RBF
        self._target_value: int = 0 #

        self._build_truth_table()
        self._select_centers()

    @property
    def truth_table(self) -> List[Tuple[int, int, int, int, int]]: #Полная таблица истинности (копия)
        return list(self._truth_table)

    @property
    def centers(self) -> List[Tuple[int, int, int, int]]: #Координаты центров RBF-нейронов (копия)
        return list(self._centers)

    @property
    def target_value(self) -> int: #Значение БФ, в точках которого расположены центры (0 или 1)
        return self._target_value

    @property
    def neuron_count(self) -> int: #Количество RBF-нейронов J = длина списка центров
        return len(self._centers)

    @staticmethod
    def evaluate(x1: int, x2: int, x3: int, x4: int) -> int: #Вычисляем значение БФ
        and_part: int = x1 & x2
        or_result: int = and_part | x3 | x4
        return or_result

    def get_input_vectors(self) -> List[Tuple[int, int, int, int]]: #Все 16 входных векторов (без значений F)
        return [(row[0], row[1], row[2], row[3])
                for row in self._truth_table]

    def get_target_outputs(self) -> List[int]: #Целевые значения F для всех 16 входных векторов
        return [row[4] for row in self._truth_table]

    def _build_truth_table(self) -> None: #построение таблицы истинности
        for combination in range(self.TRUTH_TABLE_SIZE):
            x1: int = (combination >> 3) & 1
            x2: int = (combination >> 2) & 1
            x3: int = (combination >> 1) & 1
            x4: int = combination & 1
            f_value: int = self.evaluate(x1, x2, x3, x4)
            self._truth_table.append((x1, x2, x3, x4, f_value))

    def _select_centers(self) -> None:
        zeros: List[Tuple[int, int, int, int]] = []
        ones: List[Tuple[int, int, int, int]] = []

        for row in self._truth_table:
            point: Tuple[int, int, int, int] = (row[0], row[1],
                                                row[2], row[3])
            if row[4] == 0:
                zeros.append(point)
            else:
                ones.append(point)

        if len(zeros) <= len(ones):
            self._centers = zeros
            self._target_value = 0
        else:
            self._centers = ones
            self._target_value = 1
