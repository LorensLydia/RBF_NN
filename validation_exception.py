#Модуль с пользовательским классом исключения для проверки входных данных
class ValidationException(Exception):
    def __init__(self, message: str = "Ошибка валидации входных данных"):
        # Всегда инициализируем поле, даже при наличии значения по умолчанию
        self._message: str = message
        super().__init__(self._message)

    @property
    def message(self) -> str: #возвращает текст сообщения об ошибке
        return self._message
