from app_window import AppWindow


def main() -> None: #Создать главное окно и запустить цикл событий
    application: AppWindow = AppWindow()
    application.run()


if __name__ == "__main__":
    main()
