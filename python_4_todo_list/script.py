import json  # Модуль для роботи з файлами формату JSON
import os  # Модуль для взаємодії з операційною системою (перевірка існування файлу)

# Ім'я файлу, де будуть зберігатися завдання
TASKS_FILE = "tasks.json"


def is_yes(answer: str) -> bool:
    """Повертає True, якщо відповідь користувача означає «так».
    Підтримує декілька поширених варіантів: "так", "y", "yes", "т".
    """
    if not isinstance(answer, str):
        return False
    return answer.strip().lower() in {"так", "y", "yes", "т"}


def load_tasks():
    """Завантажує список завдань із файлу JSON."""
    # Перевіряємо, чи існує файл
    if os.path.exists(TASKS_FILE):
        try:
            # Відкриваємо файл для читання
            with open(TASKS_FILE, 'r', encoding='utf-8') as f:
                # Намагаємося прочитати дані з файлу JSON
                tasks = json.load(f)
                # Переконуємося, що завантажені дані є списком
                if isinstance(tasks, list):
                    return tasks
                else:
                    print("Попередження: Формат файлу завдань некоректний. Створюється новий список.")
                    return []  # Повертаємо порожній список у разі некоректного формату
        except json.JSONDecodeError:
            # Обробка помилки, якщо файл порожній або містить невалідний JSON
            print(f"Попередження: Файл {TASKS_FILE} порожній або пошкоджений. Створюється новий список.")
            return []
        except Exception as e:
            # Обробка інших можливих помилок читання файлу
            print(f"Помилка при завантаженні завдань: {e}")
            return []
    else:
        # Якщо файл не існує, повертаємо порожній список
        return []


def save_tasks(tasks):
    """Зберігає поточний список завдань у файл JSON атомарно (через тимчасовий файл)."""
    try:
        temp_path = TASKS_FILE + ".tmp"
        # Записуємо у тимчасовий файл
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(tasks, f, indent=4, ensure_ascii=False)
            # Гарантуємо запис на диск перед заміною
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                # На деяких платформах fsync може бути недоступний/непотрібний
                pass
        # Атомарно замінюємо основний файл
        os.replace(temp_path, TASKS_FILE)
    except Exception as e:
        # Обробка можливих помилок запису
        print(f"Помилка при збереженні завдань: {e}")


def display_tasks(tasks):
    """Виводить список завдань на екран."""
    print("\n--- Список завдань ---")
    if not tasks:
        print("Список порожній.")
    else:
        # Нумеруємо завдання для зручності користувача (починаючи з 1)
        for i, task in enumerate(tasks):
            # Визначаємо статус завдання для відображення
            status = "[X]" if task.get('done', False) else "[ ]"  # .get() безпечно отримує значення
            description = task.get('description', 'Немає опису')  # .get() з значенням за замовчуванням
            print(f"{i + 1}. {status} {description}")
    print("-" * 20)


def add_task(tasks):
    """Додає нове завдання до списку."""
    description = input("Введіть опис нового завдання: ").strip()
    if description:  # Перевіряємо, чи користувач щось ввів
        # Створюємо словник для нового завдання
        new_task = {"description": description, "done": False}
        tasks.append(new_task)  # Додаємо завдання до списку
        print("Завдання успішно додано.")
        return True  # Повертаємо True, щоб позначити, що список було змінено
    else:
        print("Опис завдання не може бути порожнім.")
        return False


def mark_task_done(tasks):
    """Позначає завдання як виконане."""
    display_tasks(tasks)  # Показуємо завдання, щоб користувач міг вибрати номер
    if not tasks:
        return False  # Нічого позначати, якщо список порожній

    while True:  # Цикл для отримання коректного номера завдання
        try:
            task_num_str = input("Введіть номер завдання, яке потрібно позначити як виконане: ")
            task_num = int(task_num_str)
            # Перевіряємо, чи номер дійсний (в межах списку)
            if 1 <= task_num <= len(tasks):
                # Номер користувача починається з 1, а індекс списку - з 0
                task_index = task_num - 1
                if not tasks[task_index]['done']:
                    tasks[task_index]['done'] = True  # Змінюємо статус
                    print(f"Завдання {task_num} позначено як виконане.")
                    return True  # Список було змінено
                else:
                    print(f"Завдання {task_num} вже було позначено як виконане.")
                    return False  # Список не змінився
            else:
                print("Недійсний номер завдання.")
        except ValueError:
            print("Будь ласка, введіть дійсний номер.")
        # Запитуємо чи користувач хоче спробувати ще раз, якщо був невірний номер
        try_again = input("Спробувати ввести номер ще раз? (так/ні): ")
        if not is_yes(try_again):
            break  # Виходимо з циклу запиту номера
    return False  # Якщо не вдалося позначити завдання


def delete_task(tasks):
    """Видаляє завдання зі списку."""
    display_tasks(tasks)  # Показуємо завдання для вибору
    if not tasks:
        return False  # Нічого видаляти

    while True:  # Цикл для отримання коректного номера
        try:
            task_num_str = input("Введіть номер завдання, яке потрібно видалити: ")
            task_num = int(task_num_str)
            if 1 <= task_num <= len(tasks):
                task_index = task_num - 1
                # Запитуємо підтвердження перед видаленням
                confirm = input(
                    f"Ви впевнені, що хочете видалити завдання \"{tasks[task_index]['description']}\"? (так/ні): ")
                if is_yes(confirm):
                    # Видаляємо завдання за індексом
                    deleted_task = tasks.pop(task_index)
                    print(f"Завдання \"{deleted_task['description']}\" успішно видалено.")
                    return True  # Список змінено
                else:
                    print("Видалення скасовано.")
                    return False
            else:
                print("Недійсний номер завдання.")
        except ValueError:
            print("Будь ласка, введіть дійсний номер.")
        # Запитуємо чи користувач хоче спробувати ще раз
        try_again = input("Спробувати ввести номер ще раз? (так/ні): ")
        if not is_yes(try_again):
            break  # Виходимо з циклу запиту номера
    return False  # Якщо не вдалося видалити


def main():
    """Головна функція програми."""
    tasks = load_tasks()  # Завантажуємо завдання при старті

    while True:  # Головний цикл меню
        # Виводимо опції меню
        print("\n--- Менеджер завдань ---")
        print("1. Переглянути завдання")
        print("2. Додати завдання")
        print("3. Позначити завдання як виконане")
        print("4. Видалити завдання")
        print("5. Вийти")
        print("-" * 24)

        choice = input("Виберіть опцію (1-5): ")

        needs_saving = False  # Прапорець, чи потрібно зберігати зміни

        if choice == '1':
            display_tasks(tasks)
        elif choice == '2':
            if add_task(tasks):
                needs_saving = True
        elif choice == '3':
            if mark_task_done(tasks):
                needs_saving = True
        elif choice == '4':
            if delete_task(tasks):
                needs_saving = True
        elif choice == '5':
            # На виході зберігаємо поточний стан завдань
            save_tasks(tasks)
            print("Дякуємо за використання! Зміни збережено.")
            break  # Виходимо з головного циклу
        else:
            print("Невірний вибір. Будь ласка, введіть число від 1 до 5.")

        # Зберігаємо зміни у файл, якщо вони були
        if needs_saving:
            save_tasks(tasks)


# Запускаємо головну функцію, якщо скрипт виконується напряму
if __name__ == "__main__":
    main()
