import secrets  # Більш безпечний генератор випадкових значень
import string  # Модуль, що містить корисні рядкові константи


def generate_password(length, use_uppercase, use_digits, use_symbols):
    """
    Генерує випадковий пароль заданої довжини та складності.

    Гарантує принаймні один символ з кожної обраної категорії.

    Args:
      length (int): Бажана довжина пароля.
      use_uppercase (bool): Чи включати великі літери.
      use_digits (bool): Чи включати цифри.
      use_symbols (bool): Чи включати спеціальні символи.

    Returns:
      str: Згенерований пароль.
    """

    # Категорії символів
    lower = list(string.ascii_lowercase)  # Малі літери завжди включені
    upper = list(string.ascii_uppercase) if use_uppercase else []
    digits = list(string.digits) if use_digits else []
    symbols = list(string.punctuation) if use_symbols else []

    # Підрахунок обов'язкових категорій (ті, що вибрані) + lowercase
    required_sets = [lower]
    if use_uppercase:
        required_sets.append(upper)
    if use_digits:
        required_sets.append(digits)
    if use_symbols:
        required_sets.append(symbols)

    # Перевірка довжини: має бути не меншою за кількість обов'язкових категорій
    if length < len(required_sets):
        raise ValueError(
            f"Довжина пароля замала. Мінімум: {len(required_sets)}, задано: {length}."
        )

    # Побудова пулу символів для решти виборів
    character_pool = lower + upper + digits + symbols

    # Формуємо початковий список, гарантуючи наявність кожної категорії
    password_chars = [secrets.choice(char_set) for char_set in required_sets]

    # Дозаповнюємо до потрібної довжини з повного пулу
    remaining = length - len(password_chars)
    password_chars.extend(secrets.choice(character_pool) for _ in range(remaining))

    # Перемішуємо результуючий список символів безпечно
    for i in range(len(password_chars) - 1, 0, -1):  # Fisher–Yates
        j = secrets.randbelow(i + 1)
        password_chars[i], password_chars[j] = password_chars[j], password_chars[i]

    return "".join(password_chars)


# Основна частина програми для взаємодії з користувачем
if __name__ == "__main__":
    print("Генератор безпечних паролів")
    print("-" * 30)

    # Отримуємо бажану довжину пароля від користувача
    password_length = 12  # Ініціалізація за замовчуванням, щоб уникнути 'може бути не визначено'
    while True:  # Цикл для перевірки коректності вводу довжини
        password_length_str = input("Введіть бажану довжину пароля (Enter = 12): ").strip()
        password_length_str = password_length_str or "12"  # Якщо пустий, стає "12"

        try:
            password_length = int(password_length_str)
            if password_length <= 0:
                print("Довжина має бути додатним числом!")
                continue
            break  # валідне значення отримане, виходимо з циклу
        except ValueError:
            print("Невірний формат числа!")
            continue

    # Запитуємо користувача про використання різних типів символів
    # Функція для спрощення запиту так/ні
    def ask_yes_no(prompt):
        while True:
            answer = input(f"{prompt} (так/ні): ").lower().strip()
            if answer in ['так', 'т', 'yes', 'y', '+']:
                return True
            elif answer in ['ні', 'н', 'no', 'n', '-']:
                return False
            else:
                print("Будь ласка, введіть 'так' або 'ні'.")


    include_uppercase = ask_yes_no("Включати великі літери (A-Z)?")
    include_digits = ask_yes_no("Включати цифри (0-9)?")
    include_symbols = ask_yes_no(f"Включати спеціальні символи ({string.punctuation})?")

    # Генеруємо пароль за допомогою нашої функції
    try:
        generated_password = generate_password(password_length, include_uppercase, include_digits, include_symbols)
    except ValueError as e:
        # Наприклад, коли довжина менша за кількість вибраних категорій
        print(f"Помилка: {e}")
    else:
        # Виводимо згенерований пароль
        print("-" * 30)
        print(f"Ваш згенерований пароль: {generated_password}")
        print("-" * 30)
