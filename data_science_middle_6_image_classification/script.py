import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

print("Бібліотеки TensorFlow, NumPy, Matplotlib імпортовано.")
print(f"Версія TensorFlow: {tf.__version__}")

# --- 1. Завантаження та підготовка даних CIFAR-10 ---
print("\n--- 1. Завантаження даних CIFAR-10 ---")

# Завантажуємо датасет. Він автоматично розділений на тренувальний та тестовий набори.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(
    f"Розмір тренувальних зображень: {x_train.shape}")  # (50000, 32, 32, 3) - 50k зображень 32x32 пікселі, 3 канали (RGB)
print(f"Розмір тренувальних міток: {y_train.shape}")  # (50000, 1) - мітки від 0 до 9
print(f"Розмір тестових зображень: {x_test.shape}")  # (10000, 32, 32, 3)
print(f"Розмір тестових міток: {y_test.shape}")  # (10000, 1)

# Назви класів CIFAR-10 для візуалізації
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# --- 2. Дослідження даних (Візуалізація прикладів) ---
print("\n--- 2. Візуалізація прикладів зображень ---")
plt.figure(figsize=(10, 10))
for i in range(25):  # Покажемо перші 25 зображень
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i])
    # y_train містить масиви [мітка], тому беремо y_train[i][0]
    plt.xlabel(class_names[y_train[i][0]])
plt.suptitle("Приклади зображень з тренувального набору CIFAR-10")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Відступ для основного заголовка
plt.show()

# --- 3. Передобробка даних ---
print("\n--- 3. Передобробка даних ---")

# 3.1 Нормалізація значень пікселів
# Приводимо значення пікселів з діапазону [0, 255] до діапазону [0, 1]
# Це допомагає нейронній мережі навчатися швидше та стабільніше
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
print("Значення пікселів нормалізовано до діапазону [0, 1].")

# 3.2 Перетворення міток на One-Hot Encoding
# Наприклад, мітка '3' (bird) стане вектором [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
num_classes = 10
y_train_one_hot = to_categorical(y_train, num_classes)
y_test_one_hot = to_categorical(y_test, num_classes)
print("Мітки перетворено на формат One-Hot Encoding.")
print(f"Приклад мітки до перетворення: {y_train[0]}")
print(f"Приклад мітки після перетворення: {y_train_one_hot[0]}")

# --- 4. Побудова моделі CNN ---
print("\n--- 4. Побудова моделі CNN ---")

model = models.Sequential()

# Блок 1: Згортка + Пулінг
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
# 32 - кількість фільтрів, (3, 3) - розмір ядра, 'relu' - функція активації
# padding='same' - доповнення нулями, щоб розмір зображення не зменшувався після згортки
# input_shape - розмір вхідного зображення (лише для першого шару)
model.add(layers.MaxPooling2D((2, 2)))  # Зменшує розмір зображення вдвічі

# Блок 2: Згортка + Пулінг
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))  # Збільшуємо кількість фільтрів
model.add(layers.MaxPooling2D((2, 2)))

# Блок 3: Згортка + Пулінг
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))  # Ще збільшуємо кількість фільтрів
model.add(layers.MaxPooling2D((2, 2)))

# Додаємо Dropout для регуляризації (зменшення перенавчання)
# Він випадково "вимикає" частину нейронів під час тренування
model.add(layers.Dropout(0.25))  # Вимикаємо 25% нейронів

# Перетворення 2D-карт ознак на 1D-вектор
model.add(layers.Flatten())

# Повнозв'язний шар (Dense)
model.add(layers.Dense(512, activation='relu'))  # 512 нейронів
model.add(layers.Dropout(0.5))  # Більший Dropout перед вихідним шаром

# Вихідний шар
# Кількість нейронів = кількість класів (10)
# Функція активації 'softmax' для багатокласової класифікації (видає ймовірності для кожного класу)
model.add(layers.Dense(num_classes, activation='softmax'))

# Виводимо структуру моделі
print("\nСтруктура моделі (Model Summary):")
model.summary()

# --- 5. Компіляція моделі ---
print("\n--- 5. Компіляція моделі ---")
# Вказуємо оптимізатор, функцію втрат та метрики для відстеження
model.compile(optimizer='adam',  # Популярний оптимізатор
              loss='categorical_crossentropy',  # Функція втрат для багатокласової класифікації з one-hot мітками
              metrics=['accuracy'])  # Метрика для оцінки - точність

# --- 6. Тренування моделі ---
print("\n--- 6. Тренування моделі ---")
# Запускаємо процес тренування
# epochs - кількість проходів по всьому тренувальному датасету
# batch_size - кількість зразків, що обробляються за один крок оновлення ваг
# validation_data - дані для перевірки якості моделі після кожної епохи
epochs = 15  # Можна збільшити для кращої точності, але тренування буде довшим
batch_size = 64

history = model.fit(x_train, y_train_one_hot,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(x_test, y_test_one_hot))  # Використовуємо тестовий набір як валідаційний

print("\nТренування завершено.")

# --- 7. Оцінка моделі ---
print("\n--- 7. Оцінка моделі на тестових даних ---")
# Оцінюємо фінальну якість моделі на тестовому наборі
test_loss, test_acc = model.evaluate(x_test, y_test_one_hot, verbose=2)  # verbose=2 показує лише результат
print(f"\nВтрати на тестових даних (Test Loss): {test_loss:.4f}")
print(f"Точність на тестових даних (Test Accuracy): {test_acc:.4f}")

# --- 8. Візуалізація результатів тренування ---
print("\n--- 8. Візуалізація результатів тренування ---")

# Графіки точності та втрат під час тренування
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(12, 5))

# Графік точності
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Точність на тренуванні (Training Accuracy)')
plt.plot(epochs_range, val_acc, label='Точність на валідації (Validation Accuracy)')
plt.legend(loc='lower right')
plt.title('Точність тренування та валідації')
plt.xlabel('Епохи')
plt.ylabel('Точність')

# Графік втрат
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Втрати на тренуванні (Training Loss)')
plt.plot(epochs_range, val_loss, label='Втрати на валідації (Validation Loss)')
plt.legend(loc='upper right')
plt.title('Втрати тренування та валідації')
plt.xlabel('Епохи')
plt.ylabel('Втрати')

plt.suptitle("Результати тренування моделі CNN на CIFAR-10")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# --- 9. Приклад прогнозування (опціонально) ---
print("\n--- 9. Приклад прогнозування на тестових зображеннях ---")
predictions = model.predict(x_test)


# Функція для відображення зображення та його прогнозу
def plot_image_prediction(i, predictions_array, true_label_index, img):
    predictions_array, true_label_index, img = predictions_array[i], true_label_index[i][0], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)  # Відображаємо зображення

    predicted_label_index = np.argmax(predictions_array)  # Знаходимо індекс класу з найбільшою ймовірністю
    confidence = np.max(predictions_array)  # Ймовірність передбаченого класу

    # Колір назви: синій - правильно, червоний - неправильно
    color = 'blue' if predicted_label_index == true_label_index else 'red'

    plt.xlabel(f"{class_names[predicted_label_index]} {confidence * 100:2.0f}% ({class_names[true_label_index]})",
               color=color)


# Функція для відображення графіка ймовірностей класів
def plot_value_array(i, predictions_array, true_label_index):
    predictions_array, true_label_index = predictions_array[i], true_label_index[i][0]
    plt.grid(False)
    plt.xticks(range(num_classes))
    plt.yticks([])
    thisplot = plt.bar(range(num_classes), predictions_array, color="#777777")
    plt.ylim([0, 1])  # Ймовірності від 0 до 1
    predicted_label_index = np.argmax(predictions_array)

    # Підсвічуємо передбачений (червоний/синій) та правильний (синій) стовпчики
    thisplot[predicted_label_index].set_color('red' if predicted_label_index != true_label_index else 'blue')
    thisplot[true_label_index].set_color('blue')


# Відобразимо кілька прикладів прогнозів
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
plt.suptitle("Приклади прогнозів моделі на тестових даних")
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)  # Непарні позиції для зображень
    plot_image_prediction(i, predictions, y_test, x_test)  # x_test тут вже нормалізований
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)  # Парні позиції для графіків ймовірностей
    plot_value_array(i, predictions, y_test)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

print("\n--- Класифікація зображень CIFAR-10 завершена ---")
