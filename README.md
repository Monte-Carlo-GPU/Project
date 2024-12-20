# Расчет стоимости опциона методом Монте-Карло на GPU

**Базовая пользовательская документация**

1. **Назначение**  
   Эта программа оценивает стоимость европейских опционов (колл или пут) методом Монте-Карло с использованием GPU (CUDA) для ускорения вычислений. Результаты сравнены с аналитическим решением по формуле Блэка–Шоулза. Программа также сохраняет часть сгенерированных траекторий для дальнейшего анализа.

2. **Требования к окружению**  
   - CUDA Toolkit, совместимый с вашим GPU.  
   - Компилятор C++ (например, `nvcc`) и необходимые библиотеки (например, `curand`).

3. **Сборка**  
   Пример команды для сборки (уточните путь к файлам при необходимости):
   ```bash
   nvcc -std=c++14 main.cpp kernel.cu -o program.exe -lcurand
   ```

4. **Запуск**  
   Выполните:
   ```bash
   ./program.exe
   ```
   При запуске программа запросит у вас:  
   - Тип опциона (1 — колл, 2 — пут)  
   - Число траекторий и число шагов  
   - Цена базового актива, страйк-цена, безрисковая ставка, волатильность, время до экспирации, дрейф

   После ввода всех параметров начнётся моделирование.

5. **Вывод результатов**  
   Программа отобразит:  
   - Оценку стоимости опциона по Монте-Карло (GPU)  
   - Аналитическую цену по формуле Блэка–Шоулза  
   - Разницу между ними  
   - Время расчёта

   Результаты будут добавлены в `results.txt`.  
   Дополнительно часть траекторий будет сохранена в `trajectories.csv` для анализа.

6. **Обслуживание и диагностика**  
   При некорректном вводе или проблемах с ресурсами GPU программа отобразит сообщение об ошибке. Код структурирован и прокомментирован, что облегчает модификацию и добавление новых функций при необходимости.






---


**Основные возможности программы**

- **Метод Монте-Карло на GPU:**  
  Программа генерирует большое число случайных траекторий цены базового актива, используя нормальные случайные числа. Расчёт выполняется параллельно на GPU, что существенно ускоряет вычисления.

- **Аналитический расчёт:**  
  Дополнительно к численному результату методом Монте-Карло программа рассчитывает теоретическую цену опциона по формуле Блэка–Шоулза. Это позволяет сравнить точность метода Монте-Карло с аналитическим решением.

- **Логирование результатов:**  
  Все ключевые данные — параметры моделирования, рассчитанная цена опциона методом Монте-Карло, аналитическая цена, разница между ними, а также затраченное время — сохраняются в текстовый файл для дальнейшего анализа.

- **Сохранение траекторий:**  
  Программа может сохранить ряд сгенерированных ценовых траекторий в формат CSV. Это упрощает последующую визуализацию динамики цен с помощью внешних инструментов.






---


**Функциональные требования**

- **Расчёт справедливой стоимости опциона методом Монте-Карло:**  
  Программа должна вычислять стоимость европейского опциона (колл или пут) методом Монте-Карло.  
  **Ввод:** тип опциона, страйк, время до экспирации, текущая цена базового актива, волатильность, безрисковая ставка.  
  **Вывод:** оценка теоретической цены опциона на основе случайных симуляций.

- **Генерация случайных чисел на GPU:**  
  Использование cuRAND для генерации нормальных случайных чисел непосредственно на GPU.

- **Параллельная обработка на GPU:**  
  Запуск большого числа траекторий в параллельных потоках GPU для ускорения вычислений.

- **Возможность задания параметров моделирования:**  
  Пользователь должен иметь возможность указать количество траекторий, число временных шагов, а также параметры рынка и опциона (S0, K, T, r, σ).

- **Аналитические проверки:**  
  Сравнение численной оценки (Метод Монте-Карло) с аналитической ценой, полученной по формуле Блэка–Шоулза.

- **Логирование результатов:**  
  Сохранение параметров моделирования и результатов вычислений (опционная цена, временные характеристики) в файл для последующего анализа.

- **Визуализация (данные для неё):**  
  Сохранение ряда сгенерированных траекторий в CSV-файл для их дальнейшей визуализации внешними инструментами.

- **Ключевое требование:**  
  Программа должна как минимум сохранять результаты траекторий в файл для последующего построения графиков.






---


**Системные требования**

- **Аппаратная среда:**  
  Наличие GPU с поддержкой CUDA (NVIDIA), достаточный объём оперативной памяти и видеопамяти для работы с большими выборками случайных чисел и траекторий.

- **Программная среда:**  
  Компилятор C++17 или выше, установленный CUDA Toolkit, библиотека cuRAND.  
  Опционально любые дополнительные инструменты для визуализации данных (внешние программы или библиотеки).

- **Производительность и масштабируемость:**  
  Способность эффективно обрабатывать сотни тысяч или миллионы траекторий за разумное время, с возможностью увеличения количества траекторий и шагов по мере необходимости.

- **Надёжность:**  
  Корректная обработка ошибок GPU, проверка корректности входных параметров и предотвращение неконтролируемых сбоев.






---


**Дизайн программной архитектуры**

- **Модуль конфигурации и ввода параметров (Config/Parameters Module):**  
  - Функции: Чтение параметров опциона и моделирования, проверка валидности ввода.  
  - Интерфейс: Предоставляет структуру или класс для хранения параметров опциона и симуляции.

- **Модуль инициализации GPU и генерации случайных чисел (GPU Init & Random Module):**  
  - Функции: Инициализация GPU, создание и настройка генератора cuRAND, генерация нормальных случайных чисел.  
  - Интерфейс: Предоставляет функции для инициализации, генерации d_normals (указатель на GPU-память).

- **Модуль вычисления цены опциона методом Монте-Карло (MonteCarloPricing Module):**  
  - Функции: Запуск CUDA-ядра для расчёта стоимостей опционов по траекториям, вычисление средней цены.  
  - Интерфейс: Функция computeOptionPriceOnGPU(...), возвращающая среднюю опционную цену.

- **Модуль аналитического расчёта (AnalyticalPricing Module):**  
  - Функции: Формулы Блэка–Шоулза для европейского колл/пут опциона.  
  - Интерфейс: blackScholesCall(...), blackScholesPut(...).

- **Модуль логирования результатов (Logging Module):**  
  - Функции: Запись параметров и результатов вычислений (MC-цена, аналитическая цена, разница, время) в файл results.txt.  
  - Интерфейс: logResults(...).

- **Модуль сохранения траекторий (Trajectories Module):**  
  - Функции: Копирование части сгенерированных нормальных чисел с GPU на хост, пересчёт траекторий на CPU и сохранение их в trajectories.csv.  
  - Интерфейс: saveTrajectories(...).

- **Главный модуль (main):**  
  - Последовательность действий:  
    1. Чтение параметров (Config/Parameters).  
    2. Инициализация GPU и генерация случайных чисел (GPU Init & Random).  
    3. Запуск Монте-Карло расчётов (MonteCarloPricing).  
    4. Аналитический расчёт (AnalyticalPricing).  
    5. Логирование результатов (Logging).  
    6. Сохранение некоторых траекторий (Trajectories).  
  - Отображение результатов на экране.

**Взаимодействия между модулями:**  
- Main → Parameters: чтение и валидация входных данных.  
- Main → GPU Init & Random: инициализация CUDA и генерация случайных чисел.  
- Main → MonteCarloPricing: передача d_normals и параметров для расчёта цены опциона.  
- Main → AnalyticalPricing: расчёт аналитической цены.  
- Main → Logging: запись всех результатов и параметров в файл.  
- Main → Trajectories: сохранение ряда траекторий в CSV для внешней визуализации.


---
![image](https://github.com/user-attachments/assets/f2337183-9d52-4fd2-aa8f-5415ff196c70)

---

**Потоки данных:**  
- Ввод: Параметры от пользователя (консольный ввод).  
- Обработка: Генерация и расчёты на GPU.  
- Вывод: Результаты в файл (results.txt, trajectories.csv) и на экран.

**Расширяемость:**  
- Можно добавить другие опционные модели или источники вводных данных.  
- Можно улучшить модуль визуализации, подключив дополнительные библиотеки или инструменты.






---


**Дополнительная литература:**  
- Ценообразование опционов: Black, F. & Scholes, M. (1973). Journal of Political Economy; Hull, J.C. "Options, Futures, and Other Derivatives".  
- Метод Монте-Карло: Glasserman, P. (2004). "Monte Carlo Methods in Financial Engineering", Springer.  
- CUDA и cuRAND: Документация NVIDIA (https://docs.nvidia.com/cuda/index.html).

---

**Итоги проекта:**
Выявлены и отображены, как и все тесты и исследования в файле test&research.md

---

**Авторы проекта:**
Понетайкин Дмитрий и Черник Матвей
