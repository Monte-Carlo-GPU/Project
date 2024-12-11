# Project
Функциональные и системные требования

Функциональные требования

    Расчет справедливой стоимости опциона методом Монте-Карло:
    Программа должна оценивать справедливую стоимость европейского опциона на акцию (например, европейский колл или пут) с помощью метода Монте-Карло.
        На вход: параметры опциона (тип: колл или пут, страйк, срок до экспирации), текущая цена базового актива, волатильность, безрисковая ставка.
        На выход: оценка справедливой стоимости (теоретической цены) опциона.

    Генерация случайных чисел на GPU:
    Необходимо использовать высококачественный ГПСЧ (генератор псевдослучайных чисел) на GPU, например использование cuRAND для генерации нормальных случайных величин.

    Параллельная обработка на GPU:
    Реализация расчёта множества траекторий для стохастического процесса (модель Блэка–Шоулза) будет выполняться на GPU с целью ускорения вычислений.

    Возможность задания параметров моделирования:
        Количество траекторий (число симуляций).
        Число временных шагов внутри каждого моделирования цены.
        Параметры опциона и рынка (S0, K, T, r, σ).

    Аналитические проверки:
    Программа должна давать возможность сравнить результат численного метода с известным аналитическим решением (для европейского опциона на не дивидендную акцию есть формула Блэка–Шоулза). Это позволит оценить точность.

    Логирование результатов:
    Сохранение результатов вычислений (средняя цена опциона, доверительный интервал) и параметров симуляции в файл.

    Визуализация:
        Предоставление данных о нескольких сгенерированных траекториях цены актива для последующей визуализации.
        Визуализация может выполняться в отдельном модуле или в внешнем инструменте (например, Python + matplotlib), для построения графиков ценовых траекторий.

    Ключевое требование: программа должна как минимум сохранять результаты траекторий в файл для последующего построения графиков.

Системные требования

    Аппаратная среда:
        GPU с поддержкой CUDA (NVIDIA)
        Достаточный объем оперативной памяти (RAM) и видеопамяти (VRAM) для хранения больших массивов случайных чисел и траекторий.

    Программная среда:
        Компилятор C++17 или выше (например, g++ или clang++).
        NVIDIA CUDA Toolkit для поддержки GPU-кода.
        Библиотека cuRAND для генерации случайных чисел.
        Дополнительно: если требуется визуализация непосредственно в C++, можно использовать любую графическую библиотеку (например, gnuplot, QtCharts, SFML или вывод данных для последующей обработки в Python).

    Производительность и масштабируемость:
        Программа должна эффективно обрабатывать сотни тысяч или миллионы траекторий за разумное время (секунды или минуты).
        Поддержка масштабируемого количества траекторий и временных шагов.

    Надежность:
        Корректная обработка ошибок, связанных с GPU (переполнение памяти, неудача при инициализации генераторов чисел).
        Валидация входных параметров (например, неотрицательность сроков, корректность типа опциона и т.п.).


Ниже представлен дизайн программной архитектуры обновлённой программы, включающий основные модули, их ответственность и взаимодействие. Данный дизайн основан на текущем функционале программы, который включает:

    Чтение параметров и конфигурации.
    Расчет цены опциона методом Монте-Карло на GPU.
    Аналитический расчет цены по модели Блэка–Шоулза.
    Логирование результатов (с дописыванием в конец файла).
    Сохранение нескольких траекторий в CSV.
    Возможность дальнейшей визуализации извне.

Основные компоненты архитектуры

    Модуль конфигурации и ввода параметров (Config/Parameters Module)
    Функции и ответственность:
        Чтение параметров опциона (тип, S0, K, r, sigma, T, mu).
        Чтение параметров моделирования (N_PATHS, N_STEPS).
        Проверка корректности вводимых данных.

    Интерфейс:
        Предоставляет структуру/класс OptionParams для хранения параметров опциона.
        Предоставляет структуру или отдельные переменные для параметров симуляции.

    Модуль инициализации среды GPU и генерации случайных чисел (GPU Init & Random Module)
    Функции:
        Инициализация CUDA-устройств, проверка доступной памяти.
        Создание и инициализация cuRAND генератора.
        Генерация нормальных случайных чисел для N_PATHS * N_STEPS.

    Интерфейс:
        Функции initGPU(), initRandomGenerator(), generateNormals().
        Обеспечивает d_normals — указатель на GPU-память с нормальными числами.

    Модуль вычисления цены опциона методом Монте-Карло (MonteCarloPricing Module)
    Функции:
        Запуск CUDA-ядра (mc_call_GPU или mc_put_GPU) для расчёта выплат по опционам для большого количества траекторий.
        Возврат средней цены опциона после дисконтирования.

    Интерфейс:
        Функция computeOptionPriceOnGPU(…), принимающая параметры опциона, d_normals, N_PATHS, N_STEPS, и возвращающая среднее значение опционной цены.
        Использует памяти и данные, подготовленные в предыдущих модулях.

    Модуль аналитического расчёта по Блэку–Шоулзу (AnalyticalPricing Module)
    Функции:
        Реализация формулы Блэка–Шоулза для европейского колл/пут опциона.
        Функция blackScholesCall(...) и blackScholesPut(...).

    Интерфейс:
        Предоставляет функции для аналитического расчёта цены опциона, принимая S0, K, r, sigma, T.

    Модуль логирования результатов (Logging Module)
    Функции:
        Запись результатов (параметры, цены, время расчёта) в текстовый файл results.txt с добавлением в конец файла (append mode).

    Интерфейс:
        logResults(...) — функция, принимающая параметры опциона, Monte Carlo цену, аналитическую цену, время выполнения и записывающая их в results.txt.

    Модуль сохранения траекторий (Trajectories Module)
    Функции:
        Копирование подмножества нормальных случайных чисел (для N_TRAJ_TO_SAVE траекторий) с GPU на хост.
        Пересчет ценовых траекторий на CPU (по шагам) с использованием сгенерированных нормальных чисел.
        Запись результатов по шагам во trajectories.csv в требуемом формате.

    Интерфейс:
        saveTrajectories(...) — функция, принимающая N_TRAJ_TO_SAVE, h_normals, параметры опциона, N_STEPS, S0, sigma, r, T и записывающая файл trajectories.csv.

    Главный модуль (main)
    Функции:
        Последовательный вызов вышеописанных модулей:
            Считать параметры из консоли (Config/Parameters Module).
            Инициализировать GPU и генератор случайных чисел (GPU Init & Random Module).
            Генерировать нормальные числа для всех траекторий.
            Запустить метод Монте-Карло (MonteCarloPricing Module) и получить среднюю цену.
            Вычислить аналитическую цену (AnalyticalPricing Module).
            Протоколировать результаты (Logging Module).
            Выбрать несколько траекторий, пересчитать их на CPU и сохранить для визуализации (Trajectories Module).
        Выводить результаты на экран.

Взаимодействия между модулями:

    Main → Parameters: main вызывает модуль параметров для чтения входных данных.
    Main → GPU Init & Random: main вызывает функции инициализации GPU и генерации случайных чисел.
    Main → MonteCarloPricing: main передает d_normals и параметры в этот модуль и получает среднюю цену опциона.
    Main → AnalyticalPricing: main вызывает функции для расчёта аналитической цены, передавая параметры опциона.
    Main → Logging: main передаёт все результаты (параметры, цены, время) в функцию логирования.
    Main → Trajectories: main передает d_normals, параметры и количество сохраняемых траекторий в модуль сохранения траекторий, который на CPU пересчитывает их и пишет trajectories.csv.

Потоки данных:

    Ввод: Пользовательские параметры вводятся через стандартный ввод, проверяются и сохраняются в структуру параметров.
    GPU-память: Модули GPU Init & Random, MonteCarloPricing работают напрямую с GPU-памятью (d_normals, d_S_batch).
    Вывод: Результаты записываются в results.txt (append mode) и trajectories.csv (снова создаётся при каждом запуске), а также выводятся на экран.

Расширяемость:

    Можно добавить модуль визуализации (например, генерация gnuplot-скриптов), но в данном дизайне визуализация предполагается внешним инструментом.
    Можно добавить еще модули для чтения конфигурации из файла, если нужно.
    Можно модифицировать Trajectories Module для сохранения большего числа траекторий или параметризации количества сохраняемых траекторий.

Итог

Полученная архитектура модульна: каждый модуль отвечает за свой аспект функциональности. Главный модуль (main) выступает в роли «дирижёра», управляя последовательностью вызовов модулей. При необходимости можно изменить или улучшить отдельные модули (например, заменить способ логирования или визуализации) без значительных изменений в других частях кода.
