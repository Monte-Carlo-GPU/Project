# Тесты и исследования

*1, 2, 3, 4 скриншоты: Увеличиваем количество траекторий для увеличения точности.
Замечание: фактические результаты отдельных запусков могут не показывать ровной убывающей линии ошибки — это нормально для стохастических методов. Со временем и при достаточном числе повторений средняя тенденция действительно будет улучшаться, но не всегда строго по каждому увеличению на порядок.*

---
![image](https://github.com/user-attachments/assets/a2b3e0f4-dfe1-4ea1-864e-1d21ec80bd04)
---
![image](https://github.com/user-attachments/assets/6ec97249-04e7-4067-83e0-b11f5bd27525)
---
![image](https://github.com/user-attachments/assets/a1ffb77b-9795-4122-9523-cceb2502bc7b)
---
![image](https://github.com/user-attachments/assets/2e8d758a-9d0e-4865-92ee-fafe781fd687)
---
---
*5, 6 скриншоты: меняем значение дрейфа в большую и меньшую сторону.
Замечание: Использование дрейфа, отличного от безрисковой ставки, при оценке опциона не даёт безарбитражную цену, которая обычно используется в классической финансовой теории и необходима, например, для хеджирования и установления справедливой рыночной стоимости.*

*Уточнение к замечанию: Например, есть инсайд, что акцию недооценили, и её ожидаемая доходность выше, чем безрисковая ставка. В этом случае использование μ > r при моделировании Монте-Карло может показать, какие выплаты можно ожидать, если предположения о будущем роста акции оправдаются.*

![image](https://github.com/user-attachments/assets/30321a30-fac7-48c9-9832-8151f4040aea)
---
![image](https://github.com/user-attachments/assets/3de74e18-920c-4978-8c8e-054615287363)
---

*7, 8, 9, 10, 11, 12 скриншоты: показывают, что при увеличении количества шагов (дробления времени до момента экспирации)
гладкость траекторий заметна сильнее, но не выявлено корреляции между возрастанием количества шагов и возрастания точности*

---
![image](https://github.com/user-attachments/assets/87a0c675-2e0d-4d92-ae8a-bdcd6bf05e23)
![image](https://github.com/user-attachments/assets/93756462-8aa2-4fe2-9a4c-2b6aec0688fb)
![image](https://github.com/user-attachments/assets/0e2f1749-084f-4bd4-a94b-f405bd607669)
![image](https://github.com/user-attachments/assets/7556c6f0-a60d-4d54-95b9-4206f67fea68)
![image](https://github.com/user-attachments/assets/7c4de9a8-8714-4ac2-bbd3-606d034ea1f9)
![image](https://github.com/user-attachments/assets/dacfbf1f-8248-4a5a-9b3d-e8e254beec94)


---
---

**Итоги проекта**

Когда стоит пользоваться программой:

1) Моделирование нестандартных процессов:
Когда необходимо оценить опцион, базовый актив которого не подчиняется простой модели геометрического броуновского движения из Блэка–Шоулза, а имеет более сложные характеристики.

2) Проверка точности аналитических моделей:
Можно использовать Монте-Карло для проверки результатов аналитических формул или приближений, чтобы убедиться, что формулы дают адекватный результат.

3) Исследовательские цели:
При анализе чувствительности цены к изменениям параметров, изучении влияния нестандартных условий рынка или ограничений ликвидности.
