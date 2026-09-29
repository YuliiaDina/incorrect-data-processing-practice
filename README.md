# incorrect-data-processing-practice

This repository contains completed lab assignments (8 in total) for the **Incorrect Data Processing Problems** course (5th year of study).

The main goal of these practices is to explore floating-point issues. The assignments demonstrate how mathematically perfect formulas can fail due to the limits of machine arithmetic, and how to protect algorithms from such errors using `NumPy`, `SciPy`, `scikit-learn`, and `SymPy`.

## Interactive Reports (Google Colab)

In addition to the raw `.py` scripts required for automated grading, I have prepared extended versions of each assignment as interactive Google Colab notebooks.

These notebooks are designed to preserve the context of the solutions. They include:
* Detailed comments and step-by-step explanations of the logic.
* Mini-reports on handling edge cases and specific machine errors (e.g., mitigating machine zero or avoiding `NaN`).
* Additional conclusions that serve as a convenient reference for me and anyone interested in the topic.

**[Open the folder with all Colab reports](https://drive.google.com/drive/folders/1wIkvg-C9aDu1dbA_aip1gF4ItCnBhWo6?usp=sharing)**

## Course Structure and Progress

The assignments are divided into two parts: calling library functions (`.py` scripts) and building machine learning algorithms from scratch (`.ipynb` notebooks).

**Part 1: Foundations of Linear Algebra and Computation**
- [x] Practice 1: Vectors, norms, angles, linear systems (`vectors.py`)
- [ ] Practice 2: Matrix arithmetic, rank and basis (`matrices.py`)
- [ ] Practice 3: Linear and affine mappings (`matrix_mapping.py`)
- [ ] Practice 4: Matrix decomposition (LU, QR, SVD) (`matrix_decomposition.py`)
- [ ] Practice 5: Regularization (Ridge, Lasso) (`regularization.py`)

**Part 2: Building Algorithms**
- [ ] Practice 6: Gaussian mixtures and the EM algorithm (`tutorial_gmm.ipynb`)
- [ ] Practice 7: PCA and Eigenfaces (`tutorial_pca.ipynb`)
- [ ] Practice 8: SVM dual and the kernel trick (`tutorial_svm.ipynb`)

---

# Некоректна обробка даних — Практичні роботи

Цей репозиторій містить виконані лабораторні роботи з курсу **«Некоректна обробка даних»**. 

Головна мета практик — дослідити специфіку обчислень із плаваючою комою (floating-point issues) та навчитися захищати алгоритми від машинних похибок.

## Інтерактивні звіти (Google Colab)

Окрім чистих скриптів із кодом для автоматичної перевірки, є розширені версії кожної роботи у форматі зошитів Google Colab. Вони містять детальні коментарі, пояснення логіки, обґрунтування обробки крайніх випадків та підсумків щодо кожної роботи.

**[Відкрити папку з усіма звітами в Google Colab](https://drive.google.com/drive/folders/1wIkvg-C9aDu1dbA_aip1gF4ItCnBhWo6?usp=sharing)**

## Структура курсу
**Частина 1: Основи лінійної алгебри та обчислень**
- [x] Практика 1: Вектори, норми, кути та лінійні системи (`vectors.py`)
- [ ] Практика 2: Матрична арифметика, ранг та базис (`matrices.py`)
- [ ] Практика 3: Лінійні та афінні відображення (`matrix_mapping.py`)
- [ ] Практика 4: Декомпозиція матриць (LU, QR, SVD) (`matrix_decomposition.py`)
- [ ] Практика 5: Регуляризація моделей (Ridge, Lasso) (`regularization.py`)

**Частина 2: Побудова алгоритмів**
- [ ] Практика 6: Gaussian mixtures та EM-алгоритм (`tutorial_gmm.ipynb`)
- [ ] Практика 7: Метод головних компонент (PCA) та Eigenfaces (`tutorial_pca.ipynb`)
- [ ] Практика 8: Метод опорних векторів (SVM dual) та kernel trick (`tutorial_svm.ipynb`)
