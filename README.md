Sure. Here is a clean **README.md** for your repository covering **Mathematical Transformations, Function Transformations, and Power Transformations** in Machine Learning.

# Mathematical Transformations in Machine Learning

This repository contains my practical work on **Mathematical Transformations in Machine Learning**. The purpose of this practice is to understand how mathematical transformations can be applied to numerical features during data preprocessing and feature engineering.

Mathematical transformations are useful when the distribution of a feature is highly skewed, when values have a large range, or when transforming the data can make it more suitable for a Machine Learning model.

## Topics Covered

In this repository, I practiced different types of mathematical transformations, including:

- Function Transformations
- Power Transformations
- Log Transformation
- Square Root Transformation
- Reciprocal Transformation
- Exponential Transformation
- Box-Cox Transformation
- Yeo-Johnson Transformation

---

## 1. Function Transformations

A function transformation means applying a mathematical function to the values of a feature.

For example, if a feature is represented by:

```text
X
```

we can transform it using different mathematical functions.

### Common Function Transformations

#### Log Transformation

```text
X' = log(X)
```

Log transformation can be useful for reducing right skewness and compressing large values.

#### Square Root Transformation

```text
X' = √X
```

Square root transformation can reduce moderate positive skewness.

#### Reciprocal Transformation

```text
X' = 1/X
```

Reciprocal transformation changes the relationship between large and small values and can sometimes help with highly skewed data.

#### Exponential Transformation

```text
X' = eˣ
```

Exponential transformation increases the effect of larger values and is generally used when there is a specific reason to transform the feature in this direction.

---

# 2. Power Transformations

Power transformations apply a mathematical power to the feature values.

A general power transformation can be represented as:

```text
X' = X^λ
```

where **λ (lambda)** determines the type and strength of the transformation.

Power transformations are commonly used to make data more Gaussian-like and reduce skewness.

## Box-Cox Transformation

The **Box-Cox transformation** is a power transformation that is commonly used for positive-valued data.

In Scikit-learn, it can be applied using:

```python
from sklearn.preprocessing import PowerTransformer

transformer = PowerTransformer(method='box-cox')
```

Box-Cox requires the input data to contain **positive values**.

---

## Yeo-Johnson Transformation

The **Yeo-Johnson transformation** is another power transformation.

One important advantage is that it can work with **zero and negative values**, unlike Box-Cox.

Example:

```python
from sklearn.preprocessing import PowerTransformer

transformer = PowerTransformer(method='yeo-johnson')
```

---

# 3. Why Transform Data?

Data transformation can be useful for:

- Reducing skewness
- Making distributions more symmetrical
- Handling large differences in feature values
- Improving the suitability of features for certain models
- Improving the effectiveness of some statistical and Machine Learning techniques
- Creating better-behaved numerical features

Transformation does not automatically improve every Machine Learning model. Its usefulness depends on the dataset, feature distribution, and algorithm being used.

---

# 4. Scikit-learn Implementation

I practiced mathematical transformations using Scikit-learn preprocessing tools.

Example:

```python
from sklearn.preprocessing import PowerTransformer

pt = PowerTransformer(method='yeo-johnson')

X_transformed = pt.fit_transform(X)
```

For a Box-Cox transformation:

```python
pt = PowerTransformer(method='box-cox')

X_transformed = pt.fit_transform(X)
```

---

# 5. Before and After Transformation

During this practice, I compared feature distributions before and after applying transformations.

For example:

```text
Original Data
     ↓
Check Distribution
     ↓
Identify Skewness
     ↓
Apply Transformation
     ↓
Check Transformed Data
```

Visualization tools such as **Matplotlib** and **Seaborn** can be used to compare the distributions before and after transformation.

---

# Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

# Learning Goal

The main goal of this repository is to build a practical understanding of **mathematical transformations used in Machine Learning preprocessing**.

Through this practice, I learned how different mathematical functions and power transformations can change feature distributions and how tools such as **Box-Cox and Yeo-Johnson** can be implemented using Scikit-learn.

This repository is part of my ongoing **Machine Learning and AI learning journey**.
