# Conditional Average Treatment Effect (CATE) Estimation Project

This project aims to estimate Conditional Average Treatment Effects (CATE) using multiple machine learning methods, compare their performance, and provide baseline and advanced models for heterogeneous treatment effect analysis.

---

## Project Structure

- `baseline_random_forest.py`  
  Implements a baseline CATE estimation model using Random Forest. Suitable for quick validation of data and method effectiveness.

- `baseline_dnn.py`  
  Implements a baseline Deep Neural Network (DNN) model to estimate treatment effects, capturing complex nonlinear relationships.

- `transformer_cate.py`  
  Uses a Transformer-based architecture for CATE estimation, leveraging its ability to capture sequence and contextual information for flexible and fine-grained heterogeneous effect analysis.

- `DoubleML_PLR_CATE_Comparison.py`  
  Implements the Double Machine Learning (DML) Partial Linear Regression (PLR) framework combined with Random Forests for CATE estimation, and compares results against other models.

---

## Environment & Dependencies

It is recommended to create and activate a virtual environment before installing dependencies:

```bash
pip install -r requirements.txt
