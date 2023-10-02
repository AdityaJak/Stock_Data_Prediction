# Stock Data Prediction Analysis with Machine Learning

This repository contains Python code for performing stock data prediction analysis using various machine learning algorithms. The following machine learning algorithms are implemented in this project:

1. k-Nearest Neighbors (kNN)
2. Logistic Regression
3. Linear Regression
4. Naive Bayes
5. Linear Discriminant Analysis
6. Quadratic Discriminant Analysis
7. Decision Tree
8. Random Forest
9. Support Vector Machine (SVM) with a linear kernel

The goal of this project is to predict stock prices based on historical data and evaluate the performance of different machine learning algorithms in this context.

## Getting Started

To get started with this project, follow the steps below:

### Prerequisites

Before running the code, make sure you have the following Python packages installed:

- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn

You can install these packages using `pip`:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

### Data

You'll need historical stock price data in CSV format for the stock you want to analyze. Place the CSV file in the `data/` directory. Ensure that the CSV file includes at least the following columns:

- `Date`: Date of the stock price.
- `Open`: Opening price of the stock.
- `High`: Highest price of the stock during the day.
- `Low`: Lowest price of the stock during the day.
- `Close`: Closing price of the stock.
- `Volume`: Trading volume of the stock.

### Code

Clone this repository to your local machine:

```bash
git clone https://github.com/AdityaJak/Stock_Data_Prediction.git
```

Navigate to the project directory:

```bash
cd stock-prediction-analysis
```

### Usage

1. Open the Jupyter Notebook file `Stock_Data_Prediction.py`.

2. Follow the instructions and comments in the notebook to perform the following tasks:
   - Data loading and preprocessing.
   - Data visualization and exploratory data analysis (EDA).
   - Splitting the data into training and testing sets.
   - Training and evaluating the machine learning models using the mentioned algorithms.

3. Run each code cell in the notebook to execute the analysis and predictions.

## Results and Evaluation

The project includes model evaluation metrics such as accuracy, precision, recall, and F1-score to assess the performance of each machine learning algorithm for stock price prediction.

## Acknowledgments

- The code in this repository is for educational purposes and should not be used for actual stock trading decisions.
- Make sure to have a solid understanding of the stock market and financial data analysis before using any predictive models in real-world scenarios.

## Author

[Adithya Jakkaraju]

Feel free to reach out with any questions or suggestions!

Happy coding and happy stock analysis!
