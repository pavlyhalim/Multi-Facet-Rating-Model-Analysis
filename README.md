
# Multi-Facet Rating Model (MFRM) Analysis Script


This Project implements an advanced Multi-Facet Rating Model (MFRM) analysis for assessing multiple facets of a rating process, such as examinees, raters, and criteria. The script uses powerful tools to uncover patterns, identify potential biases, and measure the reliability of the rating process. This allows you to make more informed decisions based on your rating data.

## Features

- **Parameter Estimation:** Utilizes the L-BFGS-B optimization algorithm to estimate model parameters (e.g., measure, standard error) through maximum likelihood estimation.
- **Fit Statistics:** Calculates and reports various fit statistics, including infit, outfit, and point-measure correlation.
- **Visualization:** Generates insightful visualizations, such as Wright maps, category probability curves, information functions, vertical rulers, and heatmaps for bias interactions.
- **Reliability and Separation:** Calculates and reports reliability and separation indices for the measurement model.
- **Detailed Reports:** Produces comprehensive reports that include measurement summaries for each facet, category statistics, unexpected responses, bias interaction analysis, and IRT results.
- **Additional Analyses:** Conducts additional analyses, such as inter-rater reliability (Cohen's kappa), correlations between criteria, and grading scale usage tendencies.

## Usage

1. **Prepare your data:** Organize your rating data in a CSV file (`data.csv`) with the following columns:
    - `Examinee`: Identifier for the examinee.
    - `Rater`: Identifier for the rater.
    - `Content`, `Organization`, `Language use`: Scores for each criterion (adjust column names if needed).
    - `Overall score`: Overall score (optional, used for additional analyses).
    - `Max_Score`: Maximum possible score for each criterion. 

2. **Run the script:** Execute the script. It will:
    - Load the data from `data.csv`.
    - Run the MFRM analysis.
    - Generate visualizations and save them as PNG files.
    - Print various reports.
    - Perform additional analyses.

## Example Data

```csv
Examinee,Rater,Content,Organization,Language use,Overall score
Student1,RaterA,5,4,5,5
Student2,RaterB,4,3,4,4
Student3,RaterA,3,2,3,3
Student4,RaterB,5,5,5,5
```

## Customization

- **Facets:** Modify the `facets` list in the `__main__` section to specify the facets of your rating process.
- **Criteria:** Adjust the `value_vars` list in the `melt` function to match the names of your criteria columns.
- **Maximum score:** Set the `Max_Score` value based on your scoring system.
- **Visualization options:** Customize the visualization options by modifying the `plt` and `sns` parameters within the plotting functions.

## Notes

- The script assumes that all facets are categorical variables.
- Ensure that the data is correctly formatted before running the script.
- The `data.csv` file should be located in the same directory as the script.
- You can adjust the script to incorporate specific needs or data formats.

## Requirements

- Python 3'
- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `seaborn`

