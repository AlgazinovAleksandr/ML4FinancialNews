# ML4FinancialNews

The raw data for the dataset we used can be found at https://github.com/felixdrinkall/financial-news-dataset

To merge all the .json files into one .csv file, first uncompress the data using the following command:

```bash
xz -d data/*.xz
```
After that, run data/data_preprocesing.py in the folder with the decompressed .json files:

```bash
python data_preprocesing.py
```
By default, the output is saved into news_prices.csv

### TODO:

+ Think about explainability through SHAP (check references)

+ Chose the best regression model, try to convert the task to the classification one and check how it will work out

+ Check complex_eda.ipynb

+ Do not forget to evaluate task-specific LLMs like https://github.com/Bavest/fin-llama for recommendations, and compare them with standard LLMs like GPT

# References

https://github.com/felixdrinkall/financial-news-dataset - take data from there

https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks - S&P 500 stocks and index itself additional info


