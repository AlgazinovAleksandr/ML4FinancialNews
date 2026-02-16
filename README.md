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

Build pipeline for multiple vectorizers - Oleg. At least 3-4 vectorizers including FinBERT or smth like this, transformer like BERT, tf-idf, tf-idf + svd, word2vec, etc

Build regressor - Alex. Try multiple like boosting, RF, MLP, etc

Obtain additional features like market caps, volatilities, volumes, risk-free rates, etc - Alex

Think about explainability through SHAP (check references) - Alex

Do not forget to evaluate task-specific LLMs like https://github.com/Bavest/fin-llama for recommendations, and compare them with standard LLMs like GPT
