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
