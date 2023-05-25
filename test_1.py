import pandas as pd

from pyspark.sql import SparkSession
from pyspark.ml.feature import Imputer

# print(pd.read_csv('test1.csv'))

spark = SparkSession.builder.appName('Practise').getOrCreate()

df_pyspark = spark.read.csv('test2.csv', header=True, inferSchema=True)
# df_pyspark.show()

imputer = Imputer(
    inputCols=['age', 'Experience', 'Salary'],
    outputCols=[f'{i}_imputed' for i in ['age', 'Experience', 'Salary']]
    ).setStrategy('mean')

imputer.fit(df_pyspark).transform(df_pyspark).show()

imputer = Imputer(
    inputCols=['age', 'Experience', 'Salary'],
    outputCols=[f'{i}_imputed' for i in ['age', 'Experience', 'Salary']]
    ).setStrategy('median')

imputer.fit(df_pyspark).transform(df_pyspark).show()
