from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('Practise').getOrCreate()

training = spark.read.csv('Salary_Data.csv', header=True, inferSchema=True)

training.printSchema()

training.show(4)

from pyspark.ml.feature import VectorAssembler

# 독립변수를 만들어준다. age, Experience 를 사용한다.

featureAssembler = VectorAssembler(
    inputCols=['Age', 'YearsExperience'],
    outputCol='Independent Features'
)

output = featureAssembler.transform(training)

output.show()

finalized_data = output.select(['Independent features', 'Salary'])

from pyspark.ml.regression import LinearRegression

train_data, test_data = finalized_data.randomSplit([0.75, 0.25], seed=None)
regressor = LinearRegression(featuresCol='Independent features', labelCol='Salary')
regressor = regressor.fit(train_data)

# print(regressor.coefficients)
# print(regressor.intercept)

pred_result = regressor.evaluate(test_data)
pred_result.predictions.show()