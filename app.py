import uvicorn
from fastapi import FastAPI
from WeightPredictor import WeightPredictor
import numpy as np
from pyspark.ml.feature import StringIndexerModel, VectorAssembler
from pyspark.ml.regression import LinearRegressionModel
from pyspark.sql import  SparkSession

#create spark session
spark = SparkSession.builder.appName('new_spark').getOrCreate()

#loading the indexer
indexer_model = StringIndexerModel.load('string_indexer.model')
assembler_moddel = VectorAssembler.load('feature_assembler.model')
regressor = LinearRegressionModel.load('reg_model.model')

#create the app object
app = FastAPI()

#create index
@app.get('/')
def index():
    return {'message':'Hellow world, welcome to Weight Predictions'}


@app.get('/what_is_this')
def get_what_is_this():
    return {'Message':'This is a Weight Prediction Model'}

@app.post('/predict')
def predict_weight(data:WeightPredictor):
    data = data.model_dump()
    Height = data['Height']
    Gender = data['Gender']

    #create pyspark dataframe
    col_names = ['Height', 'Gender']
    test_df = spark.createDataFrame([(Height, Gender)], col_names)

    #convert the gender using indexer, and make vector assembler and predict using regressor
    test_df = indexer_model.transform(test_df)
    test_df = assembler_moddel.transform(test_df)
    pred = regressor.transform(test_df)
    return f"Weight predicted for the given Height and Gender is: {pred.select('prediction').head()[0]}"


if __name__ =='__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)