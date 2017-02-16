To view a more formal walkthrough of my program and the data you can view the [python notebook](https://github.com/iCurlmyster/King-County-House-Data-Set/blob/master/KC_House_Regression.ipynb) or you can view the notebook on my [Kaggle account](https://www.kaggle.com/icurlmyster/d/harlfoxem/housesalesprediction/simple-mlr-model). The original data set can be found [here](https://www.kaggle.com/harlfoxem/housesalesprediction).


### Running the program
[simple_mlr_model.py](https://github.com/iCurlmyster/King-County-House-Data-Set/blob/master/simple_mlr_model.py) is the main file and where the training and testing of the model is done.

[kc_housing_data.csv](https://github.com/iCurlmyster/King-County-House-Data-Set/blob/master/kc_house_data.csv) is the file that the main file pulls its data from.

simple_mlr_model.py has 3 flags that can be set when run:
- "-adjusted" which will show results of the values reverted from being standardized.
- "-plot" which will display all of the plot data inside the script.
- "-v" which will print out more information about the data itself and the program running.

program requires:
- python3 (maybe, I just always use python3 so it could run with python 2)
- tensorflow
- pandas
- numpy
