
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 


def summary_table(data):
    summary_col_map = {'index':'field_name',0:'data_type',1:'number_unique_values',2:'number_null_values'}
    data_summary = pd.concat([data.dtypes,data.nunique(), data.isnull().sum(axis = 0)],axis=1).reset_index().rename(mapper = summary_col_map, axis =1)
    print(f'The train dataset contains {data.shape[0]} rows and {data.shape[1]} fields after removing instance_weight field')
    print(f'Number of categorical fields are {len(data.select_dtypes(["object"]).columns)}')
    print(f'Number of continues fields are {len(data.select_dtypes(["int64"]).columns)}')
    return data_summary

def adjust_data_types(data):
    number_continues_fields = len(data.select_dtypes(['int64']).columns)
    if number_continues_fields >= 6:
        data = data.astype({'detailed_industry_code':str,'detailed_occupation_code':str , 'own_business_or_self_employed':str,
           'veterans_benefits':str, 'weeks_worked_in_year':str, 'year' :str}, copy = False)
        print(f'Correction of the data type of the fields is done! The number of continues fields are {len(data.select_dtypes(["int64"]).columns)}')
        return data
    else:
        print(f'The number of continues fields is {len(data.select_dtypes(["int64"]).columns)}')
        
def find_placeholders_with_percentages(data):
    '''it takes a df and return the name of the column, the placeholder and the percentage '''
    for col in data.columns:
        for val, percentage in data[col].value_counts(normalize = True).iteritems():
            if val in[' ?', ' NA', ' na', ' No value']: 
                print(f'The {col} field contains this placeholder{val} with a percentage of {percentage} % ')
                
def remove_duplicated_values(data, type_of_dataset):
    
    number_duplicated_values = len(data[data.duplicated(keep="first")])
    print(f'Percentage of duplicated values in the {type_of_dataset} is {number_duplicated_values/data.shape[0]}')

    if number_duplicated_values > 0:
        data.drop_duplicates(keep='first', inplace = True)
        print(f'The {type_of_dataset} dataset after removing some dupplicated values, contains {data.shape[0]} rows and {data.shape[1]} fields after removing duplicated values')
     
     
def box_plot_continues_fields(input_value, my_title):
    fig = px.box(input_value, title= my_title)
    fig.update_layout(autosize=False,width=800,height=800)
    fig.show()
   
        
def histogram_plot_continues_or_categorical_columns(data, continues):
    
    if continues == True:
        continues_columns = data.select_dtypes(['int64']).columns

        for col in continues_columns:

            if col == 'age' or col == 'num_persons_worked_for_employer':
                fig = px.histogram(data, x=col, color="target", barmode='group',nbins=80,title= f'Figure represention of {col} and the target value',width=800, height=300)
                fig.show()

            elif col == 'wage_per_hour':
                fig = px.histogram(data, x=col, color="target", barmode='group',nbins=80,title= f'Figure represention of {col} and the target value',width=800, height=300)
                fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))
                fig.update_traces(xbins=dict(size=1000))
                fig.update_xaxes(range=[0,10000])
                fig.update_yaxes(range=[0,13000])
                fig.show()

            elif col == 'capital_gains' or col == 'dividends':
                fig = px.histogram(data, x=col, color="target", barmode='group',nbins=80,title= f'Figure represention of {col} and the target value',width=800, height=300)
                fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))
                fig.update_traces(xbins=dict(size=2000))
                fig.update_xaxes(range=[0,100000])
                fig.update_yaxes(range=[0,13000])
                fig.show()


            elif col == 'capital_losses':
                fig = px.histogram(data, x=col, color="target", barmode='group',nbins=80,title= f'Figure represention of {col} and the target value',width=800, height=300)
                fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))
                fig.update_traces(xbins=dict(size=1000))
                fig.update_xaxes(range=[0,6000])
                fig.update_yaxes(range=[0,13000])
                fig.show()
    else:
        categorical_columns = data.select_dtypes(['category', 'object']).columns
        for col in categorical_columns[:-1]: # remove target field
            fig = px.histogram(data, x=col, color="target", barmode='group',title= f'Figure represention of {col} and the target value')
            fig.show()

    

def encoding_categorical_fields(data):
    categorical_columns = (data.select_dtypes(['object']).columns)
    for col in categorical_columns:
        data[col] = pd.factorize(data[col])[0]
    return data

def normalizad_continues_features(data):
    '''Nornalized features '''
    for col in data.select_dtypes(['int64']).columns:
        data[col] = (data[col]- min(data[col]))/ (max(data[col]) - min(data[col]))
    return data

def plot_confusion_matix(x ,y ,classifier):
    """ it plots a confusion matrix """
    cm = confusion_matrix(x, y)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True,fmt='g', ax = ax)
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels') 
    ax.set_title('Confusion Matrix ' + classifier)
    ax.xaxis.set_ticklabels(['Fail', 'Pass']); ax.yaxis.set_ticklabels(['Fail', 'Pass']) ## CHANGE THIS
    
def metrix_classifier(x , y):
    """ calculated the accuracy and print out accuracy and classification report"""
    accuracy = round(metrics.accuracy_score(x, y) * 100 ,2)
    print(f"Accuracy: {accuracy} %")
#     f1_score = round(metrics.f1_score(x , y) * 100 , 2)
#     print("F1 Score: {}".format(f1_score(x, y)))
    print(classification_report(x,  y)) # maybe remove this
    
def classifier(model_name, x_train_set, y_train_set,  x_test_set, y_test_set, classifier_name):
    
    model = model_name()
    model_train = model.fit(x_train_set, y_train_set)
    y_prediction = model.predict(x_test_set)
    metrix_classifier(y_test_set, y_prediction)
    plot_confusion_matix(y_test_set,y_prediction,classifier_name)

    
