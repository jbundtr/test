import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('stock_codes.csv')
stockcode_description = pd.read_csv('stock_code_to_desc.csv')

Y = data['StockCode']

unique_stockcodes = np.unique(Y)
Y[0] == unique_stockcodes
boolean_stockcodes = [StockCode == unique_stockcodes for StockCode in Y]

y = boolean_stockcodes


def create_batches(X, y=None, batch_size=32, valid_data=False, test_data=False):
    if test_data:
        data = tf.data.Dataset.from_tensor_slices(tf.constant(X))
        data_batch = data.batch(batch_size)
        return data_batch
    elif valid_data:
        data = tf.data.Dataset.from_tensor_slices((tf.constant(X),
                                                  tf.constant(y)))
        data_batch = data.batch(batch_size)
        return data_batch
    else:
        data = tf.data.Dataset.from_tensor_slices((tf.constant(X),
                                                 tf.constant(y)))
        data = data.shuffle(buffer_size=len(X))
        data_batch = data.batch(batch_size)
        return data_batch


def set_model():
    model = tf.keras.models.load_model('best_model.h5')
    return model


def get_pred_stockcode(prediction_matrix):
    return unique_stockcodes[np.argmax(prediction_matrix)]


def getKey(item):
    return item[0]


def prediction_dataframe(prob_matrix, top_five=True):
    prob_matrix = prob_matrix[0]
    matrix = np.array((prob_matrix, unique_stockcodes))
    matrix = matrix.T
    testing = tuple(map(tuple, matrix))
    if top_five:
        testing_2 = sorted(testing, key=getKey, reverse=True)
    else:
        testing_2 = sorted(testing, key=getKey, reverse=False)
    testing_3 = testing_2[:5]
    data = pd.DataFrame(testing_3, columns=['Prediction', 'Item Code'])
    data2 = data['Item Code']
    nparray = [[0, 0]]
    for x in range(5):
        guess_frame = stockcode_description[stockcode_description['StockCode'].str.contains(str(data2[x]))].values
        nparray = np.append(nparray, (guess_frame[0][0], guess_frame[0][1]))
    testing_4 = nparray[2:]
    data2 = np.reshape(testing_4, (5, 2))
    data2 = pd.DataFrame(data2, columns=['Item Code', 'Description'])
    newest = pd.concat([data, data2[['Description']]], axis=1, join="inner")
    scaled_prediction = newest.copy()

    if top_five:
        scaling = scaled_prediction['Prediction'].transform(lambda x: x / x.sum())
    else:
        scaling = scaled_prediction['Prediction'].transform(lambda x: (x / x.sum() - 1))

    col_name = ['Prediction']
    scaled_prediction[col_name] = scaling.fillna(-1)
    return scaled_prediction


def output_guess(rain, mean_temp, min_temp, max_temp):
    test_me_data = [[rain, mean_temp, min_temp, max_temp]]
    test_me_data = np.c_[test_me_data[:]]

    test_me_data = create_batches(test_me_data, test_data=True)
    attempt = model.predict(test_me_data, verbose=0)
    guess = get_pred_stockcode(attempt[0])
    guess = str(guess)

    stockcode_description = pd.read_csv('stock_code_to_desc.csv')

    guess_frame = stockcode_description[stockcode_description['StockCode'].str.contains(str(guess))]

    guess_numpy = guess_frame.to_numpy()
    stock, desc = guess_numpy[0][0], guess_numpy[0][1]
    fig, axes = plt.subplots(1, 1, figsize=(16, 10))
    fig.suptitle("Top 5 Most Likely Items to Sell")
    top_data = prediction_dataframe(attempt, top_five=True)
    top_data.update(top_data[['Prediction']].astype(float))
    top_data.update(top_data[['Prediction']].applymap(lambda x: round(x, 2)))
    sns.barplot(
        ax=axes,
        data=top_data,
        x="Item Code",
        y="Prediction"
    )
    top_data.update(top_data[['Prediction']].applymap(lambda x: (str(x * 100)[:4] + "%")))
    plt.table(cellText=top_data.values,
              colLabels=top_data.columns,
              rowLabels=['1st', '2nd', '3rd', '4th', '5th'],
              cellLoc='center', rowLoc='center',
              loc='bottom', bbox=[0, -0.4, 1, .25])
    axes.set(xlabel="Most Likely Sold Items", ylabel="Predicted Likely Sales in Percentage Out of 100%")
    fig.savefig('img1.png')
    # Image(filename='img1.png')
    bottom_data = prediction_dataframe(attempt, top_five=False)
    fig, axes = plt.subplots(1, 1, figsize=(16, 10))
    fig.suptitle("Bottom 5 Items - Least Likely to Sell")
    bottom_data.update(bottom_data[['Prediction']].astype(float))
    bottom_data.update(bottom_data[['Prediction']].applymap(lambda x: round(x, 2)))
    sns.barplot(
        ax=axes,
        data=bottom_data,
        x="Item Code",
        y="Prediction"
    )
    bottom_data.update(bottom_data[['Prediction']].applymap(lambda x: (str(x * 100)[:5] + "%")))
    plt.table(cellText=bottom_data.values,
              colLabels=bottom_data.columns,
              rowLabels=['1st', '2nd', '3rd', '4th', '5th'],
              cellLoc='center', rowLoc='center',
              loc='bottom', bbox=[0, -0.4, 1, .25])
    axes.set(xlabel="Least Likely Sold Items", ylabel="Predicted Sales in Percentage Out of -100%")
    plt.savefig('img2.png')
    # Image(filename='img2.png')
    #     display(top_data)
    #     print("Bottom Five Items")
    #     sns.barplot(
    #         ax=axes[0],
    #         data=bottom_data,
    #         x="Item Code",
    #         y="Prediction"
    #     )
    #     axes[0].set(xlabel="Least Likely Sold Items", ylabel="Predicted Lack of Sales Out of -100%")
    #     display(bottom_data)

    # return stock, desc, attempt
    return None


