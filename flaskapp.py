import random

from flask import Flask, redirect, url_for, render_template, request, session
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from user_code import create_batches, get_pred_stockcode, set_model, prediction_dataframe

app = Flask(__name__)
app.secret_key = "ugly_daughters"


@app.route("/", methods=["POST", "GET"])
def home():
    if "user" in session:
        return render_template("home.html")
    else:
        return redirect(url_for("login"))


@app.route("/prediction", methods=["POST", "GET"])
def prediction():
    if "user" in session:
        pass
    else:
        return redirect(url_for("login"))

    def output_guess(rain, mean_temp, min_temp, max_temp):
        test_me_data = [[rain, mean_temp, min_temp, max_temp]]
        test_me_data = np.c_[test_me_data[:]]

        test_me_data = create_batches(test_me_data, test_data=True)
        model = set_model()
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
        axes.set(xlabel="Most Likely Sold Items", ylabel="Predicted Likely Sales in Percentage Out of 100%")
        plt.table(cellText=top_data.values,
                  colLabels=top_data.columns,
                  rowLabels=['1st', '2nd', '3rd', '4th', '5th'],
                  cellLoc='center', rowLoc='center',
                  loc='bottom', bbox=[0, -0.4, 1, .25])
        fig.canvas.draw()
        image1_name = ('static/img1_' + str(random.randint(0, 100000)) + '.png')
        plt.savefig(image1_name, bbox_inches='tight')
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
        axes.set(xlabel="Least Likely Sold Items", ylabel="Predicted Sales in Percentage Out of Negative 100%")
        fig.canvas.draw()
        image2_name = ('static/img2_' + str(random.randint(0, 100000)) + '.png')
        plt.savefig(image2_name, bbox_inches='tight')
        return image1_name, image2_name

    rain = 7.5
    mean_temp = 120.0
    min_temp = 20.0
    max_temp = 120.0
    if request.method == "POST":
        rain = (request.form.get('rainfall', type=float))
        mean_temp = (request.form.get('mean_temp', type=float) * 10)
        min_temp = (request.form.get('min_temp', type=float) * 10)
        max_temp = (request.form.get('max_temp', type=float) * 10)
    else:
        pass
    img1, img2 = output_guess(rain=rain, mean_temp=mean_temp, min_temp=min_temp, max_temp=max_temp)
    weather_input = ("The following graphs are using the inputted parameters of: Rainfall -> " + str(rain) +
                     ", Mean Temperature -> " + str(mean_temp/10) +
                     ", Minimum Temperature -> " + str(min_temp/10) +
                     ", Maximum Temperature -> " + str(max_temp/10))
    return render_template("prediction.html", image1_name=img1, image2_name=img2, content=weather_input)


@app.route("/login", methods=["POST", "GET"])
def login():
    message = ""
    if request.method == "POST":
        user = request.form["username"]
        password = request.form["password"]
        if user == "user" and password == "password":
            session["user"] = user
            session["password"] = password
            return redirect(url_for("home"))
        else:
            session["message"] = "User/Password Combination was not found, please try again."
            message = session["message"]
            return render_template("login.html", content=message)
    else:
        return render_template("login.html")


@app.route("/logout")
def logout():
    session.pop("user", None)
    session.pop("password", None)
    return redirect(url_for("login"))


@app.route("/historical")
def historical():
    if "user" in session:
        pass
    else:
        return redirect(url_for("login"))
    return render_template("historical.html")


if __name__ == "__main__":
    app.run(debug=True)
