from flask import Flask , render_template , request 
from agent_langchain import run_agent


app = Flask(__name__)


Global_location = "Kolhapur"
Global_weather_data = None


@app.route("/assistant", methods=["GET", "POST"])
def assistant():

    response = None

    if request.method == "POST":

        user_input = request.form["query"]

        response = run_agent(user_input)

    return render_template(
        "assistant.html",
        response=response
    )


if __name__ == '__main__':
    app.run(debug=True)
    
