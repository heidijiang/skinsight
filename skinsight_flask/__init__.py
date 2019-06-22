from flask import Flask


app = Flask(__name__)
app.secret_key = 'dljsaklqk24e21cjn!Ew@@dsa5'

from skinsight_flask import app_runner
