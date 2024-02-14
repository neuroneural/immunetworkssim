#main.py

from flask import Flask, render_template, request, redirect, url_for
from modules.dist import distributed
from modules.computationdb import Computation as comp
from modules.meshnet_train import training as dsttrain
from datetime import datetime
from scripts import meshnet, dice, loader
import plotly.graph_objs as go
import threading



# keys = None
# last_auth=None
# refresh=None
comp.__init__()

app = Flask(__name__)

app.config['global_variables'] = {
    'keys': None,
    'last_auth': None,
    'refresh': None,
    'url': 'https://c813ur0lif.execute-api.us-east-1.amazonaws.com/dev'
}



def call_training(run_id):
    training =  dsttrain(
        url = app.config['global_variables']['url'],
        meshnet=meshnet, 
        comp = comp, 
        runid = run_id,
        dice=dice, 
        loader=loader,
        modelAE='./scripts/modelAE.json',
        dbfile='./scripts/mindboggle.db', 
        dist =distributed,
        classes=3, 
        epochs = 1, 
        cubes=1, 
        label = 'GWlabels', 
        keys = app.config['global_variables']['keys'], 
        last_auth=app.config['global_variables']['last_auth'],
        refresh = app.config['global_variables']['refresh'])
    training.train_f()

def refresh_auth():
    ref = distributed.refresh(app.config['global_variables']['keys'], app.config['global_variables']['last_auth'])
    if ref:
      app.config['global_variables']['keys'] = ref['body']['AuthenticationResult']['IdToken']
      app.config['global_variables']['last_auth'] = datetime.now()
      app.config['global_variables']['refresh'] = ref['body']['AuthenticationResult']['RefreshToken']



@app.route("/")
def hello():  
    return render_template('home.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login_post():
    username = request.form['username']
    password = request.form['password']
    auth =  distributed.login(app.config['global_variables']['url'],username,password)
    if auth['statusCode']==200:
      print(auth)
      app.config['global_variables']['keys'] = auth['body']['AuthenticationResult']['IdToken']
      app.config['global_variables']['last_auth'] = datetime.now()
      app.config['global_variables']['refresh'] = auth['body']['AuthenticationResult']['RefreshToken']
      return render_template(
            'runs_home.html',
             data = distributed.get_user_runs(
                app.config['global_variables']['url'],
                app.config['global_variables']['keys']))
    else:
      return render_template('login.html', message=auth['body'])

@app.route('/runs_home')
def runs_home():
    if app.config['global_variables']['keys']:
        return render_template(
            'runs_home.html',
             data = distributed.get_user_runs(
                app.config['global_variables']['url'],
                app.config['global_variables']['keys']))
    else:
        return redirect(url_for('login'))


@app.route('/computation')
def computation():
    run_id = request.args.get('run_id')
    return render_template(
        'simulator.html',
        runid = run_id,
        data= distributed.get_user_runs(
            app.config['global_variables']['url'],
            app.config['global_variables']['keys']
            ), 
            status = comp.Simulation_status(int(run_id)))

@app.route('/start_simulation')
def start_simulation():
    run_id = request.args.get('run_id')
    classes = distributed.get_user_runs(
        app.config['global_variables']['url'],
        app.config['global_variables']['keys'])['body'][str(run_id)]['classes']
    comp.start_simulation(runid=int(run_id),classes=int(classes))

    script_thread = threading.Thread(target=call_training, args=run_id)
    script_thread.start()
    return render_template(
        'simulator.html',  runid = run_id, 
        data= distributed.get_user_runs(
            app.config['global_variables']['url'],
            app.config['global_variables']['keys']
        ), 
        status = comp.Simulation_status(int(run_id)))

@app.route('/view_simulation')
def view_simulator():
    runid = request.args.get('run_id')
    fetched_data = comp.fetch_simulation_data(runid)
    print(fetched_data)
    return render_template('view_simulation.html', fetched_data=fetched_data)

if __name__ == "__main__":
  app.run(debug=False, port=5000)
   