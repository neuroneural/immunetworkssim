import requests
import json
from datetime import datetime

class distributed():
  def refresh(url, refresh, last_auth):
    if (datetime.now()-last_auth).total_seconds() / 60<=30 and 25<=(datetime.now()-last_auth).total_seconds() / 60<=30:
      data = {
        "request": "REFRESH",
        "REFRESH_TOKEN": refresh
      }
      json_data = json.dumps(data)
      headers = {'Content-Type': 'application/json'}
      return requests.post(url+'/login', data=json_data, headers=headers).json()
    return 0
      
  def login(url,username, password):
    data = {
      "username":username,
      "password":password,
      "request": "LOGIN"
    }
    json_data = json.dumps(data)
    headers = {'Content-Type': 'application/json'}
    return requests.post(url+'/login', data=json_data, headers=headers).json()


  def get_user_runs(url, keys):
    data = {}
    json_data = json.dumps(data)
    headers = {'Content-Type': 'application/json','Authorization':keys}
    return requests.post(url+'/runs', data=json_data, headers=headers).json()

  def activate_user(url, runid, keys):
    data = {
      "activate": 1,
      "runid": int(runid)  
    }
    json_data = json.dumps(data)
    headers = {'Content-Type': 'application/json','Authorization':keys}
    requests.post(url+'/activate', data=json_data, headers=headers)
  
  def deactivate_user(url,runid, keys):
    data = {
      "activate": 0,
      "runid": int(runid)  
    }
    json_data = json.dumps(data)
    headers = {'Content-Type': 'application/json','Authorization':keys}
    requests.post(url+'/activate', data=json_data, headers=headers)

  def upload_gradients(url, runid, keys, gradients):
    data = {
      "runid": str(runid),
      "gradients":  gradients
    }
    json_data = json.dumps(data)
    headers = {'Content-Type': 'application/json','Authorization':keys}
    requests.post(url+'/results', data=json_data, headers=headers)

  def get_gradients(url, runid, keys):
    data = {
      "runid": str(runid),
    }
    json_data = json.dumps(data)
    headers = {'Content-Type': 'application/json','Authorization':keys}
    return requests.post(url+'/agggrad', data=json_data, headers=headers).json()