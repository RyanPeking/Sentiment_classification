# -*- coding: utf-8 -*-
from bottle import *
import json
from predict import predict
import os
import logging

logging.basicConfig(level=logging.WARNING,
                    filename=os.path.join(os.path.abspath('./'), 'log', 'log.txt'),
                    filemode='w',
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

# 跨域
@route('/<:re:.*>', method='OPTIONS')
def enable_cors_generic_route():
  add_cors_headers()
@hook('after_request')
def enable_cors_after_request_hook():
  """
  This executes after every route. We use it to attach CORS headers when applicable.
  """
  add_cors_headers()
def add_cors_headers():
  response.set_header('Access-Control-Allow-Origin', '*')
  response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, OPTIONS'
  response.headers['Access-Control-Allow-Headers'] = 'Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token'

# summary
@route('/comments', method=['POST'])
def comment():
  text = request.json['text']
  arr = predict(text)
  response.headers['Content-type'] = 'application/json'
  res = {
    'good': [],
    'middle': [],
    'bad': []
  }
  for i in arr:
    for comment in i[2]:
      if comment[0] == 0 and comment[1] is not None:
        res['good'].append({
          'type': i[0],
          'comment': comment[1]
        })
      elif comment[0] == 1 and comment[1] is not None:
        res['middle'].append({
          'type': i[0],
          'comment': comment[1]
        })
      elif comment[0] == 2 and comment[1] is not None:
        res['bad'].append({
          'type': i[0],
          'comment': comment[1]
        })
  return json.dumps(res)


# 静态文件
@route('/static/img/<filename>')
def send_image(filename):
  return static_file(filename, root='./dist/static/img/')
@route('/static/css/<filename>')
def send_css(filename):
  return static_file(filename, root='./dist/static/css/')
@route('/static/js/<filename>')
def send_js(filename):
  return static_file(filename, root='./dist/static/js/')
@route('/static/<filename>')
def send_urlconfig(filename):
  return static_file(filename, root='./dist/static/')
@route('/static/fonts/<filename>')
def send_fonts(filename):
  return static_file(filename, root='./dist/static/fonts/')
@route('/favicon.ico')
def send_ico():
  return static_file('favicon.ico', root='./')


@route('/')
def index():
  return template('./dist/index.html')

run(host='127.0.0.1', port=1234, debug=True)
