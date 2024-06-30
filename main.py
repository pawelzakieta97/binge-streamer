from flask import Flask, render_template
import os

app = Flask('name', static_folder='static', template_folder='templates')

root_dir = 'static'

@app.route('/browse/<path:path>')
def browse(path):
    parent = os.path.split(path)[0]
    if not parent:
        parent = '.'
    elements = os.listdir(os.path.join(root_dir, path))
    files = [e for e in elements if os.path.isfile(os.path.join(root_dir, path, e))]
    directories = [e for e in elements if e not in files]
    return render_template('browse.html', path=path, files=files, parent=parent,
                           directories=directories)


@app.route('/browse/')
def index():
    path = root_dir
    elements = os.listdir(path)
    files = [e for e in elements if os.path.isfile(os.path.join(path, e))]
    directories = [e for e in elements if e not in files]
    return render_template('browse.html', path='.', files=files, parent=None, directories=directories)


@app.route('/watch/<path:video_file>')
def watch(video_file):
    return render_template('watch.html', filename=video_file)


if __name__ == '__main__':
    app.run("0.0.0.0")