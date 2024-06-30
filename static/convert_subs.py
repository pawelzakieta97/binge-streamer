f = 'sub.srt'
with open(f, 'r') as file:
    text = file.read()
with open('sub.vtt', 'w+') as file:
    file.write(text.replace(',','.'))