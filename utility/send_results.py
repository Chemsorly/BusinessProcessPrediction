import requests

#http://docs.python-requests.org/en/latest/user/quickstart/#post-a-multipart-encoded-file

def SendResultFiles(args, files):
    url = args['target_host']
    for file in files:
        try:
            fin = open(file, 'rb')
            files = {'file': fin}
            headers = {'authtoken': 'thisisatoken'}
            r = requests.post(url, files=files, headers=headers)
            print(r.text)
        finally:
            fin.close()