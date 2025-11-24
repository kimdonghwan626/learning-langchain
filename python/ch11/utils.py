import json

def printLog(msg):
    print('@@@@@ ' + str(msg))
    
def pretty_print(obj):
    printLog(json.dumps(obj.dict(), indent=2, ensure_ascii=False))