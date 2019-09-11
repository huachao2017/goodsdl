
import urllib.request
import logging
import demjson
logger = logging.getLogger("detect")
def http_post(url_api, post_data):
    logger.info("http request_params:" + str(post_data['trace_id']))
    try:
        if post_data is not None:
                data = urllib.parse.urlencode(post_data).encode('utf-8')
                request = urllib.request.Request(url_api,data)
                reponse = urllib.request.urlopen(request)
                result = reponse.read().decode('utf-8')
                reponse.close()
                return (str(result))
    except Exception as err:
        logging.error(err)
    return None

def parse_reponse_dict(result):
    if result == None:
        logger.error("http reponse None , check!" )
        return None
    jsonResult = dict(demjson.decode(str(result)))
    error = dict(demjson.decode(jsonResult['error']))
    returnCode = int(error['returnCode'])
    if returnCode == 0:
        data = dict(demjson.decode(jsonResult['data']))
        return data
    logger.error("http reponse failed , check!,returnCode=%d"%(returnCode))
    return None

def parse_reponse_list(result):
    if result == None:
        logger.error("http reponse None , check!" )
    jsonResult = dict(demjson.decode(str(result)))
    error = dict(demjson.decode(jsonResult['error']))
    returnCode = int(error['returnCode'])
    if returnCode == 0:
        data = list(demjson.decode(jsonResult['data']))
        return data
    logger.error("http reponse failed , check!,returnCode=%d"%(returnCode))
    return None

def parse_reponse_none(result):
    if result == None:
        logger.error("http reponse None , check!" )
    jsonResult = dict(demjson.decode(str(result)))
    error = dict(demjson.decode(jsonResult['error']))
    returnCode = int(error['returnCode'])
    if returnCode == 0:
        return 0
    logger.error("http reponse failed , check!,returnCode=%d"%(returnCode))
    return None