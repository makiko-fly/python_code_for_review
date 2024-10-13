# -*- coding: utf-8 -*-

import argh
import base64
import json
import requests
import logging
import os
import glob
import six

logger = logging.getLogger("UNOTIFY")
logger.setLevel(logging.DEBUG)

UNOTIFY_SERVER = "http://unotify:6000"
UNOTIFY_API_URL = UNOTIFY_SERVER + "/api/v1.0/synctasks"

# CHANNEL ALIAS FOR Lark
LARK_CHANNEL_ALIAS = {
    "dev": "oc_2f30dcd995d82fd401662ffebc9f63a0",
    "data": "oc_95ecedb8bad34e9225aca95e2eb1945d"
}


def send_slack(msg=None, channel=None, *args, **kwargs):
    body = {
        "task_type": "slack",
        "task_data": {
            "msg": msg,
            "channel": channel,
        }
    }
    _send_msg(body)


def send_mail(subject=None, channel=None, msg=None, attachments=None, *args, **kwargs):
    '''
    发送邮件
    :param subject:
    :param channel:
    :param msg:
    :param attachments: 附件文件地址，支持通配符路径，如 /data/hftdata/data_v1/hftdata/WIND/*,/data/hftdata/data_v1/hftdata/ajzq-sf/*
    :param args:
    :param kwargs:
    :return:
    '''
    # fix address.
    to_address = ",".join(
        [address if address.endswith("@ruitiancapital.com") else address + "@ruitiancapital.com" for address in
         channel.split(",")])

    attachment_body = []
    if attachments is not None:
        files = attachments.split(",")
        for file in files:
            file = file.strip()
            if file == "":
                continue
            for subfile in glob.iglob(file):
                filename = subfile.split("/")[-1]
                filedata = _read_file_data(subfile)
                attachment_body.append({
                    "filename": filename,
                    "filedata": filedata
                })

    body = {
        "task_type": "mail",
        "task_data": {
            "from_address": "dataman@ruitiancapital.com",
            "to_address": to_address,
            "password": "Ruitian@8G",
            "subject": subject,
            "msg": '' if msg is None else msg.replace("\n", "<br>"),
            "channel": channel,
            "attachment": attachment_body
        }
    }
    _send_msg(body)


def send_lark(msg=None, channel=None, *args, **kwargs):
    body = {
        "task_type": "lark",
        "task_data": {
            "msg": msg,
            "channel": LARK_CHANNEL_ALIAS.get(channel, channel)
        }
    }
    _send_msg(body)


def send_notify(task_type=None, config_name=None, *args, **kwargs):
    task_types = task_type.split(",")
    if len(task_types) > 1:
        for single_type in task_types:
            send_notify(*args, task_type=single_type, config_name=config_name, **kwargs)
        return

    if task_type == "lark":
        send_lark(*args, **kwargs)
    elif task_type == "mail":
        send_mail(*args, **kwargs)
    elif task_type == "slack":
        send_slack(*args, **kwargs)
    else:
        print("Unsupport task type[%s] yet." % task_type)


def _read_file_data(file):
    if os.path.isdir(file):
        raise NotImplemented("暂不支持发送整个文件目录")
    with open(file, 'rb') as f:
        att_str = f.read()
    att_str_base64 = base64.b64encode(att_str)
    if six.PY2:
        return att_str_base64.encode('utf8')
    return str(att_str_base64, encoding='utf8')


def _send_msg(body):
    headers = {
        'Content-Type': 'application/json'
    }
    logger.info("Send unotify server request.")
    response = requests.request("POST", UNOTIFY_API_URL, headers=headers, data=json.dumps(body))
    response_data = json.loads(response.text.encode('utf8'))
    print("Receive unotify server response: %s" % response_data)
    if response_data.get("code", 0) != 0:
        raise Exception(response_data.get("message", "Unknown exception"))


if __name__ == "__main__":
    argh.dispatch_commands([send_notify, send_lark, send_mail, send_slack])
