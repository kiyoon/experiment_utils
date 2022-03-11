#!/usr/bin/env python3


# For command line usage, include the token and chat ids in advance in the script.
# If you want to use only the functions, there's no need to include this.
"""
telegram_token = ""
telegram_chat_ids = [""]
"""

# For command line usage, include the token and chat ids in the INI file.
# If you want to use only the functions, there's no need to include this.
# The file format has to be
#
# [Telegram]
# token = ABCDEF
# chat_ids = 1234567,2345678
# 
"""
import os
import configparser
_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
config_ini_path = os.path.join(_SCRIPT_DIR, 'telegram_key.ini')
"""

try:
    from PIL import Image
except ImportError:
    # backward compatibility for the ones who don't need sending numpy photos
    pass

import requests
import io
import argparse
def get_parser():
    parser = argparse.ArgumentParser(description="Send Telegram message",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--title", help="Title")
    parser.add_argument("--body", help="Body", required=True)
    return parser


def send_text(telegram_token, chat_id, text, parse_mode=None):
    telegram_request_url = "https://api.telegram.org/bot{0}/sendMessage".format(telegram_token)
    return requests.post(telegram_request_url, data={
        'chat_id': chat_id,
        'text': text,
        'parse_mode': parse_mode
        })

def send_text_with_title(telegram_token, chat_id, title, body):
    if title:
        text = '<b>' + title + '</b>\n\n' + body
        tg_request = send_text(telegram_token, chat_id, text, parse_mode = 'HTML')
        if not tg_request.ok:
            # If failed, try send normal text instead of HTML parsing
            text = title + '\n\n' + body
            tg_request = send_text(telegram_token, chat_id, text, parse_mode = None)
    else:
        text = body
        tg_request = send_text(telegram_token, chat_id, text, parse_mode = None)

    return tg_request


def _send_photo_bytes(telegram_token, chat_id, bytes_io, caption=None):
    """Send photo in open() or io.BytesIO form.
    """
    url = "https://api.telegram.org/bot{}/sendPhoto".format(telegram_token);
    files = {'photo': bytes_io}
    data = {'chat_id' : chat_id}
    if caption is not None:
        data['caption'] = caption
    r= requests.post(url, files=files, data=data)
    return r


def send_photo(telegram_token, chat_id, img_path, caption=None):
    photo = open(img_path, 'rb')
    return _send_photo_bytes(telegram_token, chat_id, photo, caption)

def send_numpy_photo(telegram_token, chat_id, numpy_photo, caption=None):
    image = Image.fromarray(numpy_photo)
    photo = io.BytesIO()
    image.save(photo, format='jpeg')
    photo.seek(0)       # to start reading from the beginning. (After writing, the cursor is at the end)
    photo.name = 'img.jpg'
    return _send_photo_bytes(telegram_token, chat_id, photo, caption)

def send_remote_photo(telegram_token, chat_id, img_url, caption):
    remote_image = requests.get(img_url)
    photo = io.BytesIO(remote_image.content)
    photo.name = 'img.png'
    return _send_photo_bytes(telegram_token, chat_id, photo, caption)


def send_matplotlib_fig(telegram_token, chat_id, fig, caption=None):
    photo = io.BytesIO()
    fig.savefig(photo, format='png')
    photo.seek(0)       # to start reading from the beginning. (After writing, the cursor is at the end)
    photo.name = 'img.png'
    return _send_photo_bytes(telegram_token, chat_id, photo, caption)

def _send_gif_bytes(telegram_token, chat_id, bytes_io, caption=None):
    """Send GIF in open() or io.BytesIO form.
    """
    url = "https://api.telegram.org/bot{}/sendAnimation".format(telegram_token);
    files = {'animation': bytes_io}
    data = {'chat_id' : chat_id}
    if caption is not None:
        data['caption'] = caption

    r= requests.post(url, files=files, data=data)
    return r

def send_numpy_video_as_gif(telegram_token, chat_id, video, caption=None, optimize=True, duration=100, loop=0 ):
    # T, H, W, C = video.shape
    gif_frames = []
    for gif_frame in video:
        gif_frames.append(Image.fromarray(gif_frame))

    gif_bytes = io.BytesIO()
    gif_frames[0].save(gif_bytes, format='gif', save_all=True, append_images=gif_frames[1:], optimize=optimize, duration=duration, loop=loop)    
    gif_bytes.seek(0)       # to start reading from the beginning. (After writing, the cursor is at the end)
    gif_bytes.name = 'img.gif'

    return _send_gif_bytes(telegram_token, chat_id, gif_bytes, caption)

def send_document(telegram_token, chat_id, file_path):
    file_to_send = open(file_path, 'rb')
    url = "https://api.telegram.org/bot{}/sendDocument".format(telegram_token);
    files = {'document': file_to_send}
    data = {'chat_id': chat_id}
    r= requests.post(url, files=files, data=data)
    return r

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    if hasattr('telegram_token'):
        for chat_id in telegram_chat_ids:
            print(send_text_with_title(telegram_token, chat_id, args.title, args.body))
    elif hasattr('config_ini_path'):
        config = configparser.ConfigParser()
        config.read(config_ini_path)
        chat_ids = config['Telegram']['chat_ids'].split(',')
        for chat_id in chat_ids:
            print(send_text_with_title(config['Telegram']['token'], chat_id, args.title, args.body))
    else:
        raise ValueError('None of the Telegram token is specified but trying to send a message. Edit this file.')
