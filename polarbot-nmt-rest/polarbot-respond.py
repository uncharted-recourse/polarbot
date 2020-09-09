from __future__ import division

from gunicorn_app import GunicornApplication
from flask import Flask, request, jsonify

import numpy as np
import argparse



import onmt
import onmt.Markdown
import onmt.IO
import torch
import argparse
import math
import codecs
import os
import sys

from nltk.tokenize import TweetTokenizer
from sacremoses import MosesDetokenizer


app = Flask(__name__)
args = None
translator = None

dt = MosesDetokenizer()

tt = TweetTokenizer()


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/api/translate", methods=['POST'])
def score():
    line = request.get_json()['text']
    srcTokens = tt.tokenize(line.lower())
    srcBatch = [srcTokens]
    predBatch, predScore, goldScore, attn, src = translator.translate(srcBatch, None)
    
    response = predBatch[0][0]

    response = dt.detokenize(response, return_str=True)
    scores = list(predScore[0])
    result = {"score": predScore[0][0],
              "response": response,
              "version": args.version,
              "model": os.path.basename(args.model)}
    return jsonify(result), 200


def parse_arguments():
    parser = argparse.ArgumentParser(description='live_translate.py')
    onmt.Markdown.add_md_help_argument(parser)

    parser.add_argument('-m','--model', required=True,
                        help='Path to model .pt file')

    parser.add_argument('--replace_unk', action="store_true",
                        help="""Replace the generated UNK tokens with the source
                        token that had highest attention weight. If phrase_table
                        is provided, it will lookup the identified source token and
                        give the corresponding target token. If it is not provided
                        (or the identified source token does not exist in the
                        table) then it will copy the source token""")
    parser.add_argument('-verbose', action="store_true",
                        help='Print scores and predictions for each sentence')

    parser.add_argument('-dump_beam', type=str, default="",
                        help='File to dump beam information to.')

    parser.add_argument('-n_best', type=int, default=5,
                        help="""If verbose is set, will output the n_best
                        decoded sentences""")

    parser.add_argument('-gpu', type=int, default=-1,
                        help="Device to run on")
    # unused ONMT args:
    parser.add_argument('-src',   required=False,
                        help='Source sequence to decode (one line per sequence)')
    parser.add_argument('-src_img_dir',   default="",
                        help='Source image directory')
    parser.add_argument('-tgt',
                        help='True target sequence (optional)')
    parser.add_argument('-output', default='pred.txt',
                        help="""Path to output the predictions (each line will
                        be the decoded sequence""")
    parser.add_argument('-beam_size',  type=int, default=5,
                        help='Beam size')
    parser.add_argument('-batch_size', type=int, default=30,
                        help='Batch size')
    parser.add_argument('-attn_debug', action="store_true",
                        help='Print best attn for each word')
    parser.add_argument('-max_sent_length', type=int, default=100,
                        help='Maximum sentence length.')
    # original API args:
    parser.add_argument('-v', '--version', help="Version string of models", default="alpha")
    parser.add_argument('-l', '--listen-host', help="Listen host", default="0.0.0.0:5000")
    parser.add_argument('-d', '--debug', help="Run in debug mode", action="store_true")
    parser.add_argument('-t', '--threads', help="Gunicorn threads", type=int, default=1)
    parser.add_argument('-w', '--workers', help="Gunicorn workers", type=int, default=1)
    parser.add_argument('-s', '--statsd-host', help="Statsd host", type=str)
    parser.add_argument('-p', '--prefix-statsd', help="Statsd prefix", type=str)
    return parser.parse_args()


def main():
    global translator
    global args

    args = parse_arguments()
    
    print("Loading responder model.")
    print(args.model)
    if os.path.isfile(args.model):
        print("model file exists")
    else:
        print("OH NO MODEL FILE DOESN'T EXIST")
    

    args.cuda = args.gpu > -1
    if args.cuda:
        torch.cuda.set_device(args.gpu)

    translator = onmt.Translator(args)

    print("Starting service.")
    hostport = args.listen_host.split(':')
    if len(hostport) == 2:
        port = int(hostport[1])
    else:
        port = 5000
    options = {
        'bind': '{}:{}'.format(hostport[0], port),
        'threads': args.threads,
        'workers': args.workers
    }
    if args.statsd_host:
        options['statsd_host'] = args.statsd_host
    if args.prefix_statsd:
        options['statsd_prefix'] = args.prefix_statsd
    if args.debug:
        options['log_level'] = 'DEBUG'
    GunicornApplication(app, options).run()

if __name__ == "__main__":
    main()
