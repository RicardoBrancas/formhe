import base64
import pickle
from argparse import ArgumentParser

import argparse_dataclass
import bentoml

from formhe.bento_server import BentoConfig

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("--url", type=str, default="http://localhost:3000")

    subparsers = parser.add_subparsers(parser_class=argparse_dataclass.ArgumentParser, required=True, dest='command')

    load_subparser: argparse_dataclass.ArgumentParser = subparsers.add_parser("load", options_class=BentoConfig)
    subparsers._name_parser_map["unload"] = ArgumentParser()
    subparsers._name_parser_map["exit"] = ArgumentParser()

    args = parser.parse_args()

    with bentoml.SyncHTTPClient(args.url, timeout=6000) as client:
        if args.command == "load":
            client.load(base64.b64encode(pickle.dumps(load_subparser.parse_known_args()[0])).decode())
        elif args.command == "unload":
            client.unload()
        elif args.command == "exit":
            client.exit()
        else:
            raise ValueError("Unknown command")
