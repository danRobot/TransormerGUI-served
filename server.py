from transgenerator.run import Generator

import asyncio
import websockets
import argparse
import os
from time import sleep

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_type",
    default="gpt2-large",
    type=str,
    required=True,
    help="Model type selected in the list: ",
)
parser.add_argument(
    "--model_name_or_path",
    default="gpt2-large",
    type=str,
    required=True,
    help="Path to pre-trained model or shortcut name selected in the list: " ,
)

parser.add_argument("--prompt", type=str, default="")
parser.add_argument("--wpath", type=str, default="model")
parser.add_argument("--length", type=int, default=20)
parser.add_argument("--stop_token", type=str, default=None, help="Token at which text generation is stopped")

parser.add_argument(
    "--temperature",
    type=float,
    default=1.0,
    help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
)
parser.add_argument(
    "--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2"
)
parser.add_argument("--k", type=int, default=0)
parser.add_argument("--p", type=float, default=0.9)

parser.add_argument("--padding_text", type=str, default="", help="Padding text for Transfo-XL and XLNet.")
parser.add_argument("--xlm_language", type=str, default="", help="Optional language when used with the XLM model.")

parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")

args = parser.parse_args()


gen=Generator(args)


async def hello(websocket, path):
    prompt = await websocket.recv()
    salida=gen.generate(prompt,temperature=args.temperature,top_k=args.k,top_p=args.p)
    await websocket.send(salida)

start_server = websockets.serve(hello, "0.0.0.0",6006)

os.systen('./ngrok http 6006 &')
sleep(120)
print('ready')

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()

