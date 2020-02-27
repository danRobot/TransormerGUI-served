#!/usr/bin/env python

# WS client example

import asyncio
import websockets

async def hello(name):
    uri = "ws://localhost:6006"
    async with websockets.connect(uri) as websocket:
        

        await websocket.send(name)
        #print(f"> {name}")

        greeting = await websocket.recv()
    return greeting

#name = input("What's your name? ")
#txt=asyncio.get_event_loop().run_until_complete(hello(name))
#print(txt)
