import websockets
import asyncio

async def hello():
    uri = "ws://192.168.116.165:8000"
    async with websockets.connect(uri) as websocket:
        name = input("당신의 이름을 입력하세요: ")
        await websocket.send(name)
        print(f"> {name}")

        greeting = await websocket.recv()
        print(f"< {greeting}")

asyncio.run(hello())