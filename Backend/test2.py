import websockets
import asyncio
import socket

async def hello(websocket, path):
    name = await websocket.recv()
    print(f"< {name}")

    greeting = f"안녕하세요, {name}!"
    await websocket.send(greeting)
    print(f"> {greeting}")

# 호스트 이름을 IP 주소로 변환
host_ip = socket.gethostbyname(socket.gethostname())
print("호스트 ip :", host_ip)

start_server = websockets.serve(hello, host_ip, 8000)

asyncio.get_event_loop().run_until_complete(start_server)
print("서버 실행완료")
asyncio.get_event_loop().run_forever()