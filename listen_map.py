import socket
import struct

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

plt.ion()
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("192.168.128.213", 1807))
server.listen(0)


def axes_base(axes: plt.Axes):
    axes.set_ylim((-500, 6_000))
    axes.set_xlim((-2000, 2000))
    axes.set_xlabel("Width")
    axes.set_ylabel("Depth")

    axes.set_aspect("equal")
    axes.scatter(0, 0, marker="o", c="red", s=10)
    axes.add_artist(Circle((0, -225), 225, fill=True, color="black"))


try:
    while True:
        # accept connections from outside
        client_socket, client_address = server.accept()
        print(f"Accepted connection from {client_address[0]}:{client_address[1]}")

        fig, ax = plt.subplots()
        fig: plt.Figure
        axes_base(ax)
        plt.pause(0.1)
        done = False
        while not done:
            request = client_socket.recv(128)
            if not request:
                print("Empty response, closed")
                break

            if request[-6:] == b"!close":
                done = True
                request = request[:-6]
                if not request:
                    break

            plt.cla()
            axes_base(ax)
            boxes: dict[int, list[tuple[float, float]]] = {}
            for i in range(len(request) // 17):
                name, x, y = struct.unpack("<Bdd", request[17 * i : 17 * (i + 1)])
                boxes.setdefault(name, []).append((x, y))

            for name, coords in boxes.items():
                if len(coords) == 1:
                    (x, y) = coords[0]
                else:
                    ((x1, y1), (x2, y2)) = coords
                    x = (coords[0][0] + coords[1][0]) // 2
                    y = (coords[0][1] + coords[1][1]) // 2
                ax.add_artist(Circle((x, y), 200, fill=False, color="green"))
                ax.text(x - 70, y - 70, name, ma="center")

            plt.pause(0.01)

        plt.close("all")
        # plt.pause(0.01)
except KeyboardInterrupt:
    pass
finally:
    server.close()
